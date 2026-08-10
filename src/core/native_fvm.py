"""
core/native_fvm.py
==================
NativeFVMSolver: motor propio de volumenes finitos 1D para las Ecuaciones No
Lineales de Agua Poco Profunda, implementando el contrato `Solver`.

ECUACIONES RESUELTAS (1D, promediadas en la vertical)
-----------------------------------------------------
    d_t h  + d_x (h u)                       = 0
    d_t hu + d_x (h u^2 + g h^2 / 2)         = -g h d_x b - (h/rho) d_x P
                                               + tau_x/rho + f h v + h a_fric_x
    d_t hv + d_x (h u v)                     = -f h u + h a_fric_y

con h = zeta - b, b = -h0 (cota de fondo; h0 = batimetria positiva del Domain).

SUPUESTOS NUMERICOS EXPLICITOS (todos configurables via SolverConfig)
---------------------------------------------------------------------
* Malla   : cartesiana uniforme, celdas centradas, dx constante.
* Espacio : volumenes finitos + MUSCL-minmod sobre (eta, u, v, b), orden 2 en
            zonas suaves, orden 1 en extremos. `config.scheme` selecciona el
            solver de Riemann; solo 'hllc' esta implementado.
* Fondo   : reconstruccion hidrostatica de Audusse et al. (2004) => el esquema
            es well-balanced para el lago en reposo sobre batimetria arbitraria.
* Tiempo  : SSPRK3 (Shu-Osher, 3 etapas, TVD).
* CFL     : dt = cfl * dx / max(|u| + sqrt(g h)); config.cfl por defecto 0.45.
* Secado  : h se limita a >= 0; u = 0 donde h <= config.min_depth.
* Presion : gradiente centrado de segundo orden sobre el campo P evaluado en
            centros de celda, incluidas las fantasma.

GANCHO DE ACELERACION: `_rhs` es el 100% del coste. Es puro NumPy sin ramas de
Python, listo para envolver en Numba/JAX. No se optimiza de forma prematura.
"""

from __future__ import annotations

import numpy as np

from constants.titan_params import TITAN, DEFAULT_FLUID, CryogenicFluid
from core.boundary import NG, apply_bc, extend_bathymetry, validate_bc
from core.fvm_kernel import directional_rhs
from core.solver_base import (Domain, ForcingBundle, Solver, SolverConfig,
                              State)

__all__ = ["NativeFVMSolver"]


class NativeFVMSolver(Solver):
    """
    Motor FVM 1D. El estado publico son variables primitivas (zeta, u, v);
    la conversion a conservadas (h, hu, hv) es estrictamente interna (ADR-001).

    Parametros de construccion
    --------------------------
    fluid : propiedades del fluido criogenico (densidad para el forzamiento
            barometrico y el esfuerzo de viento). Por defecto DEFAULT_FLUID.
    g     : gravedad. Por defecto TITAN.g. Se expone solo para permitir tests de
            verificacion numerica pura (p.ej. dam-break con g=9.81 de la
            literatura terrestre); la fisica de Titan siempre usa TITAN.g.
    order : 1 o 2. Orden de la reconstruccion espacial.
    """

    def __init__(self, fluid: CryogenicFluid = DEFAULT_FLUID,
                 g: float | None = None, order: int = 2,
                 limiter: str = "minmod",
                 extra_source=None) -> None:
        if order not in (1, 2):
            raise ValueError("order debe ser 1 o 2")
        self.fluid = fluid
        self.g = TITAN.g if g is None else float(g)
        self.order = order
        self.limiter = limiter
        # Gancho para el metodo de soluciones manufacturadas (MMS): callable
        # (x, t) -> (S_h, S_hu, S_hv). No forma parte del contrato Solver ni de
        # la fisica; existe solo para la compuerta de orden formal.
        self.extra_source = extra_source

        self._domain: Domain | None = None
        self._config: SolverConfig | None = None
        self._forcing: ForcingBundle = ForcingBundle(coriolis_enabled=False)
        self._t = 0.0
        self._dt_last = 0.0
        self._n_steps = 0

    # ------------------------------------------------------------------
    # Contrato: inicializacion
    # ------------------------------------------------------------------
    def initialize(self, domain: Domain, initial_state: State,
                   forcing: ForcingBundle, config: SolverConfig) -> None:
        if config.scheme != "hllc":
            raise NotImplementedError(
                f"scheme={config.scheme!r} no implementado en el motor nativo; "
                "solo 'hllc'.")
        if config.time_integrator != "ssprk3":
            raise NotImplementedError(
                f"time_integrator={config.time_integrator!r} no implementado; "
                "solo 'ssprk3'.")

        x = np.asarray(domain.x, dtype=float).ravel()
        if x.size < 2 * NG + 1:
            raise ValueError("El dominio necesita al menos 5 celdas.")
        dx = np.diff(x)
        if not np.allclose(dx, dx[0], rtol=1e-12, atol=0.0):
            raise ValueError("El motor nativo exige malla uniforme en x.")

        self._domain = domain
        self._config = config
        self._forcing = forcing
        self.nx = x.size
        self.dx = float(dx[0])
        self.x = x
        self._y_scalar = float(np.asarray(domain.y).ravel()[0])
        self._west, self._east = validate_bc(config.bc)

        # Batimetria -> cota de fondo. h0 > 0 en zona liquida  =>  b = -h0.
        h0 = np.asarray(domain.bathymetry, dtype=float).ravel()
        if h0.shape != x.shape:
            raise ValueError("bathymetry y x deben tener la misma forma.")
        self.h0 = h0
        self._b_ext = extend_bathymetry(-h0, self._west, self._east)

        # Coordenadas extendidas (malla uniforme => extrapolacion lineal exacta).
        self._x_ext = x[0] + (np.arange(self.nx + 2 * NG) - NG) * self.dx
        self._y_ext = np.full_like(self._x_ext, self._y_scalar)

        # Estado interno en variables conservadas.
        zeta = np.asarray(initial_state.zeta, dtype=float).ravel().copy()
        u = np.asarray(initial_state.u, dtype=float).ravel().copy()
        v = np.asarray(initial_state.v, dtype=float).ravel().copy()
        h = zeta + h0
        if np.any(h < -config.min_depth):
            raise ValueError("Estado inicial con profundidad negativa.")
        self._h = np.maximum(h, 0.0)
        self._hu = self._h * u
        self._hv = self._h * v
        self._t = float(initial_state.t)
        self._dt_last = 0.0
        self._n_steps = 0

        # Parametro de Coriolis en el plano f de la latitud de referencia.
        self._f = TITAN.coriolis(domain.lat0_deg) if forcing.coriolis_enabled else 0.0

        # Buffers reutilizados por _rhs (evitan realojos en el bucle temporal).
        self._h_ext = np.empty(self.nx + 2 * NG)
        self._hu_ext = np.empty(self.nx + 2 * NG)
        self._hv_ext = np.empty(self.nx + 2 * NG)

        # Invariantes de referencia para diagnostics().
        self._mass0 = float(np.sum(self._h) * self.dx)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _velocity(self, h: np.ndarray, hq: np.ndarray) -> np.ndarray:
        """Velocidad desingularizada: exactamente 0 bajo el umbral de secado."""
        wet = h > self._config.min_depth
        return np.where(wet, hq / np.where(wet, h, 1.0), 0.0)

    def _rhs(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, t: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Operador espacial L(q, t): devuelve (dh/dt, dhu/dt, dhv/dt) en las
        celdas interiores. Contiene TODA la fisica del paso.
        """
        g = self.g
        dx = self.dx
        m0, m1 = NG, NG + self.nx

        # --- 1. Extension con celdas fantasma ---------------------------
        he, hue, hve = self._h_ext, self._hu_ext, self._hv_ext
        he[NG:-NG] = h
        hue[NG:-NG] = hu
        hve[NG:-NG] = hv
        apply_bc(he, hue, hve, self._west, self._east)

        # --- 2-5. Divergencia de flujo + fuente well-balanced ------------
        # Toda la discretizacion espacial vive en core/fvm_kernel.py, que el
        # motor 2D reutiliza direccion a direccion. En 1D solo hay direccion x,
        # con (mn, mt) = (hu, hv).
        rhs_h, rhs_hu, rhs_hv = directional_rhs(
            he, hue, hve, self._b_ext, dx, g, self.order,
            self._config.min_depth, self.limiter)

        # --- 6. Forzamientos neutrales al motor --------------------------
        u_i = self._velocity(h, hu)
        v_i = self._velocity(h, hv)
        fz = self._forcing

        if fz.pressure is not None:
            # Gradiente barometrico: -(h/rho) dP/dx, diferencia centrada O(dx^2).
            p = np.asarray(fz.pressure.evaluate(self._x_ext, self._y_ext, t),
                           dtype=float)
            dpdx = (p[m0 + 1:m1 + 1] - p[m0 - 1:m1 - 1]) / (2.0 * dx)
            rhs_hu -= h * dpdx / self.fluid.rho

        if fz.wind is not None:
            u10, v10 = fz.wind.evaluate(self.x, self._y_ext[m0:m1], t)
            tau_x, tau_y = _wind_stress(np.asarray(u10, dtype=float),
                                        np.asarray(v10, dtype=float))
            rhs_hu += tau_x / self.fluid.rho
            rhs_hv += tau_y / self.fluid.rho

        if fz.friction is not None:
            st = State(zeta=h - self.h0, u=u_i, v=v_i, t=t)
            ax, ay = fz.friction.stress(st, h)
            rhs_hu += h * np.asarray(ax, dtype=float)
            rhs_hv += h * np.asarray(ay, dtype=float)

        if self._f != 0.0:
            rhs_hu += self._f * hv
            rhs_hv -= self._f * hu

        if self.extra_source is not None:
            s_h, s_hu, s_hv = self.extra_source(self.x, t)
            rhs_h = rhs_h + s_h
            rhs_hu = rhs_hu + s_hu
            rhs_hv = rhs_hv + s_hv

        return rhs_h, rhs_hu, rhs_hv

    # ------------------------------------------------------------------
    # Contrato: paso temporal
    # ------------------------------------------------------------------
    def compute_stable_dt(self) -> float:
        """dt = cfl * dx / max(|u| + sqrt(g h)) sobre las celdas mojadas."""
        h = self._h
        u = self._velocity(h, self._hu)
        v = self._velocity(h, self._hv)
        speed = np.sqrt(u * u + v * v) + np.sqrt(self.g * np.maximum(h, 0.0))
        smax = float(np.max(speed))
        if not np.isfinite(smax) or smax <= 0.0:
            # Fluido en reposo absoluto sobre fondo seco: paso acotado por max_dt.
            return self._config.max_dt if self._config.max_dt else 1.0
        return self._config.cfl * self.dx / smax

    def step(self, dt: float | None = None) -> float:
        """
        Un paso SSPRK3 (Shu-Osher, 3 etapas). Devuelve el dt usado.

        FORMA INCREMENTAL (critica para la compuerta 1). Las etapas se escriben
        como q + dt * (combinacion de k), no como combinaciones convexas
        0.75*q + 0.25*(...). Ambas son ALGEBRAICAMENTE IDENTICAS, pero la forma
        convexa introduce redondeo de punto flotante incluso cuando el operador
        espacial devuelve exactamente cero: 0.75*h + 0.25*h != h para h generico.
        Ese redondeo destruia el equilibrio de reposo exacto (residuo ~1e-13 tras
        400 pasos). Con la forma incremental, k = 0 deja el estado intacto bit
        a bit.

            q1 = q + dt*k1
            q2 = q + dt*(k1 + k2)/4
            q^{n+1} = q + dt*(k1 + k2 + 4*k3)/6
        """
        if dt is None:
            dt = self.compute_stable_dt()
            if self._config.max_dt is not None:
                dt = min(dt, self._config.max_dt)
        dt = float(dt)
        t = self._t
        h0_, hu0, hv0 = self._h, self._hu, self._hv

        k1 = self._rhs(h0_, hu0, hv0, t)
        h1 = np.maximum(h0_ + dt * k1[0], 0.0)
        hu1 = hu0 + dt * k1[1]
        hv1 = hv0 + dt * k1[2]
        self._dry_out(h1, hu1, hv1)

        k2 = self._rhs(h1, hu1, hv1, t + dt)
        c = 0.25 * dt
        h2 = np.maximum(h0_ + c * (k1[0] + k2[0]), 0.0)
        hu2 = hu0 + c * (k1[1] + k2[1])
        hv2 = hv0 + c * (k1[2] + k2[2])
        self._dry_out(h2, hu2, hv2)

        k3 = self._rhs(h2, hu2, hv2, t + 0.5 * dt)
        c = dt / 6.0
        hn = np.maximum(h0_ + c * (k1[0] + k2[0] + 4.0 * k3[0]), 0.0)
        hun = hu0 + c * (k1[1] + k2[1] + 4.0 * k3[1])
        hvn = hv0 + c * (k1[2] + k2[2] + 4.0 * k3[2])
        self._dry_out(hn, hun, hvn)

        self._h, self._hu, self._hv = hn, hun, hvn
        self._t = t + dt
        self._dt_last = dt
        self._n_steps += 1
        return dt

    def _dry_out(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        """Anula el momento en celdas por debajo del umbral de secado."""
        dry = h <= self._config.min_depth
        if np.any(dry):
            hu[dry] = 0.0
            hv[dry] = 0.0

    # ------------------------------------------------------------------
    # Contrato: lectura de estado y diagnostico
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Estado publico en variables PRIMITIVAS (copia defensiva)."""
        return State(zeta=self._h - self.h0,
                     u=self._velocity(self._h, self._hu),
                     v=self._velocity(self._h, self._hv),
                     t=self._t)

    @property
    def config(self) -> SolverConfig:
        return self._config

    def diagnostics(self) -> dict:
        """
        Invariantes para validation/conservation.py.

        energy = integral [ 0.5 h (u^2+v^2) + 0.5 g eta^2 ] dx
        (energia total de SWE salvo la constante 0.5 g b^2, irrelevante para el
        seguimiento de crecimiento espurio).
        """
        h = self._h
        u = self._velocity(h, self._hu)
        v = self._velocity(h, self._hv)
        eta = h - self.h0
        dx = self.dx
        mass = float(np.sum(h) * dx)
        return {
            "t": self._t,
            "n_steps": self._n_steps,
            "dt_last": self._dt_last,
            "mass": mass,
            "mass_ref": self._mass0,
            "mass_rel_error": abs(mass - self._mass0) / max(abs(self._mass0), 1e-300),
            "momentum_x": float(np.sum(self._hu) * dx),
            "momentum_y": float(np.sum(self._hv) * dx),
            "energy": float(np.sum(0.5 * h * (u * u + v * v)
                                   + 0.5 * self.g * eta * eta) * dx),
            "h_min": float(np.min(h)),
            "zeta_max_abs": float(np.max(np.abs(eta))),
            "is_finite": bool(np.all(np.isfinite(h)) and np.all(np.isfinite(self._hu))
                              and np.all(np.isfinite(self._hv))),
        }


# ---------------------------------------------------------------------------
# Esfuerzo de viento (formulacion de arrastre cuadratico)
# ---------------------------------------------------------------------------
def _wind_stress(u10: np.ndarray, v10: np.ndarray,
                 c_d: float = 1.5e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    tau = rho_atm * C_d * |U10| * U10  [Pa].

    rho_atm proviene de constants.titan_params (TO_VERIFY). C_d es el unico
    parametro libre: 1.5e-3 es el valor de arranque; NO es una constante fisica
    de Titan verificada y debe barrerse en el analisis de sensibilidad.
    """
    speed = np.sqrt(u10 * u10 + v10 * v10)
    k = TITAN.rho_atm * c_d * speed
    return k * u10, k * v10
