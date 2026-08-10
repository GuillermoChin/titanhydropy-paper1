"""
core/native_fvm2d.py
====================
NativeFVMSolver2D: motor propio de volumenes finitos 2D para las Ecuaciones No
Lineales de Agua Poco Profunda, implementando el MISMO contrato `Solver`.

ECUACIONES RESUELTAS (2D NLSWE, promediadas en la vertical)
-----------------------------------------------------------
    d_t h  + d_x(h u)            + d_y(h v)            = 0
    d_t hu + d_x(h u^2 + g h^2/2) + d_y(h u v)         = -g h d_x b
                                                         - (h/rho) d_x P
                                                         + tau_x/rho + f h v
                                                         + h a_fric_x
    d_t hv + d_x(h u v)          + d_y(h v^2 + g h^2/2) = -g h d_y b
                                                         - (h/rho) d_y P
                                                         + tau_y/rho - f h u
                                                         + h a_fric_y

con h = zeta - b, b = -h0 (h0 = batimetria positiva del Domain, ADR-001).

=========================  SUPUESTOS NUMERICOS  =============================
* ESQUEMA DIMENSIONALMENTE **NO ESCINDIDO** (unsplit). Esto es explicito y
  deliberado: las divergencias de flujo en x y en y se evaluan sobre EL MISMO
  estado y se SUMAN dentro de una unica etapa de SSPRK3. No hay splitting de
  Strang ni de Godunov, y por tanto no hay error de escision ni dependencia del
  orden de barrido de los ejes. El precio es un CFL 2D mas restrictivo (ver
  compute_stable_dt) que el de un esquema escindido.
* Malla   : cartesiana uniforme, celdas centradas, dx y dy constantes
            (pueden diferir entre si).
* Espacio : volumenes finitos + MUSCL sobre (eta, u, v, b) con limitador
            configurable. La discretizacion direccional es EXACTAMENTE la misma
            rutina que usa el motor 1D (core.fvm_kernel.directional_rhs), de
            modo que la propiedad well-balanced demostrada bit a bit en 1D se
            hereda en cada direccion sin reimplementar nada (ADR-004, ADR-005).
* Fondo   : reconstruccion hidrostatica de Audusse et al. (2004) por direccion.
* Tiempo  : SSPRK3 en FORMA INCREMENTAL (ADR-003), imprescindible para el
            equilibrio de reposo exacto.
* CFL     : dt = cfl / max[ (|u|+c)/dx + (|v|+c)/dy ], la condicion propia de un
            esquema no escindido. Con dx = dy y v = 0 NO coincide con el CFL 1D:
            es mas restrictiva por el termino c/dy, lo cual es correcto.
* Secado  : h se limita a >= 0; u, v = 0 donde h <= config.min_depth.
* Coriolis: plano f con f = TITAN.coriolis(domain.lat0_deg), activable con
            ForcingBundle.coriolis_enabled. Se trata explicitamente (no implicito):
            valido mientras f*dt << 1, condicion holgada en Titan (f ~ 9e-6 1/s).

CONVENCION DE FORMA DE ARRAYS
-----------------------------
Todos los campos 2D son (ny, nx): la PRIMERA dimension es y y la SEGUNDA es x.
Es la convencion de matplotlib.imshow/pcolormesh y de np.meshgrid(indexing='xy'),
y evita transposiciones en la capa de visualizacion.
Domain.x tiene forma (nx,), Domain.y forma (ny,), Domain.bathymetry (ny, nx).

GANCHO DE ACELERACION: `_rhs` es puro NumPy sin ramas de Python. Candidato
directo a Numba/JAX. No se optimiza de forma prematura.
"""

from __future__ import annotations

import numpy as np

from constants.titan_params import TITAN, DEFAULT_FLUID, CryogenicFluid
from core.boundary import (NG, apply_bc_2d, extend_bathymetry_2d,
                           validate_bc_2d)
from core.fvm_kernel import desingularized_velocity, directional_rhs
from core.native_fvm import _wind_stress
from core.solver_base import (Domain, ForcingBundle, Solver, SolverConfig,
                              State)
from physics.wetting_drying import (WettingDryingConfig,
                                    desingularized_velocity_kp)

__all__ = ["NativeFVMSolver2D"]


class NativeFVMSolver2D(Solver):
    """
    Motor FVM 2D no escindido. El estado publico son variables primitivas
    (zeta, u, v) de forma (ny, nx); la conversion a conservadas (h, hu, hv) es
    estrictamente interna (ADR-001).

    Parametros de construccion
    --------------------------
    fluid        : propiedades del fluido criogenico (rho para presion y viento).
    g            : gravedad. Por defecto TITAN.g.
    order        : 1 o 2. Orden de la reconstruccion espacial.
    limiter      : 'minmod' | 'mc' | 'none'. 'none' NO es TVD; solo para el test
                   de orden formal.
    extra_source : callable (X, Y, t) -> (S_h, S_hu, S_hv). Gancho para el metodo
                   de soluciones manufacturadas. No forma parte de la fisica.
    """

    def __init__(self, fluid: CryogenicFluid = DEFAULT_FLUID,
                 g: float | None = None, order: int = 2,
                 limiter: str = "minmod", extra_source=None,
                 wetting_drying: bool | WettingDryingConfig = False) -> None:
        if order not in (1, 2):
            raise ValueError("order debe ser 1 o 2")
        self.fluid = fluid
        self.g = TITAN.g if g is None else float(g)
        self.order = order
        self.limiter = limiter
        self.extra_source = extra_source
        # Frontera movil. Desactivada por defecto: los dominios del Paper 1
        # estan siempre mojados y activarla solo anadiria coste y difusion.
        # Se concreta en initialize(), cuando se conoce config.min_depth.
        self._wd_request = wetting_drying
        self.wd: WettingDryingConfig | None = None
        self._mass_clipped = 0.0

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
                f"scheme={config.scheme!r} no implementado; solo 'hllc'.")
        if config.time_integrator != "ssprk3":
            raise NotImplementedError(
                f"time_integrator={config.time_integrator!r} no implementado; "
                "solo 'ssprk3'.")

        x = np.asarray(domain.x, dtype=float).ravel()
        y = np.asarray(domain.y, dtype=float).ravel()
        if x.size < 2 * NG + 1 or y.size < 2 * NG + 1:
            raise ValueError("El dominio 2D necesita al menos 5 celdas por eje.")
        for eje, coord in (("x", x), ("y", y)):
            d = np.diff(coord)
            if not np.allclose(d, d[0], rtol=1e-12, atol=0.0):
                raise ValueError(f"El motor nativo exige malla uniforme en {eje}.")

        self._domain = domain
        self._config = config
        self._forcing = forcing
        self.nx, self.ny = x.size, y.size
        self.dx, self.dy = float(x[1] - x[0]), float(y[1] - y[0])
        self.x, self.y = x, y
        self._west, self._east, self._south, self._north = validate_bc_2d(config.bc)

        # Batimetria (ny, nx) -> cota de fondo b = -h0.
        h0 = np.asarray(domain.bathymetry, dtype=float)
        if h0.shape != (self.ny, self.nx):
            raise ValueError(
                f"bathymetry debe tener forma (ny, nx) = ({self.ny}, {self.nx}); "
                f"recibido {h0.shape}. Ver convencion en el docstring del modulo.")
        self.h0 = h0
        self._b_ext = extend_bathymetry_2d(-h0, self._west, self._east,
                                           self._south, self._north)

        # Coordenadas extendidas (malla uniforme => extrapolacion lineal exacta).
        self._x_ext = x[0] + (np.arange(self.nx + 2 * NG) - NG) * self.dx
        self._y_ext = y[0] + (np.arange(self.ny + 2 * NG) - NG) * self.dy
        self._X_ext, self._Y_ext = np.meshgrid(self._x_ext, self._y_ext)
        self.X, self.Y = np.meshgrid(x, y)

        # Estado interno en variables conservadas.
        zeta = np.asarray(initial_state.zeta, dtype=float)
        u = np.asarray(initial_state.u, dtype=float)
        v = np.asarray(initial_state.v, dtype=float)
        for nombre, arr in (("zeta", zeta), ("u", u), ("v", v)):
            if arr.shape != (self.ny, self.nx):
                raise ValueError(f"initial_state.{nombre} debe ser (ny, nx).")
        h = zeta + h0
        if np.any(h < -config.min_depth):
            raise ValueError("Estado inicial con profundidad negativa.")
        self._h = np.maximum(h, 0.0)
        self._hu = self._h * u
        self._hv = self._h * v
        self._t = float(initial_state.t)
        self._dt_last = 0.0
        self._n_steps = 0

        self._f = TITAN.coriolis(domain.lat0_deg) if forcing.coriolis_enabled else 0.0

        shape_ext = (self.ny + 2 * NG, self.nx + 2 * NG)
        self._h_ext = np.empty(shape_ext)
        self._hu_ext = np.empty(shape_ext)
        self._hv_ext = np.empty(shape_ext)

        # Frontera movil: se concreta ahora, con el min_depth del config.
        if isinstance(self._wd_request, WettingDryingConfig):
            self.wd = self._wd_request
        elif self._wd_request:
            self.wd = WettingDryingConfig(min_depth=config.min_depth)
        else:
            self.wd = None
        self._mass_clipped = 0.0

        self._cell_area = self.dx * self.dy
        self._mass0 = float(np.sum(self._h) * self._cell_area)

    # ------------------------------------------------------------------
    # Operador espacial
    # ------------------------------------------------------------------
    def _velocity(self, h: np.ndarray, hq: np.ndarray) -> np.ndarray:
        if self.wd is not None and self.wd.desingularize:
            return desingularized_velocity_kp(h, hq, self._config.min_depth)
        return desingularized_velocity(h, hq, self._config.min_depth)

    def _rhs(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray, t: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Operador espacial L(q, t) NO ESCINDIDO: las dos direcciones se evaluan
        sobre el mismo estado y se suman.
        """
        g = self.g
        s = slice(NG, -NG)

        he, hue, hve = self._h_ext, self._hu_ext, self._hv_ext
        he[s, s] = h
        hue[s, s] = hu
        hve[s, s] = hv
        apply_bc_2d(he, hue, hve, self._west, self._east,
                    self._south, self._north)

        md, order, lim = self._config.min_depth, self.order, self.limiter

        # --- Direccion x: ultimo eje, (mn, mt) = (hu, hv) -------------------
        xh, xhu, xhv = directional_rhs(he, hue, hve, self._b_ext,
                                       self.dx, g, order, md, lim, self.wd)
        rhs_h = xh[s, :]
        rhs_hu = xhu[s, :]
        rhs_hv = xhv[s, :]

        # --- Direccion y: se transpone y se intercambian los momentos -------
        # Las vistas .T comparten memoria; directional_rhs no muta sus entradas.
        yh, yhv, yhu = directional_rhs(he.T, hve.T, hue.T, self._b_ext.T,
                                       self.dy, g, order, md, lim, self.wd)
        rhs_h = rhs_h + yh[s, :].T
        rhs_hu = rhs_hu + yhu[s, :].T
        rhs_hv = rhs_hv + yhv[s, :].T

        # --- Forzamientos neutrales al motor --------------------------------
        fz = self._forcing
        u_i = self._velocity(h, hu)
        v_i = self._velocity(h, hv)

        if fz.pressure is not None:
            # Gradiente barometrico 2D, diferencias centradas O(d^2).
            p = np.asarray(fz.pressure.evaluate(self._X_ext, self._Y_ext, t),
                           dtype=float)
            dpdx = (p[s, NG + 1:-NG + 1] - p[s, NG - 1:-NG - 1]) / (2.0 * self.dx)
            dpdy = (p[NG + 1:-NG + 1, s] - p[NG - 1:-NG - 1, s]) / (2.0 * self.dy)
            rhs_hu -= h * dpdx / self.fluid.rho
            rhs_hv -= h * dpdy / self.fluid.rho

        if fz.wind is not None:
            u10, v10 = fz.wind.evaluate(self.X, self.Y, t)
            tau_x, tau_y = _wind_stress(np.asarray(u10, dtype=float),
                                        np.asarray(v10, dtype=float))
            rhs_hu += tau_x / self.fluid.rho
            rhs_hv += tau_y / self.fluid.rho

        if fz.friction is not None:
            st = State(zeta=h - self.h0, u=u_i, v=v_i, t=t)
            ax, ay = fz.friction.stress(st, h)
            rhs_hu += h * np.asarray(ax, dtype=float)
            rhs_hv += h * np.asarray(ay, dtype=float)

        # --- Coriolis (plano f, explicito) ----------------------------------
        if self._f != 0.0:
            rhs_hu += self._f * hv
            rhs_hv -= self._f * hu

        if self.extra_source is not None:
            s_h, s_hu, s_hv = self.extra_source(self.X, self.Y, t)
            rhs_h = rhs_h + s_h
            rhs_hu = rhs_hu + s_hu
            rhs_hv = rhs_hv + s_hv

        return rhs_h, rhs_hu, rhs_hv

    # ------------------------------------------------------------------
    # Contrato: paso temporal
    # ------------------------------------------------------------------
    def compute_stable_dt(self) -> float:
        """
        CFL de esquema NO ESCINDIDO:

            dt = cfl / max[ (|u| + c)/dx + (|v| + c)/dy ],   c = sqrt(g h)

        Los dos ejes compiten por el mismo presupuesto de Courant, que es lo
        correcto cuando las dos divergencias se aplican en la misma etapa. Un
        esquema escindido admitiria el maximo de cada eje por separado.
        """
        h = self._h
        u = self._velocity(h, self._hu)
        v = self._velocity(h, self._hv)
        c = np.sqrt(self.g * np.maximum(h, 0.0))
        rate = (np.abs(u) + c) / self.dx + (np.abs(v) + c) / self.dy
        rmax = float(np.max(rate))
        if not np.isfinite(rmax) or rmax <= 0.0:
            return self._config.max_dt if self._config.max_dt else 1.0
        return self._config.cfl / rmax

    def step(self, dt: float | None = None) -> float:
        """SSPRK3 en forma incremental (ADR-003). Devuelve el dt usado."""
        if dt is None:
            dt = self.compute_stable_dt()
            if self._config.max_dt is not None:
                dt = min(dt, self._config.max_dt)
        dt = float(dt)
        t = self._t
        h0_, hu0, hv0 = self._h, self._hu, self._hv

        # El recorte de positividad se hace EXCLUSIVAMENTE en _dry_out, que es
        # tambien quien contabiliza la masa creada por el recorte. Si se
        # recortara aqui, `mass_clipped` leeria siempre cero y el diagnostico
        # seria una mentira silenciosa.
        k1 = self._rhs(h0_, hu0, hv0, t)
        h1 = h0_ + dt * k1[0]
        hu1 = hu0 + dt * k1[1]
        hv1 = hv0 + dt * k1[2]
        self._dry_out(h1, hu1, hv1)

        k2 = self._rhs(h1, hu1, hv1, t + dt)
        c = 0.25 * dt
        h2 = h0_ + c * (k1[0] + k2[0])
        hu2 = hu0 + c * (k1[1] + k2[1])
        hv2 = hv0 + c * (k1[2] + k2[2])
        self._dry_out(h2, hu2, hv2)

        k3 = self._rhs(h2, hu2, hv2, t + 0.5 * dt)
        c = dt / 6.0
        hn = h0_ + c * (k1[0] + k2[0] + 4.0 * k3[0])
        hun = hu0 + c * (k1[1] + k2[1] + 4.0 * k3[1])
        hvn = hv0 + c * (k1[2] + k2[2] + 4.0 * k3[2])
        self._dry_out(hn, hun, hvn)

        self._h, self._hu, self._hv = hn, hun, hvn
        self._t = t + dt
        self._dt_last = dt
        self._n_steps += 1
        return dt

    def _dry_out(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> None:
        """
        Anula el momento en celdas por debajo del umbral de secado y contabiliza
        la masa creada por el recorte de positividad, para que una perdida (o
        creacion) de masa se vea en diagnostics() en vez de esconderse aqui.
        """
        neg = np.minimum(h, 0.0)
        if np.any(neg < 0.0):
            self._mass_clipped += float(-np.sum(neg) * self._cell_area)
        np.maximum(h, 0.0, out=h)
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
        """Invariantes para validation/conservation.py (integrales de area)."""
        h = self._h
        u = self._velocity(h, self._hu)
        v = self._velocity(h, self._hv)
        eta = h - self.h0
        da = self._cell_area
        mass = float(np.sum(h) * da)
        return {
            "t": self._t,
            "n_steps": self._n_steps,
            "dt_last": self._dt_last,
            "mass": mass,
            "mass_ref": self._mass0,
            "mass_rel_error": abs(mass - self._mass0) / max(abs(self._mass0), 1e-300),
            # Con signo: distingue crear masa (recorte de positividad) de
            # perderla (anulacion de momento en laminas delgadas).
            "mass_rel_signed": (mass - self._mass0) / max(abs(self._mass0), 1e-300),
            "momentum_x": float(np.sum(self._hu) * da),
            "momentum_y": float(np.sum(self._hv) * da),
            "energy": float(np.sum(0.5 * h * (u * u + v * v)
                                   + 0.5 * self.g * eta * eta) * da),
            "h_min": float(np.min(h)),
            "zeta_max_abs": float(np.max(np.abs(eta))),
            "wet_fraction": float(np.mean(h > self._config.min_depth)),
            "wetting_drying": self.wd is not None,
            # Masa creada acumulada por el recorte h = max(h,0). Es el
            # diagnostico honesto de cuanto miente el tratamiento de secado.
            "mass_clipped": self._mass_clipped,
            "mass_clipped_rel": self._mass_clipped / max(abs(self._mass0), 1e-300),
            "is_finite": bool(np.all(np.isfinite(h))
                              and np.all(np.isfinite(self._hu))
                              and np.all(np.isfinite(self._hv))),
        }
