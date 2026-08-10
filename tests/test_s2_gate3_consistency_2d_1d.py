"""
SPRINT 2 - COMPUERTA 3: Consistencia 2D <-> 1D.
===============================================
En un canal 2D invariante transversalmente, el motor 2D debe reproducir la
solucion 1D de Proudman ya validada en Sprint 1.

POR QUE ESTA COMPUERTA ES FUERTE
--------------------------------
Con batimetria y forzamiento independientes de y, y con v = 0, la contribucion
de la direccion y al lado derecho debe ser EXACTAMENTE cero: las pendientes en y
son nulas, la reconstruccion hidrostatica da estados identicos a ambos lados de
cada interfaz en y, y el termino factorizado d(eta) da cero exacto (ADR-004).
Por tanto, integrando con la MISMA secuencia de dt, los dos motores deben
coincidir a precision de maquina, no solo "dentro de tolerancia". Se comprueban
las dos cosas: coincidencia exacta con dt impuesto, y coincidencia dentro de
tolerancia con el dt natural de cada motor (que difiere porque el CFL 2D no
escindido es mas restrictivo).

Ademas se comprueba la ISOTROPIA: el mismo problema girado 90 grados (canal a lo
largo de y, frente avanzando hacia el norte) debe dar el mismo resultado. Eso
ejercita la maquinaria del eje y de forma independiente.

SUPUESTOS: canal de 300 km, dx = dy = 400 m, h = 160 m, dP = 100 Pa,
sigma = 10 km, F = 0.95, recorrido D = 100 km, fronteras transmisivas en la
direccion de propagacion y reflectivas en la transversal, sin friccion,
sin Coriolis (el problema 1D no lo tiene).
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import inverse_barometer
from core.native_fvm import NativeFVMSolver
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import flat_channel, quiescent_state
from domain.geometry2d import (quiescent_state_2d,
                               transverse_invariant_channel)
from forcing.atmospheric import (MovingGaussianPressure, MovingPressureFront2D)
from physics.proudman import amplification_from_field, resonance_speed

pytestmark = pytest.mark.gate

L = 3.0e5
NX = 750             # dx = 400 m
NY = 8               # dy = 400 m  (Ly = NY*dx)
DEPTH = 160.0
DP = 100.0
SIGMA = 1.0e4
X0 = 3.0e4
FROUDE = 0.95
D_RECORRIDO = 1.0e5

C = resonance_speed(DEPTH)
U_STORM = FROUDE * C
T_END = D_RECORRIDO / U_STORM
DT_FIJO = 6.0        # por debajo del CFL de ambos motores

BC_1D = {"west": "transmissive", "east": "transmissive"}
BC_2D_X = {"west": "transmissive", "east": "transmissive",
           "south": "reflective", "north": "reflective"}
BC_2D_Y = {"west": "reflective", "east": "reflective",
           "south": "transmissive", "north": "transmissive"}


def _presion_1d():
    return MovingGaussianPressure(amplitude=DP, sigma=SIGMA, speed=U_STORM,
                                  x0=X0)


def _correr_1d(dt=None):
    dom = flat_channel(L, NX, DEPTH)
    s = NativeFVMSolver()
    s.initialize(dom, quiescent_state(dom),
                 ForcingBundle(pressure=_presion_1d(), coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc=BC_1D))
    _avanzar(s, dt)
    return dom, s


def _correr_2d_en_x(dt=None):
    dom = transverse_invariant_channel(L, NY * (L / NX), NX, NY,
                                       np.full(NX, DEPTH))
    presion = MovingPressureFront2D(amplitude=DP, sigma=SIGMA, speed=U_STORM,
                                    heading_deg=0.0, x0=X0, y0=0.0)
    s = NativeFVMSolver2D()
    s.initialize(dom, quiescent_state_2d(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc=BC_2D_X))
    _avanzar(s, dt)
    return dom, s


def _correr_2d_en_y(dt=None):
    """Mismo problema girado 90 grados: canal a lo largo de y."""
    from core.solver_base import Domain
    dx = L / NX
    x = (np.arange(NY) + 0.5) * dx
    y = (np.arange(NX) + 0.5) * dx
    dom = Domain(x=x, y=y, bathymetry=np.full((NX, NY), DEPTH), lat0_deg=78.0)
    presion = MovingPressureFront2D(amplitude=DP, sigma=SIGMA, speed=U_STORM,
                                    heading_deg=90.0, x0=0.0, y0=X0)
    s = NativeFVMSolver2D()
    s.initialize(dom, quiescent_state_2d(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 SolverConfig(cfl=0.45, bc=BC_2D_Y))
    _avanzar(s, dt)
    return dom, s


def _avanzar(s, dt):
    while s.state.t < T_END:
        paso = s.compute_stable_dt() if dt is None else dt
        paso = min(paso, T_END - s.state.t)
        if paso <= 0.0:
            break
        s.step(paso)


# ---------------------------------------------------------------------------
# 3a. La contribucion de la direccion transversal es EXACTAMENTE cero
# ---------------------------------------------------------------------------
def test_la_direccion_transversal_no_contribuye_en_absoluto():
    _, s = _correr_2d_en_x(dt=DT_FIJO)
    st = s.state
    assert np.max(np.abs(st.v)) == 0.0, "v deberia ser cero exacto"
    # zeta y u no deben variar con y: cada columna es identica bit a bit.
    assert np.all(st.zeta == st.zeta[0, :][None, :])
    assert np.all(st.u == st.u[0, :][None, :])


# ---------------------------------------------------------------------------
# 3b. LA COMPUERTA: el motor 2D reproduce la solucion 1D
# ---------------------------------------------------------------------------
def test_con_el_mismo_dt_los_dos_motores_coinciden_a_precision_de_maquina():
    _, s1 = _correr_1d(dt=DT_FIJO)
    _, s2 = _correr_2d_en_x(dt=DT_FIJO)
    z1 = s1.state.zeta
    z2 = s2.state.zeta[0, :]
    escala = float(np.max(np.abs(z1)))
    err = float(np.max(np.abs(z2 - z1))) / escala
    print(f"\n--- COMPUERTA 3: 2D vs 1D con dt = {DT_FIJO} s ---")
    print(f"max|zeta_1D| = {escala:.6f} m")
    print(f"error relativo L_inf 2D-1D = {err:.3e}")
    assert s1.state.t == pytest.approx(s2.state.t)
    assert err < 1e-12, f"error relativo {err:.3e} no es de nivel de maquina"


def test_con_dt_natural_de_cada_motor_coinciden_dentro_de_tolerancia():
    """
    Con el dt natural de cada motor (el CFL 2D es ~2x mas restrictivo) las
    soluciones difieren solo por el error de truncamiento temporal de SSPRK3.
    """
    _, s1 = _correr_1d()
    _, s2 = _correr_2d_en_x()
    z1, z2 = s1.state.zeta, s2.state.zeta[0, :]
    escala = float(np.max(np.abs(z1)))
    err = float(np.max(np.abs(z2 - z1))) / escala
    print(f"\nerror relativo L_inf con dt natural = {err:.3e}")
    assert err < 1e-3, f"error relativo {err:.3e} > 1e-3"


def test_el_factor_de_amplificacion_coincide():
    """La magnitud cientifica del Paper 1 debe salir igual en 1D y en 2D."""
    _, s1 = _correr_1d()
    dom2, s2 = _correr_2d_en_x()
    margen1 = (np.arange(NX) + 0.5) * (L / NX)
    interior = (margen1 > 2.0e4) & (margen1 < L - 2.0e4)
    r1 = amplification_from_field(s1.state.zeta[interior], DP)
    r2 = amplification_from_field(s2.state.zeta[:, interior], DP)
    print(f"\nR_1D = {r1:.4f}   R_2D = {r2:.4f}   "
          f"|zeta_IB| = {abs(inverse_barometer(DP)):.5f} m")
    assert r2 == pytest.approx(r1, rel=1e-3)


# ---------------------------------------------------------------------------
# 3c. Isotropia: el mismo problema girado 90 grados
# ---------------------------------------------------------------------------
def test_el_problema_girado_90_grados_da_el_mismo_resultado():
    """
    Ejercita la maquinaria del eje y de forma independiente. Debe coincidir a
    precision de maquina con el caso en x: el esquema no escindido no privilegia
    ningun eje.
    """
    _, sx = _correr_2d_en_x(dt=DT_FIJO)
    _, sy = _correr_2d_en_y(dt=DT_FIJO)
    zx = sx.state.zeta[0, :]          # perfil a lo largo de x
    zy = sy.state.zeta[:, 0]          # perfil a lo largo de y
    escala = float(np.max(np.abs(zx)))
    err = float(np.max(np.abs(zy - zx))) / escala
    print(f"\nerror relativo L_inf entre canal-en-x y canal-en-y = {err:.3e}")
    assert np.max(np.abs(sy.state.u)) == 0.0, "u deberia ser cero en el canal-y"
    assert err < 1e-12, f"anisotropia detectada: error {err:.3e}"


def test_los_dos_motores_pierden_exactamente_la_misma_masa():
    """
    OJO: con fronteras TRANSMISIVAS la masa NO se conserva; sale del dominio con
    las ondas radiadas. Eso es correcto y esperado (la conservacion en dominio
    cerrado es la compuerta 2). Lo que esta compuerta exige es CONSISTENCIA: los
    dos motores deben perder la misma masa a precision de maquina.
    """
    _, s1 = _correr_1d(dt=DT_FIJO)
    _, s2 = _correr_2d_en_x(dt=DT_FIJO)
    d1, d2 = s1.diagnostics(), s2.diagnostics()
    # Hay flujo saliente, luego la deriva NO es despreciable.
    assert d1["mass_rel_error"] > 1e-9, (
        "sin flujo saliente el test no comprueba nada: revisa el montaje")
    err = abs(d1["mass_rel_error"] - d2["mass_rel_error"])
    print(f"\nderiva de masa 1D = {d1['mass_rel_error']:.6e}  "
          f"2D = {d2['mass_rel_error']:.6e}  |dif| = {err:.3e}")
    assert err < 1e-12 * max(d1["mass_rel_error"], 1.0)
