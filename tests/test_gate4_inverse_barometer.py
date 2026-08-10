"""
COMPUERTA 4 - Barometro inverso estatico.
=========================================
Ante un dP constante, zeta debe tender a -dP/(rho g), el valor que devuelve
`inverse_barometer()` de constants/titan_params.py.

MONTAJE DEL EXPERIMENTO (supuestos explicitos)
----------------------------------------------
El equilibrio de barometro inverso NO puede alcanzarse con una presion UNIFORME
en una cuenca cerrada: la masa se conserva y la superficie no puede bajar en
bloque. El montaje correcto, y el que se usa aqui, es una perturbacion de
presion LOCALIZADA en una cuenca cerrada. El equilibrio es entonces

    zeta_eq(x) = -(P(x) - <P>) / (rho g)

con <P> la media espacial de P, fijada por la conservacion de masa. La
diferencia centro-borde reproduce exactamente inverse_barometer(dP), que es la
magnitud que la compuerta verifica.

* Cuenca cerrada de 200 km, 400 celdas, fondo plano de 160 m (orden Ligeia).
* Perturbacion gaussiana estatica, dP = +100 Pa (ALTA presion), sigma = 20 km.
* Rampa de encendido de 5 tiempos de cruce: forzamiento cuasi-estatico.
* Friccion lineal r = 2e-4 1/s para amortiguar el seiche residual. La friccion
  actua solo sobre u; en el equilibrio u = 0, asi que NO sesga el valor de
  equilibrio, solo acelera su alcance.
* Sin Coriolis (el equilibrio de barometro inverso es geostroficamente neutro
  en 1D).
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import DEFAULT_FLUID, TITAN, inverse_barometer
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import flat_channel, quiescent_state
from forcing.atmospheric import StaticGaussianPressure, UniformPressure
from forcing.friction import LinearFriction
from utils.runners import run_to

pytestmark = pytest.mark.gate

L = 2.0e5            # longitud de la cuenca [m]
NX = 400
DEPTH = 160.0        # profundidad [m]
DP = 100.0           # amplitud de la perturbacion de presion [Pa]
SIGMA = 2.0e4        # semiancho de la perturbacion [m]
R_FRIC = 2.0e-4      # coeficiente de friccion lineal [1/s]
TOL_REL = 0.02       # 2% de tolerancia sobre la respuesta de barometro inverso

T_CRUCE = L / np.sqrt(TITAN.g * DEPTH)
T_RAMPA = 5.0 * T_CRUCE
T_FINAL = 12.0 * T_CRUCE


def _run(amplitude=DP, coriolis=False):
    dom = flat_channel(L, NX, DEPTH)
    presion = StaticGaussianPressure(amplitude=amplitude, sigma=SIGMA,
                                     x0=0.5 * L, t_ramp=T_RAMPA)
    forz = ForcingBundle(pressure=presion, friction=LinearFriction(r=R_FRIC),
                         coriolis_enabled=coriolis)
    cfg = SolverConfig(cfl=0.45, bc={"west": "reflective", "east": "reflective"})
    res = run_to(dom, quiescent_state(dom), forz, cfg, t_end=T_FINAL,
                 sample_every=100)
    return dom, presion, res


def _zeta_equilibrio(dom, presion, t):
    """Perfil de equilibrio teorico: -(P - <P>)/(rho g)."""
    p = presion.evaluate(dom.x, np.zeros_like(dom.x), t)
    return -(p - p.mean()) / (DEFAULT_FLUID.rho * TITAN.g)


def test_diferencia_centro_borde_es_la_respuesta_de_barometro_inverso():
    """zeta(centro) - zeta(borde) = inverse_barometer(dP), dentro del 2%."""
    dom, presion, res = _run()
    zeta = res.state.zeta
    i_centro = NX // 2
    zeta_borde = 0.5 * (zeta[:20].mean() + zeta[-20:].mean())
    delta_num = zeta[i_centro] - zeta_borde

    p = presion.evaluate(dom.x, np.zeros_like(dom.x), res.state.t)
    delta_p_efectivo = p[i_centro] - 0.5 * (p[:20].mean() + p[-20:].mean())
    delta_teorico = inverse_barometer(delta_p_efectivo)

    err = abs(delta_num - delta_teorico) / abs(delta_teorico)
    print(f"\n--- COMPUERTA 4: barometro inverso estatico ---")
    print(f"dP efectivo centro-borde = {delta_p_efectivo:.4f} Pa")
    print(f"zeta numerico  = {delta_num:.6f} m")
    print(f"zeta teorico   = {delta_teorico:.6f} m   "
          f"(inverse_barometer, rho={DEFAULT_FLUID.rho}, g={TITAN.g})")
    print(f"error relativo = {err:.4%}")
    assert err < TOL_REL, f"error relativo {err:.4%} > {TOL_REL:.0%}"


def test_perfil_completo_converge_al_equilibrio_teorico():
    """L_inf del error del perfil, normalizado por la amplitud del IB."""
    dom, presion, res = _run()
    eq = _zeta_equilibrio(dom, presion, res.state.t)
    escala = abs(inverse_barometer(DP))
    err = np.max(np.abs(res.state.zeta - eq)) / escala
    print(f"\nerror L_inf del perfil / |zeta_IB| = {err:.4%}")
    assert err < 0.05, f"error de perfil {err:.4%} > 5%"


def test_signo_alta_presion_deprime_la_superficie():
    """Convencion de signo: dP > 0 (alta presion) => zeta < 0 bajo el centro."""
    _, _, res = _run(amplitude=+DP)
    assert res.state.zeta[NX // 2] < 0.0
    _, _, res_baja = _run(amplitude=-DP)
    assert res_baja.state.zeta[NX // 2] > 0.0


def test_respuesta_lineal_en_la_amplitud_de_presion():
    """Regimen lineal: duplicar dP duplica zeta (dentro del 1%)."""
    _, _, r1 = _run(amplitude=DP)
    _, _, r2 = _run(amplitude=2.0 * DP)
    z1 = r1.state.zeta[NX // 2] - r1.state.zeta[:20].mean()
    z2 = r2.state.zeta[NX // 2] - r2.state.zeta[:20].mean()
    assert z2 / z1 == pytest.approx(2.0, rel=0.01)


def test_presion_uniforme_no_produce_respuesta():
    """Control: gradiente de presion nulo => ninguna deformacion."""
    dom = flat_channel(L, NX, DEPTH)
    forz = ForcingBundle(pressure=UniformPressure(p0=TITAN.p_surface + 500.0),
                         coriolis_enabled=False)
    cfg = SolverConfig(cfl=0.45, bc={"west": "reflective", "east": "reflective"})
    res = run_to(dom, quiescent_state(dom), forz, cfg, t_end=2.0 * T_CRUCE,
                 sample_every=1000)
    assert np.max(np.abs(res.state.zeta)) == 0.0


def test_masa_conservada_bajo_forzamiento_barometrico():
    """El forzamiento de presion no debe crear ni destruir masa."""
    _, _, res = _run()
    assert res.monitor.mass_drift < 1e-12, res.monitor.mass_drift
