"""
COMPUERTA 5 - Resonancia de Proudman 1D.  (COMPUERTA CIENTIFICA CENTRAL, Paper 1)
=================================================================================
En un canal rectangular sin friccion se barre la velocidad de la perturbacion de
presion y se verifica que el factor de amplificacion de la superficie libre,
medido respecto a la respuesta estatica de barometro inverso, CRECE al acercarse
F -> 1.

    F = U / sqrt(g h)                          (physics.proudman.atmospheric_froude)
    R_num = max|zeta| / |zeta_IB|              (physics.proudman.amplification_from_field)
    R_teo(F) = 1 / |1 - F^2|                   (respuesta estacionaria lineal)

MONTAJE DEL EXPERIMENTO (supuestos explicitos)
----------------------------------------------
* Canal rectangular de 600 km, fondo plano h = 160 m (orden Ligeia Mare).
  c = sqrt(g h) = 14.71 m/s con g = TITAN.g.
* 1500 celdas => dx = 400 m; 25 celdas por semiancho gaussiano.
* Perturbacion gaussiana movil, dP = 100 Pa, sigma = 10 km, sin rampa.
  |zeta_IB| = 0.164 m << h: regimen plenamente lineal, la amplificacion medida
  es la de Proudman y no un efecto no lineal de amplitud finita.
* SIN friccion y SIN Coriolis: es el canal ideal de la teoria de Proudman.
* Fronteras transmisivas; el maximo se busca excluyendo 20 km junto a cada
  borde para descartar la reflexion residual de la frontera abierta.
* RECORRIDO FIJO D = 400 km para todos los F (t_final = D/U). Fijar la DISTANCIA
  y no el tiempo es lo que hace justa la comparacion: en resonancia exacta la
  amplitud crece con la distancia recorrida, no con el tiempo por si mismo.
  El factor medido en F = 1 esta acotado por este fetch finito (no diverge);
  esa saturacion es fisica, no un artefacto.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import DEFAULT_FLUID, TITAN, inverse_barometer
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import flat_channel, quiescent_state
from forcing.atmospheric import MovingGaussianPressure
from physics.proudman import (amplification_from_field, atmospheric_froude,
                              proudman_amplification, resonance_speed)
from utils.runners import run_to

pytestmark = pytest.mark.gate

L = 6.0e5            # longitud del canal [m]
NX = 1500            # celdas  => dx = 400 m
DEPTH = 160.0        # profundidad constante [m]
DP = 100.0           # amplitud de presion [Pa]
SIGMA = 1.0e4        # semiancho gaussiano [m]
X0 = 5.0e4           # posicion inicial del centro [m]
D_RECORRIDO = 4.0e5  # distancia recorrida por la perturbacion [m]
MARGEN = 2.0e4       # margen excluido junto a cada frontera [m]

FROUDE_SUBCRITICOS = (0.5, 0.8, 0.9, 0.95)
FROUDE_BARRIDO = FROUDE_SUBCRITICOS + (1.0, 1.05, 1.2)


def _corrida(froude: float) -> float:
    """Devuelve el factor de amplificacion numerico R_num para un F dado."""
    c = resonance_speed(DEPTH)
    u_storm = froude * c
    dom = flat_channel(L, NX, DEPTH)
    presion = MovingGaussianPressure(amplitude=DP, sigma=SIGMA,
                                     speed=u_storm, x0=X0)
    cfg = SolverConfig(cfl=0.45,
                       bc={"west": "transmissive", "east": "transmissive"})
    res = run_to(dom, quiescent_state(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 cfg, t_end=D_RECORRIDO / u_storm, order=2,
                 sample_every=100_000)
    interior = (dom.x > MARGEN) & (dom.x < L - MARGEN)
    assert res.monitor.all_finite
    return amplification_from_field(res.state.zeta[interior], DP)


@pytest.fixture(scope="module")
def barrido() -> dict[float, float]:
    """Barrido de F ejecutado una sola vez para todo el modulo."""
    r = {fr: _corrida(fr) for fr in FROUDE_BARRIDO}
    print("\n--- COMPUERTA 5: resonancia de Proudman 1D ---")
    print(f"h = {DEPTH} m,  c = sqrt(g h) = {resonance_speed(DEPTH):.4f} m/s "
          f"(g = {TITAN.g} m/s^2)")
    print(f"dP = {DP} Pa,  |zeta_IB| = {abs(inverse_barometer(DP)):.5f} m "
          f"(rho = {DEFAULT_FLUID.rho} kg/m^3)")
    print(f"recorrido D = {D_RECORRIDO/1e3:.0f} km,  sigma = {SIGMA/1e3:.0f} km")
    print(f"{'F':>6} {'U [m/s]':>9} {'R_num':>9} {'R_teo=1/|1-F^2|':>17}")
    for fr in FROUDE_BARRIDO:
        teo = proudman_amplification(fr)
        teo_s = "inf" if not np.isfinite(teo) else f"{teo:.3f}"
        print(f"{fr:6.2f} {fr*resonance_speed(DEPTH):9.3f} {r[fr]:9.3f} "
              f"{teo_s:>17}")
    return r


# ---------------------------------------------------------------------------
# 5a. Coherencia del modulo physics/proudman.py
# ---------------------------------------------------------------------------
def test_froude_y_velocidad_de_resonancia_son_consistentes():
    c = resonance_speed(DEPTH)
    assert c == pytest.approx(np.sqrt(TITAN.g * DEPTH))
    assert atmospheric_froude(c, DEPTH) == pytest.approx(1.0)
    assert atmospheric_froude(0.5 * c, DEPTH) == pytest.approx(0.5)


def test_amplificacion_teorica_diverge_en_f_igual_a_uno():
    assert proudman_amplification(0.0) == pytest.approx(1.0)
    assert np.isinf(proudman_amplification(1.0))
    assert proudman_amplification(0.9) > proudman_amplification(0.5)
    # Simetria de la respuesta estacionaria a ambos lados de F = 1.
    assert proudman_amplification(0.9) == pytest.approx(
        proudman_amplification(np.sqrt(2.0 - 0.81)))


# ---------------------------------------------------------------------------
# 5b. LA COMPUERTA: la amplificacion crece al acercarse F -> 1
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_la_amplificacion_crece_monotonamente_hacia_la_resonancia(barrido):
    subida = [barrido[fr] for fr in FROUDE_SUBCRITICOS + (1.0,)]
    diffs = np.diff(subida)
    assert np.all(diffs > 0.0), (
        f"la amplificacion no crece monotonamente hacia F=1: {subida}")


@pytest.mark.slow
def test_la_amplificacion_en_resonancia_supera_de_largo_a_la_del_caso_lento(barrido):
    """La resonancia debe ser un efecto grande, no un matiz de pocos por ciento."""
    razon = barrido[1.0] / barrido[0.5]
    assert razon > 5.0, f"R(F=1)/R(F=0.5) = {razon:.2f}, se esperaba > 5"


@pytest.mark.slow
def test_la_respuesta_supera_ampliamente_al_barometro_inverso_estatico(barrido):
    """Toda la banda resonante debe superar la respuesta estatica (R >> 1)."""
    assert barrido[0.95] > 5.0
    assert barrido[1.0] > 8.0


@pytest.mark.slow
def test_regimen_subcritico_reproduce_la_teoria_estacionaria(barrido):
    """
    Lejos de la resonancia el estado cuasi-estacionario se alcanza dentro del
    recorrido simulado y R_num debe coincidir con 1/|1-F^2| dentro del 5%.
    Es la verificacion cuantitativa de que la amplificacion medida es la de
    Proudman y no un artefacto numerico.
    """
    for fr in (0.5, 0.8):
        teo = proudman_amplification(fr)
        err = abs(barrido[fr] - teo) / teo
        assert err < 0.05, (f"F={fr}: R_num={barrido[fr]:.3f} vs "
                            f"R_teo={teo:.3f} (error {err:.2%})")


@pytest.mark.slow
def test_la_amplificacion_decae_en_regimen_supercritico(barrido):
    """Pasada la resonancia, F > 1, la amplificacion debe volver a caer."""
    assert barrido[1.05] < barrido[1.0]
    assert barrido[1.2] < barrido[1.05]
    # El caso supercritico lejano vuelve a acercarse a la teoria estacionaria.
    teo = proudman_amplification(1.2)
    assert abs(barrido[1.2] - teo) / teo < 0.10


@pytest.mark.slow
def test_la_curva_de_resonancia_tiene_su_maximo_en_f_igual_a_uno(barrido):
    fr_max = max(FROUDE_BARRIDO, key=lambda f: barrido[f])
    assert fr_max == 1.0, f"el maximo de amplificacion cae en F={fr_max}"
