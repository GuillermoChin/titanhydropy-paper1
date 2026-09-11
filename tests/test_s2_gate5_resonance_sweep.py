"""
COMPUERTA 5 (2D): Barrido de resonancia y CONVERGENCIA DEL PICO.
================================================================
COMPUERTA CIENTIFICA CENTRAL DEL PAPER 1.

Dos exigencias, no una:
  (a) la curva A(U) presenta un maximo CLARO en F -> 1, interior al barrido;
  (b) ese pico CONVERGE bajo refinamiento de malla.

(b) es la exigencia dura. Un esquema disipativo puede producir un pico bonito
cuya altura la fije la disipacion numerica y no la fisica. Si al refinar la
malla el pico siguiera creciendo sin freno, o se estabilizara en un valor que
depende de dx, la curva no seria un resultado fisico. Lo que se exige es que los
cambios sucesivos bajo refinamiento DECREZCAN de forma consistente.

MONTAJE (supuestos explicitos)
------------------------------
* Canal rectangular de 600 km, fondo plano h = 160 m (orden Ligeia Mare),
  c = sqrt(g h) = 14.7078 m/s con g = TITAN.g.
* Perturbacion gaussiana movil, dP = 100 Pa, sigma = 10 km, sin rampa.
  |zeta_IB| = 0.164 m << h: regimen plenamente lineal.
* SIN friccion y SIN Coriolis: canal ideal de la teoria de Proudman.
* Fronteras transmisivas; el maximo excluye 20 km junto a cada borde.
* RECORRIDO FIJO D = 400 km para todas las velocidades (t_final = D/U). Fijar la
  DISTANCIA y no el tiempo es lo que hace justa la comparacion: en resonancia
  exacta la amplitud crece con la distancia recorrida.
* Eje del barrido = SWEEP_SPEED_RANGE (decision de diseno, no dato).

* MOTOR: se usa el motor 1D. La compuerta 3 demostro que el motor 2D reproduce
  el 1D a precision de maquina en canal invariante transversalmente, de modo que
  el barrido en 1D es el mismo objeto cientifico a una fraccion del coste. Se
  incluye ademas un punto de control corrido con el motor 2D.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import (SWEEP_SPEED_RANGE, TITAN,
                                    inverse_barometer, resonant_depth)
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import flat_channel, quiescent_state
from domain.geometry2d import (quiescent_state_2d,
                               transverse_invariant_channel)
from forcing.atmospheric import (MovingGaussianPressure, MovingPressureFront2D)
from physics.proudman import (amplification_from_field, atmospheric_froude,
                              bound_amplification, proudman_amplification,
                              resonance_speed)
from utils.runners import run_to

pytestmark = pytest.mark.gate

L = 6.0e5
DEPTH = 160.0
DP = 100.0
SIGMA = 1.0e4
X0 = 5.0e4
D_RECORRIDO = 4.0e5
MARGEN = 2.0e4
NX_PRODUCCION = 1500          # dx = 400 m, 25 celdas por sigma

C = resonance_speed(DEPTH)
VELOCIDADES = (2.0, 5.0, 8.0, 11.0, 13.0, C, 16.0, 18.0, 20.0)
RESOLUCIONES_PICO = (750, 1500, 3000, 6000)


def _corrida(u_storm: float, nx: int = NX_PRODUCCION):
    """Devuelve (A_global, A_ligada) para una velocidad de tormenta."""
    dom = flat_channel(L, nx, DEPTH)
    presion = MovingGaussianPressure(amplitude=DP, sigma=SIGMA,
                                     speed=u_storm, x0=X0)
    cfg = SolverConfig(cfl=0.45,
                       bc={"west": "transmissive", "east": "transmissive"})
    res = run_to(dom, quiescent_state(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 cfg, t_end=D_RECORRIDO / u_storm, order=2,
                 sample_every=10 ** 9)
    assert res.monitor.all_finite
    interior = (dom.x > MARGEN) & (dom.x < L - MARGEN)
    zeta, x = res.state.zeta[interior], dom.x[interior]
    a_global = amplification_from_field(zeta, DP)
    a_ligada = bound_amplification(zeta, x, presion.center(res.state.t), DP,
                                   half_width_sigmas=3.0, sigma=SIGMA)
    return a_global, a_ligada


def _amplificacion(u_storm: float, nx: int = NX_PRODUCCION) -> float:
    return _corrida(u_storm, nx)[0]


@pytest.fixture(scope="module")
def barrido() -> dict[float, tuple[float, float]]:
    """(A_global, A_ligada) por velocidad, a resolucion de produccion."""
    a = {u: _corrida(u) for u in VELOCIDADES}
    ib = abs(inverse_barometer(DP))
    print("\n--- COMPUERTA 5: barrido de resonancia A(U) ---")
    print(f"h = {DEPTH} m,  c = {C:.4f} m/s,  dP = {DP} Pa,  "
          f"|zeta_IB| = {ib:.5f} m")
    print(f"eje del barrido SWEEP_SPEED_RANGE = {SWEEP_SPEED_RANGE} m/s,  "
          f"nx = {NX_PRODUCCION} (dx = {L/NX_PRODUCCION:.0f} m)")
    print(f"{'U [m/s]':>9} {'F':>7} {'h_res [m]':>10} {'A_global':>9} "
          f"{'A_ligada':>9} {'A_teo':>9} {'zeta [m]':>10}")
    for u in VELOCIDADES:
        fr = atmospheric_froude(u, DEPTH)
        teo = proudman_amplification(fr)
        teo_s = "inf" if not np.isfinite(teo) else f"{teo:9.3f}"
        print(f"{u:9.3f} {fr:7.3f} {resonant_depth(u):10.2f} {a[u][0]:9.3f} "
              f"{a[u][1]:9.3f} {teo_s:>9} {a[u][0]*ib:10.4f}")
    return a


@pytest.fixture(scope="module")
def curva(barrido) -> dict[float, float]:
    """Curva A(U) global: la magnitud reportada por el Paper 1."""
    return {u: v[0] for u, v in barrido.items()}


@pytest.fixture(scope="module")
def convergencia_pico() -> list[float]:
    """Amplificacion en F = 1 para resoluciones crecientes."""
    vals = [_amplificacion(C, nx) for nx in RESOLUCIONES_PICO]
    print("\n--- COMPUERTA 5: convergencia de malla del PICO (F = 1) ---")
    print(f"{'nx':>6} {'dx [m]':>9} {'sigma/dx':>9} {'A':>9} {'cambio':>10}")
    for i, nx in enumerate(RESOLUCIONES_PICO):
        dx = L / nx
        ch = "" if i == 0 else \
            f"{abs(vals[i]-vals[i-1])/vals[i-1]:9.4%}"
        print(f"{nx:6d} {dx:9.1f} {SIGMA/dx:9.1f} {vals[i]:9.4f} {ch:>10}")
    return vals


# ---------------------------------------------------------------------------
# 5a. La curva tiene un maximo claro en F -> 1
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_el_maximo_de_la_curva_cae_en_la_velocidad_resonante(curva):
    u_max = max(VELOCIDADES, key=lambda u: curva[u])
    assert u_max == pytest.approx(C, rel=1e-9), (
        f"el maximo cae en U = {u_max:.3f} m/s, no en c = {C:.3f} m/s")


@pytest.mark.slow
def test_el_maximo_es_interior_al_barrido(curva):
    """
    Si el maximo cayera en un extremo del eje, el barrido no lo habria
    capturado y la curva no demostraria nada.
    """
    assert SWEEP_SPEED_RANGE[0] < C < SWEEP_SPEED_RANGE[1]
    assert curva[VELOCIDADES[0]] < curva[C]
    assert curva[VELOCIDADES[-1]] < curva[C]


@pytest.mark.slow
def test_la_curva_crece_monotonamente_hasta_la_resonancia_y_decae_despues(curva):
    subida = [curva[u] for u in VELOCIDADES if u <= C]
    bajada = [curva[u] for u in VELOCIDADES if u >= C]
    assert np.all(np.diff(subida) > 0.0), f"la subida no es monotona: {subida}"
    assert np.all(np.diff(bajada) < 0.0), f"la bajada no es monotona: {bajada}"


@pytest.mark.slow
def test_el_pico_supera_de_largo_la_respuesta_estatica(curva):
    """La resonancia debe ser un efecto grande, no un matiz."""
    assert curva[C] > 8.0
    assert curva[C] / curva[VELOCIDADES[0]] > 5.0


@pytest.mark.slow
def test_lejos_de_la_resonancia_la_respuesta_ligada_recupera_la_teoria(barrido):
    """
    Verificacion cuantitativa de que lo medido es Proudman y no un artefacto.

    La teoria estacionaria 1/|1-F^2| describe la RESPUESTA LIGADA, la que viaja
    pegada a la perturbacion. Es contra ella contra la que hay que contrastar, a
    ambos lados de F = 1. (El maximo global incluye ademas la onda libre de
    arranque; ver el test siguiente.)
    """
    print(f"\n{'U':>6} {'F':>7} {'A_ligada':>9} {'A_teo':>9} {'error':>8}")
    for u in (5.0, 8.0, 11.0, 16.0, 18.0, 20.0):
        fr = atmospheric_froude(u, DEPTH)
        teo = proudman_amplification(fr)
        err = abs(barrido[u][1] - teo) / teo
        print(f"{u:6.1f} {fr:7.3f} {barrido[u][1]:9.4f} {teo:9.4f} {err:8.3%}")
        assert err < 0.06, (f"U={u}: A_ligada={barrido[u][1]:.3f} vs "
                            f"A_teo={teo:.3f} (error {err:.2%})")


@pytest.mark.slow
def test_el_maximo_global_sesga_al_alza_solo_en_regimen_supercritico(barrido):
    """
    DIAGNOSTICO EXPLICITO. El maximo global y la respuesta ligada
    coinciden en subcritico (la onda libre de arranque adelanta al disturbio y
    sale por la frontera abierta) y divergen en supercritico (la onda libre
    queda rezagada dentro del dominio). Documentarlo evita atribuir a la fisica
    de Proudman un sesgo que es del observable elegido.
    """
    for u in (5.0, 8.0, 11.0):
        assert barrido[u][0] == pytest.approx(barrido[u][1], rel=1e-9), (
            f"U={u} (subcritico): las dos metricas deberian coincidir")
    sesgo_20 = (barrido[20.0][0] - barrido[20.0][1]) / barrido[20.0][1]
    print(f"\nsesgo del maximo global en U=20 m/s (F=1.36): {sesgo_20:+.2%}")
    assert sesgo_20 > 0.05, "el sesgo supercritico deberia ser apreciable"


# ---------------------------------------------------------------------------
# 5b. LA EXIGENCIA DURA: el pico converge bajo refinamiento
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_el_pico_converge_bajo_refinamiento_de_malla(convergencia_pico):
    """
    Los cambios sucesivos deben DECRECER de forma consistente. Se comprueba:
      * cada cambio es menor que el anterior,
      * la razon entre cambios sucesivos es > 1.5 (convergencia efectiva),
      * el cambio en el nivel mas fino es < 5%.
    """
    a = np.array(convergencia_pico)
    cambios = np.abs(np.diff(a)) / a[:-1]
    razones = cambios[:-1] / cambios[1:]
    print(f"\ncambios sucesivos = {[f'{c:.4%}' for c in cambios]}")
    print(f"razones entre cambios = {[f'{r:.2f}' for r in razones]}")
    assert np.all(np.diff(cambios) < 0.0), "los cambios no decrecen"
    assert np.all(razones > 1.5), f"convergencia demasiado lenta: {razones}"
    assert cambios[-1] < 0.05, f"cambio en el nivel mas fino = {cambios[-1]:.3%}"


@pytest.mark.slow
def test_el_pico_crece_al_refinar_luego_la_disipacion_no_lo_fija(convergencia_pico):
    """
    Diagnostico explicito: la disipacion numerica REDUCE el pico, no lo fija.
    Al refinar, A crece hacia un limite finito. Si A decreciera al refinar, el
    pico grueso seria un artefacto de dispersion y la conclusion se invertiria.
    """
    a = np.array(convergencia_pico)
    assert np.all(np.diff(a) > 0.0), f"el pico no crece monotonamente: {a}"


@pytest.mark.slow
def test_extrapolacion_de_richardson_del_pico(convergencia_pico):
    """
    Estimacion del valor libre de malla y del sesgo que arrastra la resolucion
    de produccion. Es un numero que el Paper 1 debe reportar, no ocultar.
    """
    a = np.array(convergencia_pico)
    cambios = np.abs(np.diff(a))
    r = float(cambios[-2] / cambios[-1])          # razon observada
    a_inf = a[-1] + cambios[-1] / (r - 1.0)       # extrapolacion de Richardson
    sesgo = (a_inf - a[1]) / a_inf                # a[1] = nx de produccion
    print(f"\nA (nx={RESOLUCIONES_PICO[-1]}) = {a[-1]:.4f}")
    print(f"A extrapolado (Richardson, r={r:.2f}) = {a_inf:.4f}")
    print(f"sesgo de la resolucion de produccion (nx={RESOLUCIONES_PICO[1]}) "
          f"= {sesgo:.2%} por defecto")
    assert np.isfinite(a_inf) and a_inf > a[-1]
    assert sesgo < 0.20, f"la resolucion de produccion subestima un {sesgo:.1%}"


@pytest.mark.slow
def test_lejos_del_pico_el_resultado_es_practicamente_independiente_de_la_malla():
    """Control: fuera de resonancia la malla apenas importa (< 0.5%)."""
    vals = [_amplificacion(8.0, nx) for nx in (750, 1500, 3000)]
    cambios = np.abs(np.diff(vals)) / np.array(vals[:-1])
    print(f"\nA(U=8) por resolucion = {[f'{v:.4f}' for v in vals]}, "
          f"cambios = {[f'{c:.4%}' for c in cambios]}")
    assert np.all(cambios < 0.005)


# ---------------------------------------------------------------------------
# 5c. Punto de control con el motor 2D
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_el_motor_2d_reproduce_el_pico(curva):
    """El objeto cientifico no depende del motor usado para calcularlo."""
    ny = 6
    dx = L / NX_PRODUCCION
    dom = transverse_invariant_channel(L, ny * dx, NX_PRODUCCION, ny,
                                       np.full(NX_PRODUCCION, DEPTH))
    presion = MovingPressureFront2D(amplitude=DP, sigma=SIGMA, speed=C,
                                    heading_deg=0.0, x0=X0, y0=0.0)
    s = NativeFVMSolver2D()
    s.initialize(dom, quiescent_state_2d(dom),
                 ForcingBundle(pressure=presion, coriolis_enabled=False),
                 SolverConfig(cfl=0.45,
                              bc={"west": "transmissive", "east": "transmissive",
                                  "south": "reflective", "north": "reflective"}))
    t_end = D_RECORRIDO / C
    while s.state.t < t_end:
        dt = min(s.compute_stable_dt(), t_end - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)
    interior = (dom.x > MARGEN) & (dom.x < L - MARGEN)
    a2d = amplification_from_field(s.state.zeta[:, interior], DP)
    print(f"\nA(F=1) 1D = {curva[C]:.4f}   2D = {a2d:.4f}")
    assert a2d == pytest.approx(curva[C], rel=2e-3)
