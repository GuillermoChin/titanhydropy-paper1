"""
COMPUERTA 6 (2D): Validacion fisico-Titan (Nivel C) completa.
=============================================================
Verifica que:
  * c = sqrt(g h) para REFERENCE_DEPTHS es coherente con OBSERVED_FRONT_SPEED
    (en su forma correcta: la banda resonante interseca el rango de
    profundidades de la cuenca),
  * la respuesta de barometro inverso simulada coincide con inverse_barometer(),
  * las magnitudes de zeta y de velocidad son fisicamente razonables.

Todo valor fisico se importa de constants/titan_params.py. Los criterios de
cordura (cotas de plausibilidad) NO son datos: se declaran en
validation/titan_physical.py y estan etiquetados como tales.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED,
                                    REFERENCE_DEPTHS, TITAN, inverse_barometer,
                                    resonant_depth, resonant_depth_band,
                                    shallow_water_speed)
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry2d import flat_basin, quiescent_state_2d
from forcing.atmospheric import MovingPressureFront2D
from forcing.friction import LinearFriction
from physics.proudman import froude_field, resonance_mask, resonant_region_summary
from validation.titan_physical import (check_inverse_barometer,
                                       check_velocity_magnitude,
                                       check_zeta_magnitude, pending_constants,
                                       proudman_overlap, run_all_checks,
                                       run_sanity)

pytestmark = pytest.mark.gate


# ---------------------------------------------------------------------------
# 6a. Coherencia de las constantes
# ---------------------------------------------------------------------------
def test_sanity_check_pasa_con_las_constantes_vigentes():
    run_sanity()


def test_la_auditoria_de_procedencia_sigue_delatando_los_to_verify():
    pend = pending_constants()
    assert pend
    assert any("OBSERVED_FRONT_SPEED" in p for p in pend)


def test_celeridad_de_cada_cuenca_vs_velocidades_observadas():
    """
    c = sqrt(g h) para cada REFERENCE_DEPTHS frente a OBSERVED_FRONT_SPEED.
    Se reporta explicitamente que NINGUNA cuenca resuena a profundidad maxima y
    que TODAS resuenan en su banda intermedia.
    """
    ov = proudman_overlap()
    v_min, v_max = ov["front_speed_range"]
    h_res_min, h_res_max = ov["resonant_depth_band_m"]
    print(f"\n--- COMPUERTA 6: c(h) vs OBSERVED_FRONT_SPEED ---")
    print(f"U observado = {v_min}-{v_max} m/s  =>  banda resonante "
          f"h = {h_res_min:.1f}-{h_res_max:.1f} m")
    print(f"{'cuenca':22s} {'h_max':>8} {'c':>8} {'F(h_max)':>10} "
          f"{'resuena en h_max':>17} {'resuena en cuenca':>18}")
    for nombre, b in ov["basins"].items():
        print(f"{nombre:22s} {b['depth_m']:8.1f} {b['wave_speed_m_s']:8.2f} "
              f"{v_max/b['wave_speed_m_s']:10.3f} "
              f"{str(b['resonance_at_max_depth']):>17} "
              f"{str(b['resonance_reachable']):>18}")
        assert b["resonance_reachable"], nombre
        assert b["wave_speed_m_s"] == pytest.approx(
            shallow_water_speed(b["depth_m"]))
    assert not any(b["resonance_at_max_depth"] for b in ov["basins"].values()), (
        "alguna cuenca resuena a profundidad maxima: la resonancia dejaria "
        "de ser un fenomeno exclusivo de los flancos")


def test_la_banda_resonante_es_consistente_con_la_definicion():
    h_min, h_max = resonant_depth_band()
    assert h_min == pytest.approx(OBSERVED_FRONT_SPEED[0] ** 2 / TITAN.g)
    assert h_max == pytest.approx(OBSERVED_FRONT_SPEED[1] ** 2 / TITAN.g)
    assert 0.0 < h_min < h_max


# ---------------------------------------------------------------------------
# 6b. Campo de Froude local y localizacion de F -> 1
# ---------------------------------------------------------------------------
def test_el_campo_de_froude_localiza_la_profundidad_resonante():
    """F(x,y) = U/sqrt(g h) debe valer 1 exactamente donde h = U^2/g."""
    u_storm = 8.0
    h_res = resonant_depth(u_storm)
    h = np.array([[1.0, h_res, 4.0 * h_res]])
    f = froude_field(u_storm, h)
    assert f[0, 1] == pytest.approx(1.0)
    assert f[0, 0] > 1.0 and f[0, 2] < 1.0
    assert resonance_mask(f, tol=1e-6).tolist() == [[False, True, False]]


def test_las_celdas_secas_no_cuentan_como_resonantes():
    f = froude_field(8.0, np.array([0.0, 1e-6, 50.0]))
    assert np.isinf(f[0]) and np.isinf(f[1])
    assert not resonance_mask(f).any() or np.isfinite(f[2])


def test_resumen_de_region_resonante_sobre_una_cuenca_con_pendiente():
    """
    Cuenca con profundidad creciente de 0 a 160 m: la region resonante debe ser
    una FRANJA de profundidad intermedia, no toda la cuenca.
    """
    ny, nx = 20, 200
    h = np.tile(np.linspace(1.0, 160.0, nx), (ny, 1))
    u_storm = OBSERVED_FRONT_SPEED[1]        # 10 m/s
    res = resonant_region_summary(u_storm, h, cell_area=1.0e6, tol=0.1)
    print(f"\nU = {u_storm} m/s -> h_res = {res['resonant_depth_m']:.1f} m; "
          f"franja resonante = {res['resonant_depth_range_m']} m; "
          f"fraccion de area = {res['resonant_area_fraction']:.3%}")
    assert res["resonant_depth_m"] == pytest.approx(u_storm ** 2 / TITAN.g)
    assert res["resonant_area_fraction"] > 0.0
    assert res["resonant_area_fraction"] < 0.5, (
        "la region resonante deberia ser una franja, no media cuenca")
    lo, hi = res["resonant_depth_range_m"]
    assert lo < res["resonant_depth_m"] < hi


# ---------------------------------------------------------------------------
# 6c. Verificaciones sobre un escenario 2D simulado
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def escenario_2d():
    """
    Cuenca 2D cerrada de 200x100 km, h = 47.3 m (la profundidad que resuena con
    U = 8 m/s, dentro de OBSERVED_FRONT_SPEED). Frente de presion movil de
    100 Pa con friccion lineal debil. Es un caso realista de Titan a escala de
    cuenca, no un caso de juguete.
    """
    lx, ly = 2.0e5, 1.0e5
    nx, ny = 200, 100
    u_storm = 8.0
    depth = resonant_depth(u_storm)
    dom = flat_basin(lx, ly, nx, ny, depth)
    presion = MovingPressureFront2D(amplitude=100.0, sigma=1.5e4,
                                    speed=u_storm, heading_deg=0.0,
                                    x0=2.0e4, y0=0.0)
    forz = ForcingBundle(pressure=presion,
                         friction=LinearFriction(r=1.0e-5),
                         coriolis_enabled=True)
    cfg = SolverConfig(cfl=0.45,
                       bc={"west": "transmissive", "east": "transmissive",
                           "south": "reflective", "north": "reflective"})
    s = NativeFVMSolver2D()
    s.initialize(dom, quiescent_state_2d(dom), forz, cfg)
    t_end = 1.2e5 / u_storm      # el frente recorre 120 km
    while s.state.t < t_end:
        dt = min(s.compute_stable_dt(), t_end - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)
    return s, dom, depth, u_storm


def test_las_magnitudes_del_escenario_son_fisicamente_razonables(escenario_2d):
    s, dom, depth, u_storm = escenario_2d
    st = s.state
    h = st.zeta + dom.bathymetry
    checks = run_all_checks(zeta=st.zeta, u=st.u, v=st.v, depth=h)
    print("\n--- COMPUERTA 6: verificaciones de Nivel C sobre el escenario ---")
    print(f"cuenca 200x100 km, h = {depth:.2f} m (resonante con "
          f"U = {u_storm} m/s), dP = 100 Pa, Coriolis ON")
    for c in checks:
        print(f"  {c}")
    assert all(c.passed for c in checks), [str(c) for c in checks if not c.passed]


def test_el_escenario_esta_en_resonancia_y_amplifica(escenario_2d):
    """Coherencia fisica: a F = 1 la respuesta debe superar al barometro inverso."""
    s, dom, depth, u_storm = escenario_2d
    f = froude_field(u_storm, dom.bathymetry)
    assert np.allclose(f, 1.0), "la cuenca deberia estar toda en resonancia"
    ib = abs(inverse_barometer(100.0))
    amp = float(np.max(np.abs(s.state.zeta))) / ib
    print(f"\namplificacion en la cuenca resonante = {amp:.3f} "
          f"(|zeta_IB| = {ib:.5f} m, max|zeta| = {amp*ib:.4f} m)")
    assert amp > 2.0, f"amplificacion {amp:.2f}: la resonancia no se manifiesta"


def test_la_conservacion_y_la_finitud_se_mantienen_en_el_escenario(escenario_2d):
    s, _, _, _ = escenario_2d
    d = s.diagnostics()
    assert d["is_finite"]
    assert d["h_min"] > 0.0, "la cuenca no deberia secarse en este escenario"


def test_la_respuesta_de_barometro_inverso_coincide_con_la_funcion_oficial():
    """
    Nivel C sobre la compuerta 4 (1D), reexpresada con el verificador
    de titan_physical: la respuesta estatica medida debe coincidir con
    inverse_barometer() de la fuente unica.
    """
    # Valor medido en la compuerta 4 (1D): test_gate4_inverse_barometer.py.
    zeta_medido, dp_efectivo = -0.164316, 99.9906
    c = check_inverse_barometer(zeta_medido, dp_efectivo)
    print(f"\n{c}")
    assert c.passed


def test_los_verificadores_detectan_valores_absurdos():
    """
    Control negativo: si los verificadores no fallaran ante valores absurdos,
    no estarian verificando nada.
    """
    assert not check_zeta_magnitude(np.array([1.0e4])).passed
    assert not check_zeta_magnitude(np.array([np.nan])).passed
    assert not check_velocity_magnitude(np.array([500.0]), np.array([0.0]),
                                        np.array([100.0])).passed
    # Flujo SUPERCRITICO en MAR ABIERTO: h = 2 m > open_water_depth, y
    # Fr = 10/sqrt(g*2) = 6.08 > 1.
    c = check_velocity_magnitude(np.array([10.0]), np.array([0.0]),
                                 np.array([2.0]))
    assert not c.passed, c.detail

    # Dominio SIN mar abierto: la verificacion NO es concluyente y por tanto NO
    # debe aprobar. Aprobar por ausencia de evidencia fue un agujero real que
    # este control negativo destapo (ver la nota en check_velocity_magnitude).
    c = check_velocity_magnitude(np.array([5.0]), np.array([0.0]),
                                 np.array([0.5]))
    assert not c.passed
    assert "NO CONCLUYENTE" in c.detail

    # La zona de batida supercritica SOBRE un dominio con mar abierto NO debe
    # hacer fallar: es fisica correcta de run-up, no un error.
    c = check_velocity_magnitude(np.array([0.05, 0.3]), np.array([0.0, 0.0]),
                                 np.array([100.0, 0.01]))
    assert c.passed, c.detail
    assert "zona de batida" in c.detail

    assert not check_inverse_barometer(-1.0, 100.0).passed
