"""
COMPUERTA 3 - Rotura de presa 1D contra la solucion exacta de Riemann.
======================================================================
La solucion numerica debe converger a la solucion exacta bajo refinamiento de
malla, con orden de convergencia medido y reportado.

SUPUESTOS DEL CASO
------------------
* Canal de 10 km, fondo plano h0 = 100 m, discontinuidad en x = 5 km.
* Estado inicial: h_izq = 200 m, h_der = 100 m, velocidad nula.
* g = TITAN.g (gravedad de Titan; el caso es adimensionalmente el clasico).
* Fronteras transmisivas; t_final = 100 s, elegido para que ni el choque ni la
  rarefaccion alcancen las fronteras (c ~ 16.4 m/s, recorrido < 2.5 km).
* CFL 0.45, SSPRK3, HLLC, MUSCL-minmod.
* Resoluciones: 200 / 400 / 800 / 1600 celdas.

ORDEN ESPERADO
--------------
La solucion contiene un CHOQUE. Para leyes de conservacion con discontinuidades
el orden en norma L1 esta limitado a ~1 sea cual sea el orden formal del
esquema; la reconstruccion MUSCL mejora la constante, no la pendiente asintotica.
Por eso la compuerta exige orden L1 >= 0.8 y NO exige orden 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import TITAN
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import cell_centers, flat_channel, step_state
from utils.runners import l1_error, run_to
from validation.analytic import dam_break_exact, star_depth
from validation.conservation import convergence_order

pytestmark = pytest.mark.gate

L = 1.0e4
X0 = 0.5 * L
H_BED = 100.0
H_LEFT = 200.0
H_RIGHT = 100.0
T_END = 100.0
RESOLUCIONES = (200, 400, 800, 1600)
ORDEN_L1_MINIMO = 0.8


def _numeric(nx, order=2):
    dom = flat_channel(L, nx, H_BED)
    init = step_state(dom, H_LEFT, H_RIGHT, X0)
    cfg = SolverConfig(cfl=0.45,
                       bc={"west": "transmissive", "east": "transmissive"})
    res = run_to(dom, init, ForcingBundle(coriolis_enabled=False), cfg,
                 t_end=T_END, order=order, sample_every=10_000)
    st = res.state
    return dom.x, st.zeta + H_BED, st.u, L / nx


# ---------------------------------------------------------------------------
# 3a. La solucion exacta es correcta por si misma
# ---------------------------------------------------------------------------
def test_solucion_exacta_satisface_las_relaciones_de_salto():
    g = TITAN.g
    hs = star_depth(H_LEFT, 0.0, H_RIGHT, 0.0, g)
    assert H_RIGHT < hs < H_LEFT, hs
    # La region estrella debe aparecer en el perfil y ser plana.
    x, _ = cell_centers(L, 2000)
    h, u = dam_break_exact(x, T_END, H_LEFT, H_RIGHT, g, x0=X0)
    assert np.all(h > 0.0)
    # Estados lejanos intactos.
    assert h[0] == pytest.approx(H_LEFT)
    assert h[-1] == pytest.approx(H_RIGHT)
    assert u[0] == pytest.approx(0.0)
    assert u[-1] == pytest.approx(0.0)
    # Meseta estrella: existe una franja donde h == hs y u == us.
    meseta = np.isclose(h, hs, rtol=0, atol=1e-9)
    assert meseta.sum() > 10
    assert np.allclose(u[meseta], u[meseta][0])


def test_solucion_exacta_en_t0_reproduce_el_dato_inicial():
    x, _ = cell_centers(L, 100)
    h, u = dam_break_exact(x, 0.0, H_LEFT, H_RIGHT, TITAN.g, x0=X0)
    assert np.all(h[x < X0] == H_LEFT)
    assert np.all(h[x > X0] == H_RIGHT)
    assert np.all(u == 0.0)


# ---------------------------------------------------------------------------
# 3b. Convergencia del motor a la solucion exacta
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_convergencia_a_la_solucion_exacta_de_riemann():
    errores_h, errores_u = [], []
    for nx in RESOLUCIONES:
        x, h_num, u_num, dx = _numeric(nx)
        h_ex, u_ex = dam_break_exact(x, T_END, H_LEFT, H_RIGHT, TITAN.g, x0=X0)
        errores_h.append(l1_error(h_num, h_ex, dx))
        errores_u.append(l1_error(u_num, u_ex, dx))

    errores_h = np.array(errores_h)
    errores_u = np.array(errores_u)
    ordenes_h, pendiente_h = convergence_order(np.array(RESOLUCIONES), errores_h)
    ordenes_u, pendiente_u = convergence_order(np.array(RESOLUCIONES), errores_u)

    print("\n--- COMPUERTA 3: convergencia dam-break (norma L1) ---")
    print(f"{'nx':>6} {'L1(h)':>14} {'orden':>8} {'L1(u)':>14} {'orden':>8}")
    for i, nx in enumerate(RESOLUCIONES):
        oh = f"{ordenes_h[i-1]:8.3f}" if i else " " * 8
        ou = f"{ordenes_u[i-1]:8.3f}" if i else " " * 8
        print(f"{nx:6d} {errores_h[i]:14.6e} {oh} {errores_u[i]:14.6e} {ou}")
    print(f"orden global (ajuste log-log): h = {pendiente_h:.3f}, "
          f"u = {pendiente_u:.3f}")

    assert np.all(np.diff(errores_h) < 0.0), "el error en h no decrece"
    assert np.all(np.diff(errores_u) < 0.0), "el error en u no decrece"
    assert pendiente_h >= ORDEN_L1_MINIMO, f"orden L1(h) = {pendiente_h:.3f}"
    assert pendiente_u >= ORDEN_L1_MINIMO, f"orden L1(u) = {pendiente_u:.3f}"


def test_la_meseta_estrella_numerica_coincide_con_la_exacta():
    """Verificacion puntual: altura y velocidad de la region estrella."""
    x, h_num, u_num, _ = _numeric(1600)
    hs = star_depth(H_LEFT, 0.0, H_RIGHT, 0.0, TITAN.g)
    h_ex, u_ex = dam_break_exact(x, T_END, H_LEFT, H_RIGHT, TITAN.g, x0=X0)
    meseta = np.isclose(h_ex, hs, rtol=0, atol=1e-9)
    # Se descartan 10 celdas junto a cada extremo de la meseta (zona de
    # transicion numerica del choque y de la cola de la rarefaccion).
    idx = np.flatnonzero(meseta)[10:-10]
    assert idx.size > 50
    assert np.max(np.abs(h_num[idx] - hs)) / hs < 5e-3
    assert np.max(np.abs(u_num[idx] - u_ex[idx])) / abs(u_ex[idx][0]) < 5e-3


def test_la_masa_se_conserva_durante_la_rotura():
    """Fronteras transmisivas: la masa cambia solo por el flujo saliente, nulo
    mientras las ondas no alcancen los bordes."""
    dom = flat_channel(L, 800, H_BED)
    init = step_state(dom, H_LEFT, H_RIGHT, X0)
    cfg = SolverConfig(cfl=0.45,
                       bc={"west": "transmissive", "east": "transmissive"})
    res = run_to(dom, init, ForcingBundle(coriolis_enabled=False), cfg,
                 t_end=T_END, order=2, sample_every=25)
    assert res.monitor.mass_drift < 1e-12
