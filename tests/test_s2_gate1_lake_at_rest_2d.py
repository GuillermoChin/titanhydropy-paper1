"""
COMPUERTA 1 (2D): Lago en reposo 2D (well-balanced).
====================================================
Cero EXACTO bit a bit sobre batimetria 2D variable. Primera prueba: si falla,
todo lo demas de la validacion 2D es ruido.

La exactitud no es automatica al pasar a 2D. Depende de que:
  * el kernel direccional sea el MISMO que el validado en 1D,
  * SSPRK3 este en forma incremental,
  * las fronteras de los CUATRO lados y las esquinas no inyecten desequilibrio,
  * la suma no escindida de las dos direcciones no introduzca redondeo (con
    ambas contribuciones exactamente cero, 0.0 + 0.0 = 0.0 es exacto).

SUPUESTOS: cuenca de 20x15 km, 80x60 celdas (dx != dy a proposito, para que un
error de simetria de ejes no pase inadvertido), profundidad media 160 m con
monticulo gaussiano de 120 m, CFL 0.45, SSPRK3, HLLC + Audusse. 300 pasos.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry2d import (bumpy_basin, flat_basin, quiescent_state_2d)

pytestmark = pytest.mark.gate

LX, LY = 2.0e4, 1.5e4
NX, NY = 80, 60          # dx = 250 m, dy = 250 m
DEPTH = 160.0
N_STEPS = 300

TODAS = ("reflective", "transmissive", "periodic")


def _bumpy():
    return bumpy_basin(LX, LY, NX, NY, depth_mean=DEPTH, bump_height=120.0,
                       bump_center=(0.45 * LX, 0.55 * LY), bump_width=0.08 * LX)


def _run(domain, order=2, zeta0=0.0, bc=None, n_steps=N_STEPS, limiter="minmod"):
    if bc is None:
        bc = {"west": "reflective", "east": "reflective",
              "south": "reflective", "north": "reflective"}
    cfg = SolverConfig(cfl=0.45, bc=bc)
    s = NativeFVMSolver2D(order=order, limiter=limiter)
    s.initialize(domain, quiescent_state_2d(domain, zeta0),
                 ForcingBundle(coriolis_enabled=False), cfg)
    for _ in range(n_steps):
        s.step()
    return s


@pytest.mark.parametrize("order", [1, 2])
def test_reposo_exacto_sobre_batimetria_2d_variable(order):
    """zeta = 0, u = 0 y v = 0 en aritmetica EXACTA (cero bit a bit)."""
    st = _run(_bumpy(), order=order).state
    assert np.max(np.abs(st.zeta)) == 0.0, (
        f"orden {order}: max|zeta| = {np.max(np.abs(st.zeta)):.3e} != 0")
    assert np.max(np.abs(st.u)) == 0.0
    assert np.max(np.abs(st.v)) == 0.0


@pytest.mark.parametrize("limiter", ["minmod", "mc"])
def test_reposo_exacto_con_cada_limitador_tvd(limiter):
    st = _run(_bumpy(), limiter=limiter).state
    assert np.max(np.abs(st.zeta)) == 0.0
    assert np.max(np.abs(st.u)) == 0.0 and np.max(np.abs(st.v)) == 0.0


def test_reposo_exacto_sobre_fondo_plano():
    st = _run(flat_basin(LX, LY, NX, NY, DEPTH)).state
    assert np.max(np.abs(st.zeta)) == 0.0
    assert np.max(np.abs(st.u)) == 0.0 and np.max(np.abs(st.v)) == 0.0


@pytest.mark.parametrize("kind", TODAS)
def test_reposo_exacto_con_cada_frontera_en_los_cuatro_lados(kind):
    """Ninguna de las cuatro fronteras puede inyectar movimiento espurio."""
    bc = {"west": kind, "east": kind, "south": kind, "north": kind}
    st = _run(_bumpy(), bc=bc, n_steps=150).state
    assert np.max(np.abs(st.zeta)) == 0.0
    assert np.max(np.abs(st.u)) == 0.0 and np.max(np.abs(st.v)) == 0.0


def test_reposo_exacto_con_fronteras_mixtas_y_esquinas():
    """Las esquinas se rellenan en la segunda pasada: hay que probarlas."""
    bc = {"west": "reflective", "east": "transmissive",
          "south": "transmissive", "north": "reflective"}
    st = _run(_bumpy(), bc=bc, n_steps=150).state
    assert np.max(np.abs(st.zeta)) == 0.0
    assert np.max(np.abs(st.u)) == 0.0 and np.max(np.abs(st.v)) == 0.0


def test_reposo_exacto_con_coriolis_activo():
    """
    Coriolis actua sobre el momento; en reposo (hu = hv = 0) su contribucion es
    exactamente nula y no puede romper el equilibrio.
    """
    dom = _bumpy()
    cfg = SolverConfig(cfl=0.45, bc={"west": "reflective", "east": "reflective",
                                     "south": "reflective", "north": "reflective"})
    s = NativeFVMSolver2D()
    s.initialize(dom, quiescent_state_2d(dom),
                 ForcingBundle(coriolis_enabled=True), cfg)
    assert s._f != 0.0, "Coriolis deberia estar activo en lat0=78 N"
    for _ in range(150):
        s.step()
    assert np.max(np.abs(s.state.zeta)) == 0.0
    assert np.max(np.abs(s.state.u)) == 0.0
    assert np.max(np.abs(s.state.v)) == 0.0


def test_reposo_con_superficie_desplazada_a_precision_de_maquina():
    """
    Con zeta0 != 0 el equilibrio ya no es exacto bit a bit ((eta-b)+b no lo es en
    punto flotante), pero el residuo debe quedar al nivel del epsilon de maquina
    escalado por la altura de columna.
    """
    zeta0 = 3.0
    st = _run(_bumpy(), zeta0=zeta0).state
    tol = 1e-11 * (DEPTH + zeta0)
    assert np.max(np.abs(st.zeta - zeta0)) < tol
    assert np.max(np.abs(st.u)) < 1e-11 and np.max(np.abs(st.v)) < 1e-11


def test_masa_conservada_exactamente_en_reposo_2d():
    d = _run(_bumpy()).diagnostics()
    assert d["mass_rel_error"] == 0.0
    assert d["is_finite"]


def test_malla_anisotropa_tambien_da_cero_exacto():
    """dx != dy: un error de simetria de ejes se delataria aqui."""
    dom = bumpy_basin(2.0e4, 1.0e4, 100, 25, DEPTH, 120.0,
                      (0.4 * 2.0e4, 0.5e4), 0.08 * 2.0e4)
    st = _run(dom, n_steps=150).state
    assert dom.x.size == 100 and dom.y.size == 25
    assert np.max(np.abs(st.zeta)) == 0.0
    assert np.max(np.abs(st.u)) == 0.0 and np.max(np.abs(st.v)) == 0.0
