"""
SPRINT 2 - COMPUERTA 2: Conservacion 2D.
========================================
En cuenca cerrada 2D y sin forzamiento:
  * masa conservada a tolerancia estricta,
  * energia sin crecimiento espurio.

SUPUESTOS: cuenca cerrada de 20x15 km, 100x75 celdas (dx = dy = 200 m),
batimetria con monticulo, perturbacion inicial gaussiana 2D de 2 m,
CFL 0.45, SSPRK3 + HLLC orden 2, sin friccion, sin presion. La onda rebota
varias veces contra las cuatro paredes durante la corrida.

TOLERANCIAS (declaradas, no ajustadas a posteriori):
  masa    < 1e-12 relativo
  energia < 1e-6  de crecimiento relativo
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import TITAN
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry2d import bumpy_basin, gaussian_surface_state_2d
from validation.conservation import ConservationMonitor

pytestmark = pytest.mark.gate

LX, LY = 2.0e4, 1.5e4
NX, NY = 100, 75
DEPTH = 160.0
AMP = 2.0
SIGMA = 0.06 * LX

TOL_MASA = 1e-12
TOL_ENERGIA = 1e-6

CERRADA = {"west": "reflective", "east": "reflective",
           "south": "reflective", "north": "reflective"}


def _crossing_time():
    return LX / np.sqrt(TITAN.g * DEPTH)


def _run(t_end, order=2, cfl=0.45, coriolis=False, sample_every=20):
    dom = bumpy_basin(LX, LY, NX, NY, DEPTH, bump_height=100.0,
                      bump_center=(0.65 * LX, 0.4 * LY), bump_width=0.07 * LX)
    init = gaussian_surface_state_2d(dom, AMP, SIGMA,
                                     center=(0.3 * LX, 0.6 * LY))
    s = NativeFVMSolver2D(order=order)
    s.initialize(dom, init, ForcingBundle(coriolis_enabled=coriolis),
                 SolverConfig(cfl=cfl, bc=CERRADA))
    mon = ConservationMonitor()
    mon.sample(s)
    n = 0
    while s.state.t < t_end:
        dt = min(s.compute_stable_dt(), t_end - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)
        n += 1
        if n % sample_every == 0:
            mon.sample(s)
    mon.sample(s)
    return s, mon


def test_masa_conservada_en_cuenca_cerrada_2d():
    _, mon = _run(t_end=3.0 * _crossing_time())
    assert mon.all_finite
    assert mon.mass_drift < TOL_MASA, f"deriva de masa = {mon.mass_drift:.3e}"


def test_energia_sin_crecimiento_espurio_2d():
    _, mon = _run(t_end=3.0 * _crossing_time())
    assert mon.energy_growth < TOL_ENERGIA, (
        f"crecimiento de energia = {mon.energy_growth:+.3e}")


def test_energia_disipa_y_no_diverge():
    _, mon = _run(t_end=3.0 * _crossing_time())
    e = np.array([r["energy"] for r in mon.records])
    assert e[-1] <= e[0] * (1.0 + TOL_ENERGIA)
    assert e[-1] > 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_masa_conservada_para_ambos_ordenes(order):
    _, mon = _run(t_end=1.5 * _crossing_time(), order=order)
    assert mon.mass_drift < TOL_MASA


def test_coriolis_no_crea_ni_destruye_masa_ni_energia():
    """
    La fuerza de Coriolis es perpendicular a la velocidad: no realiza trabajo.
    Con integracion explicita introduce un error O(dt^3) por paso, que debe
    quedar muy por debajo de la tolerancia de energia.
    """
    _, mon = _run(t_end=2.0 * _crossing_time(), coriolis=True)
    assert mon.mass_drift < TOL_MASA
    assert mon.energy_growth < TOL_ENERGIA


def test_estable_con_cfl_alto():
    """CFL 0.8 en esquema no escindido debe seguir siendo estable."""
    _, mon = _run(t_end=1.5 * _crossing_time(), cfl=0.8)
    assert mon.all_finite
    assert mon.mass_drift < TOL_MASA


def test_momento_acotado_por_las_paredes():
    """En cuenca cerrada el momento no se conserva, pero no puede diverger."""
    _, mon = _run(t_end=3.0 * _crossing_time())
    escala = AMP * np.sqrt(TITAN.g * DEPTH) * LX * LY
    for clave in ("momentum_x", "momentum_y"):
        p = np.array([r[clave] for r in mon.records])
        assert np.all(np.isfinite(p))
        assert np.max(np.abs(p)) < escala, clave


def test_cfl_2d_es_mas_restrictivo_que_el_1d():
    """
    Verificacion del supuesto declarado: en un esquema NO escindido con dx = dy
    y fluido en reposo, dt_2d = dt_1d / 2 porque los dos ejes comparten el
    presupuesto de Courant.
    """
    from domain.geometry import flat_channel, quiescent_state
    from domain.geometry2d import flat_basin, quiescent_state_2d
    from core.native_fvm import NativeFVMSolver

    L, n, cfl = 2.0e4, 100, 0.45
    d1 = flat_channel(L, n, DEPTH)
    s1 = NativeFVMSolver()
    s1.initialize(d1, quiescent_state(d1), ForcingBundle(coriolis_enabled=False),
                  SolverConfig(cfl=cfl))
    d2 = flat_basin(L, L, n, n, DEPTH)
    s2 = NativeFVMSolver2D()
    s2.initialize(d2, quiescent_state_2d(d2),
                  ForcingBundle(coriolis_enabled=False),
                  SolverConfig(cfl=cfl, bc=CERRADA))
    assert s2.dx == pytest.approx(s1.dx)
    assert s2.compute_stable_dt() == pytest.approx(0.5 * s1.compute_stable_dt(),
                                                   rel=1e-12)
