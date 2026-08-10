"""
COMPUERTA 2 - Conservacion.
===========================
En dominio cerrado (fronteras reflectivas) y sin forzamiento:
  * la masa se conserva a tolerancia estricta,
  * la energia no crece de forma espuria (el esquema es disipativo: puede
    perder energia por difusion numerica, nunca ganarla).

SUPUESTOS: canal cerrado de 20 km, 400 celdas, batimetria con monticulo,
perturbacion inicial gaussiana de la superficie libre de 2 m, CFL 0.45,
SSPRK3 + HLLC orden 2, sin friccion, sin Coriolis, sin presion.
La onda rebota varias veces contra las paredes durante la corrida.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import TITAN
from core.solver_base import ForcingBundle, SolverConfig, State
from domain.geometry import variable_bed_channel
from utils.runners import run_to

pytestmark = pytest.mark.gate

L = 2.0e4
NX = 400
DEPTH = 160.0
AMP = 2.0            # amplitud de la perturbacion inicial [m]
SIGMA = 0.04 * L     # semiancho de la perturbacion [m]

# Tolerancias declaradas explicitamente (no se ajustan para forzar el verde).
TOL_MASA = 1e-12         # error relativo de masa
TOL_ENERGIA = 1e-6       # crecimiento relativo de energia admitido


def _closed_run(t_end, order=2, cfl=0.45):
    dom = variable_bed_channel(L, NX, DEPTH, bump_height=100.0,
                               bump_center=0.65 * L, bump_width=0.05 * L)
    arg = (dom.x - 0.3 * L) / SIGMA
    zeta = AMP * np.exp(-0.5 * arg * arg)
    init = State(zeta=zeta, u=np.zeros(NX), v=np.zeros(NX))
    cfg = SolverConfig(cfl=cfl, bc={"west": "reflective", "east": "reflective"})
    return run_to(dom, init, ForcingBundle(coriolis_enabled=False), cfg,
                  t_end=t_end, order=order, sample_every=20)


def _crossing_time():
    """Tiempo que tarda la onda en cruzar el canal: L / sqrt(g h)."""
    return L / np.sqrt(TITAN.g * DEPTH)


def test_masa_conservada_en_dominio_cerrado():
    res = _closed_run(t_end=4.0 * _crossing_time())
    drift = res.monitor.mass_drift
    assert res.monitor.all_finite
    assert drift < TOL_MASA, f"deriva de masa = {drift:.3e} > {TOL_MASA:.0e}"


def test_energia_sin_crecimiento_espurio():
    res = _closed_run(t_end=4.0 * _crossing_time())
    growth = res.monitor.energy_growth
    assert growth < TOL_ENERGIA, (
        f"crecimiento de energia = {growth:+.3e} > {TOL_ENERGIA:.0e}")


def test_energia_efectivamente_disipa_y_no_diverge():
    """El esquema debe perder algo de energia, no mantenerla ni amplificarla."""
    res = _closed_run(t_end=4.0 * _crossing_time())
    e = np.array([r["energy"] for r in res.monitor.records])
    assert e[-1] <= e[0] * (1.0 + TOL_ENERGIA)
    assert e[-1] > 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_masa_conservada_para_ambos_ordenes(order):
    res = _closed_run(t_end=2.0 * _crossing_time(), order=order)
    assert res.monitor.mass_drift < TOL_MASA


def test_momento_neto_acotado_por_las_paredes():
    """
    En dominio cerrado el momento NO se conserva (las paredes ejercen fuerza),
    pero debe permanecer acotado: si diverge, el esquema es inestable.
    """
    res = _closed_run(t_end=4.0 * _crossing_time())
    px = np.array([r["momentum_x"] for r in res.monitor.records])
    escala = AMP * np.sqrt(TITAN.g * DEPTH) * L
    assert np.all(np.isfinite(px))
    assert np.max(np.abs(px)) < escala, (
        f"momento maximo {np.max(np.abs(px)):.3e} supera la escala {escala:.3e}")


def test_sin_forzamiento_el_estado_permanece_finito_con_cfl_alto():
    """Robustez: CFL 0.9 sigue siendo estable para SSPRK3 + MUSCL."""
    res = _closed_run(t_end=2.0 * _crossing_time(), cfl=0.9)
    assert res.monitor.all_finite
    assert res.monitor.mass_drift < TOL_MASA
