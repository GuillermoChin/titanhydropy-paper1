"""
COMPUERTA 1 - Lago en reposo (propiedad well-balanced).
=======================================================
Con velocidad inicial nula y superficie plana sobre batimetria variable, zeta y
las velocidades deben permanecer a cero hasta precision de maquina.

Si esta compuerta falla, todo lo demas es ruido. Es la primera prueba.

SUPUESTOS: fondo con monticulo gaussiano, fronteras reflectivas, sin
forzamiento, CFL 0.45, SSPRK3, HLLC + reconstruccion hidrostatica de Audusse.
Se prueban orden 1 y orden 2 de reconstruccion.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.native_fvm import NativeFVMSolver
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import (flat_channel, quiescent_state,
                             variable_bed_channel)

pytestmark = pytest.mark.gate

L = 2.0e4          # longitud del canal [m]
NX = 200           # celdas
DEPTH = 160.0      # profundidad media [m] (orden de Ligeia Mare)
N_STEPS = 400      # pasos de integracion


def _bumpy_domain():
    return variable_bed_channel(length=L, nx=NX, depth_mean=DEPTH,
                                bump_height=120.0, bump_center=0.5 * L,
                                bump_width=0.05 * L)


def _run(domain, order, zeta0=0.0, bc=None, n_steps=N_STEPS):
    cfg = SolverConfig(cfl=0.45, bc=bc or {"west": "reflective",
                                           "east": "reflective"})
    s = NativeFVMSolver(order=order)
    s.initialize(domain, quiescent_state(domain, zeta0),
                 ForcingBundle(coriolis_enabled=False), cfg)
    for _ in range(n_steps):
        s.step()
    return s


@pytest.mark.parametrize("order", [1, 2])
def test_reposo_exacto_sobre_batimetria_variable(order):
    """zeta = 0 y u = 0 en aritmetica EXACTA (cero bit a bit)."""
    s = _run(_bumpy_domain(), order)
    st = s.state
    assert np.max(np.abs(st.zeta)) == 0.0, (
        f"orden {order}: max|zeta| = {np.max(np.abs(st.zeta)):.3e} != 0")
    assert np.max(np.abs(st.u)) == 0.0
    assert np.max(np.abs(st.v)) == 0.0


@pytest.mark.parametrize("order", [1, 2])
def test_reposo_exacto_sobre_fondo_plano(order):
    s = _run(flat_channel(L, NX, DEPTH), order)
    assert np.max(np.abs(s.state.zeta)) == 0.0
    assert np.max(np.abs(s.state.u)) == 0.0


@pytest.mark.parametrize("bc", ["reflective", "transmissive", "periodic"])
def test_reposo_exacto_con_cada_frontera(bc):
    """Ninguna condicion de frontera puede inyectar movimiento espurio."""
    s = _run(_bumpy_domain(), order=2, bc={"west": bc, "east": bc},
             n_steps=200)
    assert np.max(np.abs(s.state.zeta)) == 0.0
    assert np.max(np.abs(s.state.u)) == 0.0


def test_reposo_con_superficie_desplazada_a_precision_de_maquina():
    """
    Superficie plana en zeta0 != 0: el equilibrio ya no es exacto bit a bit
    porque (eta - b) + b no es exacto en punto flotante, pero el residuo debe
    quedar al nivel del epsilon de maquina escalado por la altura de columna.
    """
    zeta0 = 3.0
    s = _run(_bumpy_domain(), order=2, zeta0=zeta0)
    st = s.state
    tol = 1e-11 * (DEPTH + zeta0)
    assert np.max(np.abs(st.zeta - zeta0)) < tol, np.max(np.abs(st.zeta - zeta0))
    assert np.max(np.abs(st.u)) < 1e-11


def test_masa_conservada_exactamente_en_reposo():
    s = _run(_bumpy_domain(), order=2)
    d = s.diagnostics()
    assert d["mass_rel_error"] == 0.0
    assert d["is_finite"]
