"""
Sprint 0/1: el motor nativo cumple el contrato abstracto `Solver` y respeta
ADR-001 (la frontera publica son variables primitivas).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.native_fvm import NativeFVMSolver
from core.solver_base import (Domain, ForcingBundle, FrictionModel,
                              PressureField, Solver, SolverConfig, State,
                              WindField)
from domain.geometry import flat_channel, quiescent_state
from forcing.atmospheric import StaticGaussianPressure, UniformPressure
from forcing.friction import LinearFriction, NoFriction
from forcing.wind import NoWind, UniformWind


def _setup(nx=64, depth=160.0, **cfg):
    dom = flat_channel(length=1.0e4, nx=nx, depth=depth)
    st = quiescent_state(dom)
    s = NativeFVMSolver()
    s.initialize(dom, st, ForcingBundle(coriolis_enabled=False),
                 SolverConfig(**cfg))
    return s


def test_es_subclase_del_contrato():
    assert issubclass(NativeFVMSolver, Solver)
    assert isinstance(_setup(), Solver)


def test_forzamientos_satisfacen_sus_protocolos():
    assert isinstance(UniformPressure(), PressureField)
    assert isinstance(StaticGaussianPressure(amplitude=1.0, sigma=1.0),
                      PressureField)
    assert isinstance(NoWind(), WindField)
    assert isinstance(UniformWind(u10=1.0), WindField)
    assert isinstance(NoFriction(), FrictionModel)
    assert isinstance(LinearFriction(r=1e-4), FrictionModel)


def test_estado_publico_es_primitivo_y_es_copia():
    """ADR-001: la frontera publica expone (zeta,u,v); mutarla no toca el motor."""
    s = _setup()
    st = s.state
    assert isinstance(st, State)
    assert st.zeta.shape == st.u.shape == st.v.shape == s.x.shape
    st.zeta[:] = 999.0
    assert np.max(np.abs(s.state.zeta)) < 1e-12


def test_cfl_dinamico_usa_celeridad_mas_flujo():
    """dt = cfl*dx/max(|u|+sqrt(g h)); en reposo se reduce a cfl*dx/sqrt(g h)."""
    depth, nx, L = 160.0, 64, 1.0e4
    s = _setup(nx=nx, depth=depth, cfl=0.4)
    c = np.sqrt(s.g * depth)
    assert s.compute_stable_dt() == pytest.approx(0.4 * (L / nx) / c, rel=1e-12)

    # Con flujo no nulo el paso debe encogerse.
    dom = flat_channel(L, nx, depth)
    st = quiescent_state(dom)
    st.u[:] = 5.0
    s2 = NativeFVMSolver()
    s2.initialize(dom, st, ForcingBundle(coriolis_enabled=False),
                  SolverConfig(cfl=0.4))
    assert s2.compute_stable_dt() == pytest.approx(
        0.4 * (L / nx) / (5.0 + c), rel=1e-12)


def test_step_devuelve_dt_y_avanza_el_tiempo():
    s = _setup()
    dt = s.step()
    assert dt > 0.0
    assert s.state.t == pytest.approx(dt)
    dt2 = s.step(1.0)
    assert dt2 == 1.0
    assert s.state.t == pytest.approx(dt + 1.0)


def test_advance_del_contrato_base_llega_exactamente_a_t_end():
    s = _setup()
    s.advance(37.0)
    assert s.state.t == pytest.approx(37.0, abs=1e-9)


def test_diagnostics_expone_los_invariantes_requeridos():
    d = _setup().diagnostics()
    for k in ("mass", "momentum_x", "momentum_y", "energy", "is_finite", "t"):
        assert k in d


def test_esquemas_no_implementados_fallan_explicitamente():
    dom = flat_channel(1.0e4, 64, 160.0)
    st = quiescent_state(dom)
    fb = ForcingBundle(coriolis_enabled=False)
    with pytest.raises(NotImplementedError):
        NativeFVMSolver().initialize(dom, st, fb, SolverConfig(scheme="roe"))
    with pytest.raises(NotImplementedError):
        NativeFVMSolver().initialize(dom, st, fb,
                                     SolverConfig(time_integrator="euler"))


def test_frontera_desconocida_falla():
    dom = flat_channel(1.0e4, 64, 160.0)
    st = quiescent_state(dom)
    cfg = SolverConfig(bc={"west": "sponge", "east": "reflective"})
    with pytest.raises(ValueError):
        NativeFVMSolver().initialize(dom, st,
                                     ForcingBundle(coriolis_enabled=False), cfg)


def test_malla_no_uniforme_falla():
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
    dom = Domain(x=x, y=np.array([0.0]), bathymetry=np.full(6, 100.0),
                 lat0_deg=78.0)
    st = State(np.zeros(6), np.zeros(6), np.zeros(6))
    with pytest.raises(ValueError):
        NativeFVMSolver().initialize(dom, st,
                                     ForcingBundle(coriolis_enabled=False),
                                     SolverConfig())
