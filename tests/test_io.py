"""Pruebas de persistencia: el estado guardado se recupera intacto y con
la lista de constantes TO_VERIFY vigente en la corrida."""

from __future__ import annotations

import numpy as np

from core.native_fvm import NativeFVMSolver
from core.solver_base import ForcingBundle, SolverConfig
from domain.geometry import flat_channel, quiescent_state
from titan_io.writers import TimeSeriesRecorder, load_state, save_state


def _solver():
    dom = flat_channel(1.0e4, 64, 160.0)
    s = NativeFVMSolver()
    s.initialize(dom, quiescent_state(dom, zeta0=0.5),
                 ForcingBundle(coriolis_enabled=False), SolverConfig())
    return s


def test_ida_y_vuelta_de_estado(tmp_path):
    s = _solver()
    s.step()
    p = save_state(tmp_path / "salida.npz", s, caso="prueba")
    st, meta = load_state(p)
    assert np.allclose(st.zeta, s.state.zeta)
    assert st.t == s.state.t
    assert meta["caso"] == "prueba"


def test_la_salida_embebe_la_auditoria_de_procedencia(tmp_path):
    p = save_state(tmp_path / "salida.npz", _solver())
    _, meta = load_state(p)
    assert meta["to_verify_constants"], "falta la lista TO_VERIFY"
    assert "diagnostics" in meta


def test_el_grabador_respeta_el_intervalo_de_muestreo():
    s = _solver()
    rec = TimeSeriesRecorder(every_seconds=10.0)
    assert rec.maybe_sample(s)           # t = 0
    s.step(4.0)
    assert not rec.maybe_sample(s)       # t = 4 < 10
    s.step(7.0)
    assert rec.maybe_sample(s)           # t = 11 >= 10
    t, zeta, u = rec.as_arrays()
    assert t.shape == (2,)
    assert zeta.shape == (2, s.nx)
    assert u.shape == (2, s.nx)
