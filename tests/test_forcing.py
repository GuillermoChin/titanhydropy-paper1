"""Pruebas unitarias de los campos de forzamiento."""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import TITAN
from core.solver_base import State
from forcing.atmospheric import (MovingGaussianPressure, StaticGaussianPressure,
                                 UniformPressure)
from forcing.friction import LinearFriction, ManningFriction, NoFriction
from forcing.wind import NoWind, UniformWind

X = np.linspace(0.0, 1.0e5, 251)
Y = np.zeros_like(X)


# --- presion ---------------------------------------------------------------
def test_presion_uniforme_tiene_gradiente_nulo():
    p = UniformPressure(p0=TITAN.p_surface).evaluate(X, Y, 123.0)
    assert np.all(p == TITAN.p_surface)
    assert np.all(np.diff(p) == 0.0)


def test_gaussiana_movil_traslada_su_centro_a_velocidad_constante():
    U = 12.0
    p = MovingGaussianPressure(amplitude=100.0, sigma=5.0e3, speed=U,
                               x0=2.0e4)
    for t in (0.0, 500.0, 2000.0):
        campo = p.evaluate(X, Y, t)
        assert X[np.argmax(campo)] == pytest.approx(p.center(t), abs=X[1] - X[0])
    assert p.center(1000.0) == pytest.approx(2.0e4 + U * 1000.0)


def test_gaussiana_movil_amplitud_y_forma():
    A, s = 150.0, 8.0e3
    p = MovingGaussianPressure(amplitude=A, sigma=s, speed=0.0, x0=5.0e4)
    campo = p.evaluate(X, Y, 0.0)
    assert np.max(campo) - TITAN.p_surface == pytest.approx(A, rel=1e-6)
    # A una distancia sigma el exceso vale A*exp(-1/2).
    en_sigma = p.evaluate(np.array([5.0e4 + s]), np.array([0.0]), 0.0)[0]
    assert en_sigma - TITAN.p_surface == pytest.approx(A * np.exp(-0.5))


def test_rampa_temporal_es_monotona_y_satura_en_uno():
    A, t_ramp = 100.0, 1000.0
    p = StaticGaussianPressure(amplitude=A, sigma=1.0e4, x0=5.0e4,
                               t_ramp=t_ramp)
    xc = np.array([5.0e4])
    yc = np.array([0.0])
    ts = np.linspace(0.0, 2.0 * t_ramp, 41)
    exceso = np.array([p.evaluate(xc, yc, t)[0] - TITAN.p_surface for t in ts])
    assert exceso[0] == pytest.approx(0.0)
    assert np.all(np.diff(exceso) >= -1e-12)
    assert exceso[-1] == pytest.approx(A, rel=1e-9)
    assert p.evaluate(xc, yc, 0.5 * t_ramp)[0] - TITAN.p_surface == \
        pytest.approx(0.5 * A, rel=1e-9)


def test_sin_rampa_el_forzamiento_es_inmediato():
    p = StaticGaussianPressure(amplitude=50.0, sigma=1.0e4, x0=5.0e4, t_ramp=0.0)
    assert p.evaluate(np.array([5.0e4]), np.array([0.0]), 0.0)[0] - \
        TITAN.p_surface == pytest.approx(50.0)


# --- viento ----------------------------------------------------------------
def test_viento_nulo_y_uniforme():
    u, v = NoWind().evaluate(X, Y, 0.0)
    assert np.all(u == 0.0) and np.all(v == 0.0)
    u, v = UniformWind(u10=7.0, v10=-2.0).evaluate(X, Y, 0.0)
    assert np.all(u == 7.0) and np.all(v == -2.0)


def test_esfuerzo_de_viento_es_cuadratico_y_alineado():
    from core.native_fvm import _wind_stress
    tau1 = _wind_stress(np.array([5.0]), np.array([0.0]))[0][0]
    tau2 = _wind_stress(np.array([10.0]), np.array([0.0]))[0][0]
    assert tau2 / tau1 == pytest.approx(4.0)
    # Viento negativo => esfuerzo negativo.
    assert _wind_stress(np.array([-5.0]), np.array([0.0]))[0][0] == \
        pytest.approx(-tau1)


# --- friccion --------------------------------------------------------------
def _estado(u, v=0.0, n=5):
    return State(zeta=np.zeros(n), u=np.full(n, u), v=np.full(n, v))


def test_friccion_nula():
    ax, ay = NoFriction().stress(_estado(3.0), np.full(5, 100.0))
    assert np.all(ax == 0.0) and np.all(ay == 0.0)


def test_friccion_lineal_se_opone_al_movimiento_y_escala_con_r():
    r = 1e-4
    ax, ay = LinearFriction(r=r).stress(_estado(3.0, -1.0), np.full(5, 100.0))
    assert np.all(ax == pytest.approx(-r * 3.0))
    assert np.all(ay == pytest.approx(+r * 1.0))
    ax2, _ = LinearFriction(r=2 * r).stress(_estado(3.0), np.full(5, 100.0))
    assert np.all(ax2 == pytest.approx(2.0 * np.asarray(ax)))


def test_friccion_de_manning_es_cuadratica_y_se_anula_en_seco():
    m = ManningFriction(n=0.02, g=TITAN.g, min_depth=1e-3)
    h = np.full(5, 100.0)
    a1 = m.stress(_estado(1.0), h)[0][0]
    a2 = m.stress(_estado(2.0), h)[0][0]
    assert a2 / a1 == pytest.approx(4.0)
    assert a1 < 0.0
    seco = m.stress(_estado(2.0), np.zeros(5))[0]
    assert np.all(seco == 0.0)
