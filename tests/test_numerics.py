"""
Pruebas unitarias de los bloques numericos: HLLC, reconstruccion MUSCL,
condiciones de frontera y utilidades de convergencia.
"""

from __future__ import annotations

import numpy as np
import pytest

from constants.titan_params import TITAN
from core.boundary import NG, apply_bc, extend_bathymetry, validate_bc
from core.reconstruction import (hydrostatic_reconstruction, minmod_slope,
                                 muscl_edges)
from core.riemann import hllc_flux, hydrostatic_pressure, wave_speeds
from validation.conservation import convergence_order

G = TITAN.g


# --- HLLC ------------------------------------------------------------------
def test_hllc_en_estado_uniforme_en_reposo_devuelve_solo_presion():
    """Requisito EXACTO (bit a bit) del que depende la compuerta 1."""
    h = np.array([160.0, 85.0, 3.7])
    z = np.zeros_like(h)
    Fh, Fhu, Fhv, _ = hllc_flux(h, z, z, h, z, z, G)
    assert np.all(Fh == 0.0)
    assert np.all(Fhu == hydrostatic_pressure(h, G))
    assert np.all(Fhv == 0.0)


def test_hllc_es_consistente_con_el_flujo_fisico_en_estado_uniforme_con_flujo():
    h = np.array([100.0])
    u = np.array([2.0])
    z = np.array([0.0])
    Fh, Fhu, _, _ = hllc_flux(h, u, z, h, u, z, G)
    assert Fh[0] == pytest.approx(h[0] * u[0])
    assert Fhu[0] == pytest.approx(h[0] * u[0] ** 2 + 0.5 * G * h[0] ** 2)


def test_hllc_es_antisimetrico_bajo_reflexion_espacial():
    """F(qL,qR) reflejado debe coincidir con -F de los estados intercambiados."""
    hL, uL, hR, uR = 150.0, 1.5, 90.0, -0.3
    z = np.array([0.0])
    a = hllc_flux(np.array([hL]), np.array([uL]), z,
                  np.array([hR]), np.array([uR]), z, G)
    b = hllc_flux(np.array([hR]), np.array([-uR]), z,
                  np.array([hL]), np.array([-uL]), z, G)
    assert a[0][0] == pytest.approx(-b[0][0])   # flujo de masa cambia de signo
    assert a[1][0] == pytest.approx(b[1][0])    # flujo de momento no


def test_velocidades_de_onda_encierran_las_caracteristicas():
    hL, uL, hR, uR = 200.0, 0.0, 100.0, 0.0
    sL, sR = wave_speeds(np.array([hL]), np.array([uL]),
                         np.array([hR]), np.array([uR]), G, 1e-12)
    assert sL[0] <= uL - np.sqrt(G * hL)
    assert sR[0] >= uR + np.sqrt(G * hR)


def test_hllc_maneja_estado_seco_sin_nan():
    z = np.array([0.0])
    out = hllc_flux(np.array([100.0]), z, z, np.array([0.0]), z, z, G)
    assert all(np.all(np.isfinite(f)) for f in out)
    out2 = hllc_flux(np.array([0.0]), z, z, np.array([0.0]), z, z, G)
    assert all(np.all(np.isfinite(f)) for f in out2)


# --- reconstruccion --------------------------------------------------------
def test_minmod_anula_la_pendiente_en_los_extremos():
    q = np.array([0.0, 1.0, 5.0, 1.0, 0.0])
    s = minmod_slope(q)
    assert s[2] == 0.0            # maximo local
    assert s[1] == pytest.approx(1.0)
    assert s[3] == pytest.approx(-1.0)


def test_minmod_reproduce_pendiente_exacta_en_campo_lineal():
    q = 3.0 * np.arange(7) + 1.0
    s = minmod_slope(q)
    assert np.allclose(s[1:-1], 3.0)


def test_muscl_orden_uno_no_reconstruye():
    q = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    qL, qR = muscl_edges(q, order=1)
    assert np.all(qL == q) and np.all(qR == q)


def test_muscl_orden_dos_es_conservativo_en_media():
    q = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    qL, qR = muscl_edges(q, order=2)
    assert np.allclose(0.5 * (qL + qR), q)


def test_reconstruccion_hidrostatica_iguala_alturas_en_reposo():
    """Con eta identico a ambos lados, h*_L == h*_R (base del well-balanced)."""
    bL = np.array([-160.0, -80.0])
    bR = np.array([-90.0, -155.0])
    hL, hR = -bL, -bR                       # eta = 0 exactamente
    hsL, hsR = hydrostatic_reconstruction(hL, bL, hR, bR)
    assert np.all(hsL == hsR)
    assert np.all(hsL == np.maximum(0.0, -np.maximum(bL, bR)))


def test_reconstruccion_hidrostatica_nunca_devuelve_altura_negativa():
    hsL, hsR = hydrostatic_reconstruction(np.array([0.5]), np.array([-1.0]),
                                          np.array([0.1]), np.array([5.0]))
    assert hsL[0] >= 0.0 and hsR[0] >= 0.0


# --- fronteras -------------------------------------------------------------
def test_validate_bc_rechaza_tipos_y_periodicidad_incoherente():
    assert validate_bc({"west": "reflective", "east": "transmissive"}) == \
        ("reflective", "transmissive")
    with pytest.raises(ValueError):
        validate_bc({"west": "absorbente", "east": "reflective"})
    with pytest.raises(ValueError):
        validate_bc({"west": "periodic", "east": "reflective"})


def test_frontera_reflectiva_invierte_el_momento_normal():
    n = 6
    h = np.zeros(n + 2 * NG)
    hu = np.zeros(n + 2 * NG)
    hv = np.zeros(n + 2 * NG)
    h[NG:-NG] = np.arange(1.0, n + 1.0)
    hu[NG:-NG] = np.arange(1.0, n + 1.0)
    hv[NG:-NG] = np.arange(1.0, n + 1.0)
    apply_bc(h, hu, hv, "reflective", "reflective")
    assert h[1] == h[NG] and h[0] == h[NG + 1]
    assert hu[1] == -hu[NG] and hu[0] == -hu[NG + 1]
    assert hv[1] == +hv[NG]


def test_frontera_transmisiva_copia_la_celda_del_borde():
    n = 6
    a = np.zeros(n + 2 * NG)
    a[NG:-NG] = np.arange(1.0, n + 1.0)
    b = a.copy()
    c = a.copy()
    apply_bc(a, b, c, "transmissive", "transmissive")
    assert a[0] == a[1] == a[NG]
    assert a[-1] == a[-2] == a[NG + n - 1]


def test_frontera_periodica_envuelve():
    n = 6
    a = np.zeros(n + 2 * NG)
    a[NG:-NG] = np.arange(1.0, n + 1.0)
    b, c = a.copy(), a.copy()
    apply_bc(a, b, c, "periodic", "periodic")
    assert a[0] == a[NG + n - 2] and a[1] == a[NG + n - 1]
    assert a[-2] == a[NG] and a[-1] == a[NG + 1]


def test_extend_bathymetry_no_altera_el_interior():
    b = np.array([-160.0, -150.0, -100.0, -120.0, -160.0])
    be = extend_bathymetry(b, "reflective", "reflective")
    assert np.all(be[NG:-NG] == b)
    assert be.size == b.size + 2 * NG


# --- utilidades de convergencia -------------------------------------------
def test_convergence_order_recupera_una_pendiente_conocida():
    n = np.array([100.0, 200.0, 400.0, 800.0])
    e = 3.0 * n ** -2.0                      # orden 2 exacto
    pares, pendiente = convergence_order(n, e)
    assert np.allclose(pares, 2.0)
    assert pendiente == pytest.approx(2.0)


def test_convergence_order_exige_dos_resoluciones():
    with pytest.raises(ValueError):
        convergence_order(np.array([100.0]), np.array([1.0]))
