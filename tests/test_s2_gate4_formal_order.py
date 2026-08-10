"""
SPRINT 2 - COMPUERTA 4: Orden formal de exactitud sobre solucion SUAVE.
=======================================================================
Blinda la afirmacion metodologica del Paper 1: el esquema es de orden 2 en
regiones suaves. El dam-break NO puede demostrarlo (su ~1 en L1 es el limite
teorico impuesto por el choque); esta compuerta lo hace con el Metodo de
Soluciones Manufacturadas (MMS), ver validation/mms.py.

MONTAJE (supuestos explicitos)
------------------------------
* Dominio PERIODICO en ambos ejes, 1000 m de periodo. La periodicidad elimina
  todo error de frontera: lo que se mide es discretizacion interior pura.
* Solucion manufacturada C-infinito sobre fondo ondulado suave (ver mms.py).
* g = TITAN.g. Sin friccion, sin viento, sin presion, sin Coriolis: el unico
  termino fuente es el manufacturado.
* CFL 0.45, SSPRK3. Como dt ~ dx, el error temporal de SSPRK3 va como dx^3 y
  queda subdominante frente al espacial dx^2: lo que se mide es el orden
  ESPACIAL.
* Resoluciones 32/64/128/256 por eje. t_final = 20 s (fraccion del periodo de
  la onda manufacturada).

ORDEN ESPERADO, POR LIMITADOR
-----------------------------
Los limitadores TVD degradan el orden en los EXTREMOS suaves; es una propiedad
demostrada de la clase TVD, no un defecto de esta implementacion. Por eso se
mide con los tres:
  'none'   -> pendiente centrada sin limitar. Mide el orden formal del esquema
              base. Debe dar 2. NO es TVD: jamas se usa en produccion.
  'mc'     -> limitador monotonized-central, menos disipativo.
  'minmod' -> el de produccion, el mas disipativo.
La compuerta exige orden 2 al esquema base y cuantifica el peaje del limitador
de produccion sin disimularlo.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.native_fvm import NativeFVMSolver
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import Domain, ForcingBundle, SolverConfig, State
from validation.conservation import convergence_order
from validation.mms import default_mms_1d, default_mms_2d

pytestmark = pytest.mark.gate

LX = LY = 1000.0
T_END = 20.0
CFL = 0.45
RESOLUCIONES = (32, 64, 128, 256)

PERIODICA_1D = {"west": "periodic", "east": "periodic"}
PERIODICA_2D = {"west": "periodic", "east": "periodic",
                "south": "periodic", "north": "periodic"}

ORDEN_BASE_MINIMO = 1.90     # esquema sin limitar: debe ser 2
ORDEN_TVD_MINIMO = 1.40      # limitadores TVD: degradacion admitida en extremos


# ---------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------
def _correr_1d(n, limiter):
    mms = default_mms_1d(LX)
    dx = LX / n
    x = (np.arange(n) + 0.5) * dx
    y0 = np.zeros_like(x)
    h0 = mms.rest_depth(x, y0)
    dom = Domain(x=x, y=np.array([0.0]), bathymetry=h0, lat0_deg=0.0)

    zeta, u, v = mms.exact_primitive(x, y0, 0.0)
    init = State(zeta=zeta.copy(), u=u.copy(), v=v.copy(), t=0.0)

    s = NativeFVMSolver(g=mms.g, order=2, limiter=limiter,
                        extra_source=mms.source_1d())
    s.initialize(dom, init, ForcingBundle(coriolis_enabled=False),
                 SolverConfig(cfl=CFL, bc=PERIODICA_1D))
    _avanzar(s)
    st = s.state
    h_num = st.zeta + h0
    h_ex, u_ex, _ = mms.exact(x, y0, st.t)
    return (h_num, st.u), (h_ex, u_ex), dx


def _correr_2d(n, limiter):
    mms = default_mms_2d(LX, LY)
    dx, dy = LX / n, LY / n
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(n) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    h0 = mms.rest_depth(X, Y)
    dom = Domain(x=x, y=y, bathymetry=h0, lat0_deg=0.0)

    zeta, u, v = mms.exact_primitive(X, Y, 0.0)
    init = State(zeta=zeta.copy(), u=u.copy(), v=v.copy(), t=0.0)

    s = NativeFVMSolver2D(g=mms.g, order=2, limiter=limiter,
                          extra_source=mms.source_2d())
    s.initialize(dom, init, ForcingBundle(coriolis_enabled=False),
                 SolverConfig(cfl=CFL, bc=PERIODICA_2D))
    _avanzar(s)
    st = s.state
    h_num = st.zeta + h0
    h_ex, u_ex, v_ex = mms.exact(X, Y, st.t)
    return (h_num, st.u, st.v), (h_ex, u_ex, v_ex), dx * dy


def _avanzar(s):
    while s.state.t < T_END:
        dt = min(s.compute_stable_dt(), T_END - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)


def _norma_l1(num, ex, medida):
    return float(np.sum(np.abs(num - ex)) * medida)


def _barrido(correr, limiter):
    """Devuelve (errores_L1_de_h, ordenes, pendiente_global)."""
    errores = []
    for n in RESOLUCIONES:
        num, ex, medida = correr(n, limiter)
        errores.append(_norma_l1(num[0], ex[0], medida))
    errores = np.array(errores)
    ordenes, pendiente = convergence_order(np.array(RESOLUCIONES), errores)
    return errores, ordenes, pendiente


def _tabla(titulo, errores, ordenes, pendiente):
    print(f"\n{titulo}")
    print(f"{'n':>6} {'L1(h)':>14} {'orden':>8}")
    for i, n in enumerate(RESOLUCIONES):
        o = f"{ordenes[i-1]:8.3f}" if i else " " * 8
        print(f"{n:6d} {errores[i]:14.6e} {o}")
    print(f"orden global (ajuste log-log) = {pendiente:.3f}")


# ---------------------------------------------------------------------------
# 4a. La solucion manufacturada es coherente consigo misma
# ---------------------------------------------------------------------------
def test_la_solucion_manufacturada_es_valida():
    mms = default_mms_2d(LX, LY)
    x = np.linspace(0.0, LX, 37)
    X, Y = np.meshgrid(x, x)
    h, u, v = mms.exact(X, Y, 3.0)
    assert np.all(h > 0.0), "la columna debe permanecer mojada"
    assert np.all(mms.rest_depth(X, Y) > 0.0), "h0 debe ser positiva"
    # Periodicidad exacta en ambos ejes.
    for arr in mms.exact(np.array([0.0]), np.array([0.0]), 1.0):
        pass
    a0 = mms.exact_primitive(np.array([0.0]), np.array([0.0]), 1.0)
    aL = mms.exact_primitive(np.array([LX]), np.array([LY]), 1.0)
    for c0, cL in zip(a0, aL):
        assert c0 == pytest.approx(cL, abs=1e-9)


def test_el_termino_fuente_se_calcula_con_derivadas_analiticas():
    """
    El residuo manufacturado debe coincidir con el residuo evaluado por
    diferencias finitas de MUY alta resolucion. Si no coincidiera, el algebra de
    source() estaria mal y la compuerta mediria un orden falso.
    """
    mms = default_mms_2d(LX, LY)
    x0, y0, t0 = 137.0, 293.0, 4.0
    eps_s, eps_t = 1e-3, 1e-4

    def flujos(xx, yy, tt):
        h, u, v = mms.exact(np.array([xx]), np.array([yy]), tt)
        return h[0], u[0], v[0]

    def q(xx, yy, tt):
        h, u, v = flujos(xx, yy, tt)
        return np.array([h, h * u, h * v])

    def fx(xx, yy, tt):
        h, u, v = flujos(xx, yy, tt)
        return np.array([h * u, h * u * u + 0.5 * mms.g * h * h, h * u * v])

    def fy(xx, yy, tt):
        h, u, v = flujos(xx, yy, tt)
        return np.array([h * v, h * u * v, h * v * v + 0.5 * mms.g * h * h])

    dq_dt = (q(x0, y0, t0 + eps_t) - q(x0, y0, t0 - eps_t)) / (2 * eps_t)
    dfx = (fx(x0 + eps_s, y0, t0) - fx(x0 - eps_s, y0, t0)) / (2 * eps_s)
    dfy = (fy(x0, y0 + eps_s, t0) - fy(x0, y0 - eps_s, t0)) / (2 * eps_s)

    h, _, _ = flujos(x0, y0, t0)
    b_x = (mms.bed_elevation(np.array([x0 + eps_s]), np.array([y0]))[0]
           - mms.bed_elevation(np.array([x0 - eps_s]), np.array([y0]))[0]) / (2 * eps_s)
    b_y = (mms.bed_elevation(np.array([x0]), np.array([y0 + eps_s]))[0]
           - mms.bed_elevation(np.array([x0]), np.array([y0 - eps_s]))[0]) / (2 * eps_s)

    esperado = dq_dt + dfx + dfy + np.array([0.0, mms.g * h * b_x,
                                             mms.g * h * b_y])
    obtenido = np.array([c[0] for c in
                         mms.source(np.array([x0]), np.array([y0]), t0)])
    np.testing.assert_allclose(obtenido, esperado, rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# 4b. LA COMPUERTA: orden 2 sobre solucion suave
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_orden_formal_2_del_esquema_base_1d():
    errores, ordenes, pendiente = _barrido(_correr_1d, "none")
    _tabla("--- COMPUERTA 4 (1D, sin limitar): orden formal ---",
           errores, ordenes, pendiente)
    assert np.all(np.diff(errores) < 0.0)
    assert pendiente >= ORDEN_BASE_MINIMO, f"orden = {pendiente:.3f}"


@pytest.mark.slow
def test_orden_formal_2_del_esquema_base_2d():
    errores, ordenes, pendiente = _barrido(_correr_2d, "none")
    _tabla("--- COMPUERTA 4 (2D, sin limitar): orden formal ---",
           errores, ordenes, pendiente)
    assert np.all(np.diff(errores) < 0.0)
    assert pendiente >= ORDEN_BASE_MINIMO, f"orden = {pendiente:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("limiter", ["mc", "minmod"])
def test_los_limitadores_tvd_convergen_con_orden_alto_2d(limiter):
    """
    Cuantifica el peaje del limitador. No se exige 2: se exige convergencia
    monotona y orden claramente superior a 1, que es lo que distingue a un
    esquema de segundo orden con limitador de uno de primer orden.
    """
    errores, ordenes, pendiente = _barrido(_correr_2d, limiter)
    _tabla(f"--- COMPUERTA 4 (2D, limitador '{limiter}') ---",
           errores, ordenes, pendiente)
    assert np.all(np.diff(errores) < 0.0)
    assert pendiente >= ORDEN_TVD_MINIMO, f"orden = {pendiente:.3f}"


@pytest.mark.slow
def test_el_orden_1_es_efectivamente_de_orden_1():
    """
    Control negativo: con order=1 el esquema debe medir ~1. Si midiera 2, el
    test de convergencia estaria midiendo otra cosa (p.ej. un error dominado por
    el termino fuente) y todos los ordenes anteriores serian sospechosos.
    """
    errores = []
    for n in RESOLUCIONES:
        mms = default_mms_2d(LX, LY)
        dx = LX / n
        x = (np.arange(n) + 0.5) * dx
        X, Y = np.meshgrid(x, x)
        h0 = mms.rest_depth(X, Y)
        dom = Domain(x=x, y=x.copy(), bathymetry=h0, lat0_deg=0.0)
        zeta, u, v = mms.exact_primitive(X, Y, 0.0)
        s = NativeFVMSolver2D(g=mms.g, order=1, extra_source=mms.source_2d())
        s.initialize(dom, State(zeta.copy(), u.copy(), v.copy()),
                     ForcingBundle(coriolis_enabled=False),
                     SolverConfig(cfl=CFL, bc=PERIODICA_2D))
        _avanzar(s)
        h_ex, _, _ = mms.exact(X, Y, s.state.t)
        errores.append(_norma_l1(s.state.zeta + h0, h_ex, dx * dx))
    errores = np.array(errores)
    ordenes, pendiente = convergence_order(np.array(RESOLUCIONES), errores)
    _tabla("--- COMPUERTA 4 (2D, order=1): control negativo ---",
           errores, ordenes, pendiente)
    assert 0.7 < pendiente < 1.4, f"order=1 midio {pendiente:.3f}, se esperaba ~1"
