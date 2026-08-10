"""
core/fvm_kernel.py
==================
Kernel direccional de volumenes finitos: calcula la divergencia de flujo mas los
terminos fuente well-balanced A LO LARGO DEL ULTIMO EJE de los arrays.

Este modulo es el unico sitio donde vive la discretizacion espacial. El motor 1D
lo llama una vez (direccion x); el motor 2D lo llama dos veces (direccion x, y
direccion y sobre los arrays transpuestos con los momentos intercambiados). Asi
la propiedad well-balanced demostrada bit a bit en 1D se hereda EXACTAMENTE en
cada direccion de 2D, sin reimplementar nada.

CONVENCION DE MOMENTOS
----------------------
  mn = momento NORMAL a las caras que se estan cruzando (el que aparece en el
       flujo de presion hidrostatica).
  mt = momento TANGENCIAL (transportado pasivamente por la onda de contacto).
Para la direccion x: (mn, mt) = (hu, hv).
Para la direccion y: (mn, mt) = (hv, hu)  sobre arrays transpuestos.

ESQUEMA (supuestos numericos explicitos)
----------------------------------------
* Volumenes finitos, celdas centradas, malla uniforme.
* Reconstruccion MUSCL sobre (eta, un, ut, b), NUNCA sobre h (ADR-005).
* Reconstruccion hidrostatica de Audusse et al. (2004) en cada interfaz.
* Solver de Riemann HLLC.
* Termino fuente en FORMA FACTORIZADA g*h_barra*d(eta) (ADR-004): unica forma
  que cancela bit a bit en el reposo.

GANCHO DE ACELERACION: puro NumPy sin ramas de Python. Candidato directo a
@numba.njit o jax.jit sin cambio de firma. No se optimiza prematuramente.
"""

from __future__ import annotations

import numpy as np

from core.reconstruction import hydrostatic_reconstruction, muscl_edges
from core.riemann import hllc_flux, hydrostatic_pressure
from physics.wetting_drying import (WettingDryingConfig,
                                    desingularized_velocity_kp, front_mask)

__all__ = ["NG", "directional_rhs", "desingularized_velocity"]

NG = 2  # celdas fantasma por lado, impuesto por la reconstruccion MUSCL


def desingularized_velocity(h: np.ndarray, m: np.ndarray,
                            min_depth: float) -> np.ndarray:
    """Velocidad u = m/h, exactamente 0 por debajo del umbral de secado."""
    wet = h > min_depth
    return np.where(wet, m / np.where(wet, h, 1.0), 0.0)


def directional_rhs(h_ext: np.ndarray, mn_ext: np.ndarray, mt_ext: np.ndarray,
                    b_ext: np.ndarray, d: float, g: float, order: int,
                    min_depth: float, limiter: str = "minmod",
                    wd: "WettingDryingConfig | None" = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contribucion de UNA direccion al lado derecho de las ecuaciones.

    Parametros
    ----------
    h_ext, mn_ext, mt_ext, b_ext : arrays (..., N) con N = n + 2*NG, es decir con
        las celdas fantasma YA rellenadas a lo largo del ultimo eje.
    d  : paso de malla en esa direccion [m].
    wd : configuracion de frontera movil. None = desactivada (el caso de los
         dominios siempre mojados de los Papers 1). Cuando esta activa:
           * la velocidad se desingulariza (Kurganov-Petrova) en vez de usar un
             umbral duro, y
           * la reconstruccion se degrada a orden 1 en las celdas del frente,
             que es lo que garantiza positividad.

    Devuelve
    --------
    (rhs_h, rhs_mn, rhs_mt) de forma (..., n): solo las celdas interiores del
    ultimo eje. Los demas ejes se devuelven intactos (el motor 2D los recorta).
    """
    n = h_ext.shape[-1] - 2 * NG

    if wd is not None and wd.desingularize:
        un = desingularized_velocity_kp(h_ext, mn_ext, min_depth)
        ut = desingularized_velocity_kp(h_ext, mt_ext, min_depth)
    else:
        un = desingularized_velocity(h_ext, mn_ext, min_depth)
        ut = desingularized_velocity(h_ext, mt_ext, min_depth)
    eta = h_ext + b_ext

    # --- Reconstruccion MUSCL sobre (eta, un, ut, b) ------------------------
    etaL, etaR = muscl_edges(eta, order, limiter)
    unL, unR = muscl_edges(un, order, limiter)
    utL, utR = muscl_edges(ut, order, limiter)
    bL, bR = muscl_edges(b_ext, order, limiter)

    # --- Reduccion de orden en el frente mojado/seco ------------------------
    # Sin esto, la extrapolacion lineal produce h < 0 en celdas parcialmente
    # mojadas. Con pendiente nula la positividad es incondicional. No afecta al
    # equilibrio de reposo: con eta constante la pendiente ya era cero.
    if wd is not None and wd.reduce_order:
        fr = front_mask(h_ext, min_depth, wd.front_stencil)
        if np.any(fr):
            etaL = np.where(fr, eta, etaL)
            etaR = np.where(fr, eta, etaR)
            unL = np.where(fr, un, unL)
            unR = np.where(fr, un, unR)
            utL = np.where(fr, ut, utL)
            utR = np.where(fr, ut, utR)
            bL = np.where(fr, b_ext, bL)
            bR = np.where(fr, b_ext, bR)
    hLc = np.maximum(0.0, etaL - bL)   # altura en el borde izquierdo de celda
    hRc = np.maximum(0.0, etaR - bR)   # altura en el borde derecho de celda

    # --- Estados a ambos lados de cada interfaz -----------------------------
    # Interfaz k (k = 0..n) separa la celda j = NG-1+k de la j+1.
    j0, j1 = NG - 1, NG + n
    Lh, Lb = hRc[..., j0:j1], bR[..., j0:j1]
    Lun, Lut = unR[..., j0:j1], utR[..., j0:j1]
    Rh, Rb = hLc[..., j0 + 1:j1 + 1], bL[..., j0 + 1:j1 + 1]
    Run, Rut = unL[..., j0 + 1:j1 + 1], utL[..., j0 + 1:j1 + 1]

    # --- Reconstruccion hidrostatica + flujo HLLC ---------------------------
    hsL, hsR = hydrostatic_reconstruction(Lh, Lb, Rh, Rb)
    Fh, Fmn, Fmt, _ = hllc_flux(hsL, Lun, Lut, hsR, Run, Rut, g)

    # --- Terminos fuente well-balanced (Audusse, forma factorizada) ---------
    # (a) FLUCTUACION del flujo, F_num - g h*^2/2: exactamente cero en reposo
    #     porque HLLC devuelve justo el termino hidrostatico.
    # (b) gradiente hidrostatico interno FACTORIZADO g*h_barra*d(eta), unica
    #     forma que cancela bit a bit: con eta = 0 se tiene h = -b exactamente y
    #     (h_R - h_L) + (b_R - b_L) da cero exacto (ADR-004).
    m0, m1 = NG, NG + n
    h_left = hLc[..., m0:m1]
    h_right = hRc[..., m0:m1]

    fluct_right = Fmn[..., 1:] - hydrostatic_pressure(hsL[..., 1:], g)
    fluct_left = Fmn[..., :-1] - hydrostatic_pressure(hsR[..., :-1], g)

    h_bar = 0.5 * (h_left + h_right)
    d_eta = (h_right - h_left) + (bR[..., m0:m1] - bL[..., m0:m1])

    rhs_h = -(Fh[..., 1:] - Fh[..., :-1]) / d
    rhs_mn = -(fluct_right - fluct_left) / d - g * h_bar * d_eta / d
    rhs_mt = -(Fmt[..., 1:] - Fmt[..., :-1]) / d
    return rhs_h, rhs_mn, rhs_mt
