"""
validation/analytic.py
======================
Soluciones analiticas de referencia para las compuertas de validacion.

1. `dam_break_exact`: solucion EXACTA del problema de Riemann para las SWE 1D
   sobre fondo plano y sin friccion (ambos lados mojados). Estructura clasica:
   rarefaccion izquierda + discontinuidad de contacto + choque derecho (o la
   configuracion simetrica). Se resuelve h* por Newton sobre la funcion de
   presion, siguiendo la formulacion estandar del problema de Riemann para
   aguas someras (analoga a la de Toro para gas dinamico).

   El unico dato fisico usado es g, que se pasa explicitamente; ningun valor se
   inventa aqui.
"""

from __future__ import annotations

import numpy as np

__all__ = ["star_depth", "dam_break_exact"]

_MAX_ITER = 100
_TOL = 1e-14


def _f_and_df(h: float, hk: float, g: float) -> tuple[float, float]:
    """
    Funcion de onda f_K(h) y su derivada.

    Rarefaccion (h <= hk):  f = 2 (sqrt(g h) - sqrt(g hk)),  f' = sqrt(g/h)
    Choque      (h  > hk):  f = (h - hk) sqrt( g (h + hk) / (2 h hk) )
    """
    ck = np.sqrt(g * hk)
    if h <= hk:                       # rarefaccion
        return 2.0 * (np.sqrt(g * h) - ck), np.sqrt(g / h)
    gel = np.sqrt(0.5 * g * (h + hk) / (h * hk))
    f = (h - hk) * gel
    # d/dh de (h-hk)*sqrt(g(h+hk)/(2 h hk))
    dgel = 0.5 * g * (-(h + hk) / (h * h * hk) + 1.0 / (h * hk)) / (2.0 * gel)
    df = gel + (h - hk) * dgel
    return f, df


def star_depth(hL: float, uL: float, hR: float, uR: float, g: float) -> float:
    """
    Profundidad de la region estrella h*, raiz de
        f_L(h) + f_R(h) + (uR - uL) = 0.
    Newton con arranque en la aproximacion de dos rarefacciones.
    """
    if hL <= 0.0 or hR <= 0.0:
        raise ValueError("dam_break_exact solo cubre el caso mojado-mojado.")
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    du = uR - uL
    h = max((0.5 * (cL + cR) - 0.25 * du) ** 2 / g, 1e-12)
    for _ in range(_MAX_ITER):
        fL, dfL = _f_and_df(h, hL, g)
        fR, dfR = _f_and_df(h, hR, g)
        f = fL + fR + du
        df = dfL + dfR
        h_new = h - f / df
        if h_new <= 0.0:
            h_new = 0.5 * h
        if abs(h_new - h) <= _TOL * max(h_new, 1.0):
            return h_new
        h = h_new
    raise RuntimeError("star_depth no converge; revisa los estados iniciales.")


def dam_break_exact(x: np.ndarray, t: float, hL: float, hR: float, g: float,
                    x0: float = 0.0, uL: float = 0.0, uR: float = 0.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solucion exacta (h, u) del problema de Riemann en el instante t.

    Autosemejante en xi = (x - x0)/t. Para t = 0 devuelve el dato inicial.
    """
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        h = np.where(x < x0, hL, hR)
        u = np.where(x < x0, uL, uR)
        return h.astype(float), u.astype(float)

    hs = star_depth(hL, uL, hR, uR, g)
    cs = np.sqrt(g * hs)
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    fL, _ = _f_and_df(hs, hL, g)
    fR, _ = _f_and_df(hs, hR, g)
    us = 0.5 * (uL + uR) + 0.5 * (fR - fL)

    xi = (x - x0) / t
    h = np.empty_like(x)
    u = np.empty_like(x)

    # ---- lado izquierdo -------------------------------------------------
    if hs > hL:                                   # choque izquierdo
        qL = np.sqrt(0.5 * (hs + hL) * hs / (hL * hL))
        sL = uL - cL * qL
        left_outside = xi <= sL
        left_star = (~left_outside)
        fan_L = np.zeros_like(x, dtype=bool)
    else:                                         # rarefaccion izquierda
        s_head = uL - cL
        s_tail = us - cs
        left_outside = xi <= s_head
        fan_L = (xi > s_head) & (xi < s_tail)
        left_star = xi >= s_tail

    # ---- lado derecho ---------------------------------------------------
    if hs > hR:                                   # choque derecho
        qR = np.sqrt(0.5 * (hs + hR) * hs / (hR * hR))
        sR = uR + cR * qR
        right_outside = xi >= sR
        right_star = ~right_outside
        fan_R = np.zeros_like(x, dtype=bool)
    else:                                         # rarefaccion derecha
        s_head = uR + cR
        s_tail = us + cs
        right_outside = xi >= s_head
        fan_R = (xi < s_head) & (xi > s_tail)
        right_star = xi <= s_tail

    star = left_star & right_star & (~fan_L) & (~fan_R)

    h[:] = np.nan
    u[:] = np.nan
    h = np.where(left_outside, hL, h)
    u = np.where(left_outside, uL, u)
    h = np.where(right_outside, hR, h)
    u = np.where(right_outside, uR, u)
    h = np.where(star, hs, h)
    u = np.where(star, us, u)

    if np.any(fan_L):
        c_fan = (uL + 2.0 * cL - xi) / 3.0
        h = np.where(fan_L, c_fan * c_fan / g, h)
        u = np.where(fan_L, (uL + 2.0 * cL + 2.0 * xi) / 3.0, u)
    if np.any(fan_R):
        c_fan = (-uR + 2.0 * cR + xi) / 3.0
        h = np.where(fan_R, c_fan * c_fan / g, h)
        u = np.where(fan_R, (uR - 2.0 * cR + 2.0 * xi) / 3.0, u)

    if np.any(~np.isfinite(h)):
        raise RuntimeError("Region no cubierta en dam_break_exact (bug logico).")
    return h, u
