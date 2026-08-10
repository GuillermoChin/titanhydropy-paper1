"""
domain/geometry.py
==================
Constructores de dominios 1D idealizados.

CONVENCION DE MALLA (explicita para evitar el error clasico de medio dx):
    Celdas centradas. Para un canal [0, L] con nx celdas,
        dx = L / nx,   x_i = (i + 1/2) dx,  i = 0..nx-1
    Los extremos fisicos del dominio son x = 0 y x = L, no x_0 y x_{nx-1}.

CONVENCION DE BATIMETRIA: Domain.bathymetry es la PROFUNDIDAD DE REPOSO h0 > 0.
El motor la convierte internamente a cota de fondo b = -h0.
"""

from __future__ import annotations

import numpy as np

from core.solver_base import Domain, State

__all__ = ["cell_centers", "flat_channel", "variable_bed_channel",
           "quiescent_state", "step_state"]


def cell_centers(length: float, nx: int) -> tuple[np.ndarray, float]:
    """Centros de celda y dx para el intervalo [0, length] con nx celdas."""
    if nx < 5:
        raise ValueError("nx debe ser >= 5 (2 celdas fantasma por lado).")
    dx = length / nx
    return (np.arange(nx) + 0.5) * dx, dx


def flat_channel(length: float, nx: int, depth: float,
                 lat0_deg: float = 78.0) -> Domain:
    """
    Canal rectangular de profundidad constante.

    lat0_deg = 78 N por defecto: latitud representativa de los mares polares
    del norte de Titan. Solo interviene si coriolis_enabled=True.
    """
    x, _ = cell_centers(length, nx)
    return Domain(x=x, y=np.array([0.0]),
                  bathymetry=np.full(nx, float(depth)),
                  lat0_deg=lat0_deg)


def variable_bed_channel(length: float, nx: int, depth_mean: float,
                         bump_height: float, bump_center: float,
                         bump_width: float, lat0_deg: float = 78.0) -> Domain:
    """
    Canal con un monticulo gaussiano en el fondo:
        h0(x) = depth_mean - bump_height * exp(-(x-xc)^2 / (2 w^2))

    Se exige h0 > 0 en todo el dominio: la compuerta 1 valida el equilibrio de
    reposo sobre batimetria variable, no el secado.
    """
    x, _ = cell_centers(length, nx)
    arg = (x - bump_center) / bump_width
    h0 = depth_mean - bump_height * np.exp(-0.5 * arg * arg)
    if np.any(h0 <= 0.0):
        raise ValueError("El monticulo emerge: h0 <= 0. Reduce bump_height.")
    return Domain(x=x, y=np.array([0.0]), bathymetry=h0, lat0_deg=lat0_deg)


def quiescent_state(domain: Domain, zeta0: float = 0.0) -> State:
    """Estado de reposo: superficie plana en zeta = zeta0 y velocidad nula."""
    n = domain.x.size
    return State(zeta=np.full(n, float(zeta0)),
                 u=np.zeros(n), v=np.zeros(n), t=0.0)


def step_state(domain: Domain, h_left: float, h_right: float,
               x_discontinuity: float) -> State:
    """
    Estado inicial de rotura de presa: escalon en la ALTURA TOTAL de columna.
    zeta = h - h0. Solo tiene sentido fisico sobre fondo plano.
    """
    h = np.where(domain.x < x_discontinuity, float(h_left), float(h_right))
    n = domain.x.size
    return State(zeta=h - domain.bathymetry,
                 u=np.zeros(n), v=np.zeros(n), t=0.0)
