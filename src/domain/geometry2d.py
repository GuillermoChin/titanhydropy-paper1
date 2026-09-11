"""
domain/geometry2d.py
====================
Constructores de dominios 2D idealizados.

CONVENCION DE MALLA (explicita para evitar el error clasico de medio dx):
    Celdas centradas. Para [0,Lx] x [0,Ly] con nx x ny celdas,
        dx = Lx/nx,  x_i = (i+1/2) dx
        dy = Ly/ny,  y_j = (j+1/2) dy
    Los extremos fisicos son 0 y Lx (resp. Ly), no x_0 y x_{nx-1}.

CONVENCION DE FORMA: todos los campos 2D son (ny, nx). Domain.bathymetry es la
PROFUNDIDAD DE REPOSO h0 > 0 de forma (ny, nx); el motor la convierte a cota de
fondo b = -h0 internamente.
"""

from __future__ import annotations

import numpy as np

from core.solver_base import Domain, State

__all__ = ["cell_centers_2d", "flat_basin", "bumpy_basin",
           "transverse_invariant_channel", "quiescent_state_2d",
           "gaussian_surface_state_2d", "parabolic_basin"]


def cell_centers_2d(lx: float, ly: float, nx: int, ny: int
                    ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Centros de celda (x, y) y pasos (dx, dy)."""
    if nx < 5 or ny < 5:
        raise ValueError("nx y ny deben ser >= 5 (2 celdas fantasma por lado).")
    dx, dy = lx / nx, ly / ny
    return (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, dx, dy


def flat_basin(lx: float, ly: float, nx: int, ny: int, depth: float,
               lat0_deg: float = 78.0) -> Domain:
    """
    Cuenca rectangular de profundidad constante.

    lat0_deg = 78 N por defecto: latitud representativa de los mares polares del
    norte de Titan. Solo interviene si coriolis_enabled=True.
    """
    x, y, _, _ = cell_centers_2d(lx, ly, nx, ny)
    return Domain(x=x, y=y, bathymetry=np.full((ny, nx), float(depth)),
                  lat0_deg=lat0_deg)


def bumpy_basin(lx: float, ly: float, nx: int, ny: int, depth_mean: float,
                bump_height: float, bump_center: tuple[float, float],
                bump_width: float, lat0_deg: float = 78.0) -> Domain:
    """
    Cuenca con un monticulo gaussiano 2D en el fondo:
        h0(x,y) = depth_mean - H * exp(-((x-xc)^2 + (y-yc)^2) / (2 w^2))

    Se exige h0 > 0: la compuerta 1 valida el equilibrio de reposo sobre
    batimetria variable, no el secado.
    """
    x, y, _, _ = cell_centers_2d(lx, ly, nx, ny)
    X, Y = np.meshgrid(x, y)
    xc, yc = bump_center
    r2 = (X - xc) ** 2 + (Y - yc) ** 2
    h0 = depth_mean - bump_height * np.exp(-0.5 * r2 / bump_width ** 2)
    if np.any(h0 <= 0.0):
        raise ValueError("El monticulo emerge: h0 <= 0. Reduce bump_height.")
    return Domain(x=x, y=y, bathymetry=h0, lat0_deg=lat0_deg)


def transverse_invariant_channel(lx: float, ly: float, nx: int, ny: int,
                                 depth_profile: np.ndarray,
                                 lat0_deg: float = 78.0) -> Domain:
    """
    Canal 2D cuya batimetria NO depende de y: h0(x,y) = depth_profile(x).

    Es el dominio de la compuerta 3 (consistencia 2D<->1D): con forzamiento
    tambien invariante en y y fronteras norte/sur reflectivas, el problema 2D es
    matematicamente identico al 1D y debe reproducirlo.
    """
    x, y, _, _ = cell_centers_2d(lx, ly, nx, ny)
    perfil = np.asarray(depth_profile, dtype=float).ravel()
    if perfil.size != nx:
        raise ValueError(f"depth_profile debe tener nx={nx} valores.")
    return Domain(x=x, y=y, bathymetry=np.tile(perfil, (ny, 1)),
                  lat0_deg=lat0_deg)


def parabolic_basin(lx: float, ly: float, nx: int, ny: int, depth_center: float,
                    radius: float, center: tuple[float, float] | None = None,
                    lat0_deg: float = 78.0) -> Domain:
    """
    Cuenca paraboloide de revolucion:  h0(r) = D * (1 - r^2/L^2).

    Es la geometria del test analitico de Thacker (oscilacion en cuenca
    parabolica), que se usa para validar wetting-drying. Fuera del
    radio la "profundidad" se vuelve negativa: el fondo emerge. Se devuelve tal
    cual, y es responsabilidad del motor tratar la zona seca; los constructores
    de estado inicial deben respetar h = max(0, zeta - b).
    """
    x, y, _, _ = cell_centers_2d(lx, ly, nx, ny)
    X, Y = np.meshgrid(x, y)
    xc, yc = center if center is not None else (0.5 * lx, 0.5 * ly)
    r2 = (X - xc) ** 2 + (Y - yc) ** 2
    h0 = depth_center * (1.0 - r2 / radius ** 2)
    return Domain(x=x, y=y, bathymetry=h0, lat0_deg=lat0_deg)


def quiescent_state_2d(domain: Domain, zeta0: float = 0.0) -> State:
    """Estado de reposo: superficie plana en zeta0 y velocidad nula."""
    shape = (domain.y.size, domain.x.size)
    return State(zeta=np.full(shape, float(zeta0)),
                 u=np.zeros(shape), v=np.zeros(shape), t=0.0)


def gaussian_surface_state_2d(domain: Domain, amplitude: float, sigma: float,
                              center: tuple[float, float] | None = None
                              ) -> State:
    """Perturbacion gaussiana 2D de la superficie libre, en reposo."""
    X, Y = np.meshgrid(domain.x, domain.y)
    xc, yc = center if center is not None else (domain.x.mean(), domain.y.mean())
    r2 = (X - xc) ** 2 + (Y - yc) ** 2
    zeta = amplitude * np.exp(-0.5 * r2 / sigma ** 2)
    return State(zeta=zeta, u=np.zeros_like(zeta), v=np.zeros_like(zeta), t=0.0)
