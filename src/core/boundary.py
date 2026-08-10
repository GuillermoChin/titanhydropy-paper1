"""
core/boundary.py
================
Condiciones de frontera por celdas fantasma para el motor FVM 1D.

Tipos soportados (declarados en SolverConfig.bc como 'west'/'east'):
  * 'reflective'   : pared solida. h y hv se reflejan con signo +, hu con signo -.
                     Preserva EXACTAMENTE el equilibrio de reposo.
  * 'transmissive' : frontera abierta de orden cero (extrapolacion constante).
                     Radia el grueso de la energia saliente; deja una reflexion
                     residual de segundo orden que se acota alejando la frontera.
  * 'periodic'     : envolvente.

El numero de celdas fantasma es NG = 2, impuesto por la reconstruccion MUSCL.
"""

from __future__ import annotations
import numpy as np

NG = 2
_VALID = ("reflective", "transmissive", "periodic")

__all__ = ["NG", "apply_bc", "extend_bathymetry", "validate_bc",
           "apply_bc_2d", "extend_bathymetry_2d", "validate_bc_2d"]


def validate_bc(bc: dict) -> tuple[str, str]:
    """Extrae y valida los tipos de frontera oeste/este del diccionario bc."""
    west = bc.get("west", "reflective")
    east = bc.get("east", "reflective")
    for name, kind in (("west", west), ("east", east)):
        if kind not in _VALID:
            raise ValueError(f"Frontera '{name}' desconocida: {kind!r}. "
                             f"Validas: {_VALID}")
    if (west == "periodic") != (east == "periodic"):
        raise ValueError("La frontera periodica debe declararse en ambos extremos.")
    return west, east


def _fill(arr: np.ndarray, west: str, east: str, sign: float) -> None:
    """
    Rellena in-place las NG celdas fantasma de cada extremo DEL ULTIMO EJE.

    Generico en la dimension: sirve para arrays 1D (nx+2*NG,) y para arrays 2D
    (ny+2*NG, nx+2*NG). El motor 2D cubre el eje y transponiendo.
    """
    n = arr.shape[-1]
    if west == "periodic":
        arr[..., :NG] = arr[..., n - 2 * NG:n - NG]
        arr[..., n - NG:] = arr[..., NG:2 * NG]
        return
    # Oeste (extremo bajo del eje)
    if west == "reflective":
        arr[..., 0] = sign * arr[..., 2 * NG - 1]   # espejo respecto a la cara
        arr[..., 1] = sign * arr[..., 2 * NG - 2]
    else:  # transmissive
        arr[..., 0] = arr[..., NG]
        arr[..., 1] = arr[..., NG]
    # Este (extremo alto del eje)
    if east == "reflective":
        arr[..., n - 1] = sign * arr[..., n - 2 * NG]
        arr[..., n - 2] = sign * arr[..., n - 2 * NG + 1]
    else:  # transmissive
        arr[..., n - 1] = arr[..., n - NG - 1]
        arr[..., n - 2] = arr[..., n - NG - 1]


def apply_bc(h_ext: np.ndarray, hu_ext: np.ndarray, hv_ext: np.ndarray,
             west: str, east: str) -> None:
    """Aplica las condiciones de frontera in-place sobre los arrays extendidos."""
    _fill(h_ext, west, east, +1.0)
    _fill(hu_ext, west, east, -1.0)   # velocidad normal: antisimetrica en pared
    _fill(hv_ext, west, east, +1.0)   # velocidad tangencial: free-slip


def extend_bathymetry(b: np.ndarray, west: str, east: str) -> np.ndarray:
    """
    Construye el array extendido de cota de fondo b 1D (fijo en el tiempo).
    Se replica la misma regla que para h para no romper el equilibrio de reposo.
    """
    b_ext = np.empty(b.size + 2 * NG, dtype=float)
    b_ext[NG:-NG] = b
    _fill(b_ext, west, east, +1.0)
    return b_ext


# ---------------------------------------------------------------------------
# Version 2D: se aplica primero al eje x (ultimo) y luego al eje y (primero,
# via transposicion). Las esquinas quedan rellenadas por la segunda pasada.
# ---------------------------------------------------------------------------
def validate_bc_2d(bc: dict) -> tuple[str, str, str, str]:
    """Valida las cuatro fronteras: (oeste, este, sur, norte)."""
    west, east = validate_bc(bc)
    south = bc.get("south", "reflective")
    north = bc.get("north", "reflective")
    for name, kind in (("south", south), ("north", north)):
        if kind not in _VALID:
            raise ValueError(f"Frontera '{name}' desconocida: {kind!r}. "
                             f"Validas: {_VALID}")
    if (south == "periodic") != (north == "periodic"):
        raise ValueError("La frontera periodica debe declararse en ambos "
                         "extremos del eje y.")
    return west, east, south, north


def apply_bc_2d(h: np.ndarray, hu: np.ndarray, hv: np.ndarray,
                west: str, east: str, south: str, north: str) -> None:
    """
    Aplica in-place las cuatro fronteras sobre arrays extendidos (ny+2NG, nx+2NG).

    En cada eje, el momento NORMAL a la pared se refleja con signo negativo y el
    TANGENCIAL con signo positivo (free-slip):
      * eje x: hu es normal, hv tangencial.
      * eje y: hv es normal, hu tangencial.
    """
    # Eje x (ultimo eje)
    _fill(h, west, east, +1.0)
    _fill(hu, west, east, -1.0)
    _fill(hv, west, east, +1.0)
    # Eje y (primer eje, via vistas transpuestas: comparten memoria)
    _fill(h.T, south, north, +1.0)
    _fill(hv.T, south, north, -1.0)
    _fill(hu.T, south, north, +1.0)


def extend_bathymetry_2d(b: np.ndarray, west: str, east: str,
                         south: str, north: str) -> np.ndarray:
    """Cota de fondo 2D extendida con celdas fantasma (fija en el tiempo)."""
    ny, nx = b.shape
    b_ext = np.empty((ny + 2 * NG, nx + 2 * NG), dtype=float)
    b_ext[NG:-NG, NG:-NG] = b
    _fill(b_ext, west, east, +1.0)
    _fill(b_ext.T, south, north, +1.0)
    return b_ext
