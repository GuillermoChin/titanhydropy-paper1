"""
forcing/wind.py
===============
Campos de viento a 10 m que satisfacen el protocolo `WindField`.

Sprint 1 solo necesita el caso nulo y uno uniforme; el acoplamiento
viento-presion (experimentos A/B/C) llega en sprints posteriores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoWind", "UniformWind"]


@dataclass
class NoWind:
    """Viento nulo."""

    def evaluate(self, x: np.ndarray, y: np.ndarray,
                 t: float) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros(np.shape(x), dtype=float)
        return z, z.copy()


@dataclass
class UniformWind:
    """Viento uniforme y estacionario (u10, v10) [m/s]."""
    u10: float
    v10: float = 0.0

    def evaluate(self, x: np.ndarray, y: np.ndarray,
                 t: float) -> tuple[np.ndarray, np.ndarray]:
        shape = np.shape(x)
        return (np.full(shape, self.u10, dtype=float),
                np.full(shape, self.v10, dtype=float))
