"""
forcing/friction.py
===================
Leyes de friccion de fondo que satisfacen el protocolo `FrictionModel`.

Convencion del contrato: stress() devuelve el ARRASTRE POR UNIDAD DE MASA
(aceleracion, m/s^2), no un esfuerzo en Pa. El motor lo multiplica por h para
obtener el termino fuente del momento.

ADVERTENCIA FISICA: ningun coeficiente de friccion de fondo esta medido para
los mares de Titan. Estos modelos son parametros libres del estudio de
sensibilidad, NO constantes fisicas. Por eso no viven en titan_params.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.solver_base import State

__all__ = ["NoFriction", "LinearFriction", "ManningFriction"]


@dataclass
class NoFriction:
    """Friccion nula. Es el modelo de las compuertas 1-3 y 5 (canal ideal)."""

    def stress(self, state: State, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros_like(np.asarray(state.u, dtype=float))
        return z, z.copy()


@dataclass
class LinearFriction:
    """
    Arrastre lineal (Rayleigh):  a = -r * U.

    r [1/s] es un coeficiente de relajacion. Se usa para amortiguar transitorios
    en el test de barometro inverso; r debe ser << la frecuencia de las ondas
    resueltas para no contaminar la respuesta de equilibrio.
    """
    r: float

    def stress(self, state: State, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return -self.r * np.asarray(state.u, dtype=float), \
               -self.r * np.asarray(state.v, dtype=float)


@dataclass
class ManningFriction:
    """
    Arrastre cuadratico tipo Manning:  a = -g n^2 |U| U / h^(4/3).

    n [s/m^(1/3)] es el coeficiente de Manning. PARAMETRO LIBRE: no existe valor
    medido para lechos de hidrocarburos en Titan.
    """
    n: float
    g: float
    min_depth: float = 1e-3

    def stress(self, state: State, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(state.u, dtype=float)
        v = np.asarray(state.v, dtype=float)
        h = np.asarray(depth, dtype=float)
        wet = h > self.min_depth
        h_safe = np.where(wet, h, 1.0)
        speed = np.sqrt(u * u + v * v)
        k = np.where(wet, self.g * self.n ** 2 * speed / h_safe ** (4.0 / 3.0), 0.0)
        return -k * u, -k * v
