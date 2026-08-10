"""
forcing/atmospheric.py
======================
Campos de presion atmosferica superficial que satisfacen el protocolo
`PressureField` de core.solver_base.

Convencion de signo (critica, se verifica en tests):
    amplitude > 0  -> ALTA presion  -> depresion de la superficie libre
    amplitude < 0  -> BAJA presion  -> elevacion de la superficie libre
La respuesta estatica es zeta = -dP/(rho g) = inverse_barometer(dP).

La presion de fondo por defecto es TITAN.p_surface (TO_VERIFY). Solo el
GRADIENTE de P entra en las ecuaciones, asi que el valor de fondo no altera la
dinamica; se mantiene por coherencia dimensional y para la salida a disco.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants.titan_params import TITAN

__all__ = ["UniformPressure", "MovingGaussianPressure", "StaticGaussianPressure",
           "MovingGaussianPressure2D", "MovingPressureFront2D"]


def _ramp(t: float, t_ramp: float) -> float:
    """
    Rampa temporal suave C^1 en [0, t_ramp] (medio coseno). t_ramp <= 0 -> 1.

    Sirve para encender el forzamiento de forma cuasi-estatica y separar la
    respuesta de equilibrio de los transitorios de arranque.
    """
    if t_ramp <= 0.0:
        return 1.0
    if t >= t_ramp:
        return 1.0
    if t <= 0.0:
        return 0.0
    return 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))


@dataclass
class UniformPressure:
    """Presion uniforme constante. Gradiente nulo: no forza. Util como control."""
    p0: float = TITAN.p_surface

    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        return np.full(np.shape(x), self.p0, dtype=float)


@dataclass
class MovingGaussianPressure:
    """
    Perturbacion de presion gaussiana que se traslada a velocidad constante.

        P(x,y,t) = p0 + A * r(t) * exp( -(x - x0 - U t)^2 / (2 sigma^2) )

    Parametros
    ----------
    amplitude : A [Pa]. Positivo = alta presion.
    sigma     : semiancho gaussiano [m].
    speed     : U, velocidad de traslacion [m/s]. El numero de Froude
                atmosferico-marino es F = U / sqrt(g h) (physics.proudman).
    x0        : posicion del centro en t = 0 [m].
    t_ramp    : duracion de la rampa de encendido [s]. 0 = escalon.
    p0        : presion de fondo [Pa].
    """
    amplitude: float
    sigma: float
    speed: float
    x0: float = 0.0
    t_ramp: float = 0.0
    p0: float = TITAN.p_surface

    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xc = self.x0 + self.speed * t
        arg = (x - xc) / self.sigma
        return self.p0 + self.amplitude * _ramp(t, self.t_ramp) * \
            np.exp(-0.5 * arg * arg)

    def center(self, t: float) -> float:
        """Posicion del centro de la perturbacion en el instante t [m]."""
        return self.x0 + self.speed * t


@dataclass
class MovingGaussianPressure2D:
    """
    Perturbacion de presion gaussiana ISOTROPA 2D que se traslada en linea recta
    con velocidad y direccion configurables.

        P(x,y,t) = p0 + A r(t) exp( -[(x-xc)^2 + (y-yc)^2] / (2 sigma^2) )
        xc = x0 + U cos(theta) t
        yc = y0 + U sin(theta) t

    Parametros
    ----------
    amplitude   : A [Pa]. Positivo = alta presion.
    sigma       : semiancho gaussiano [m].
    speed       : U, modulo de la velocidad de traslacion [m/s].
    heading_deg : theta, direccion de propagacion en grados medida desde el eje
                  +x en sentido antihorario. 0 = hacia el este.
    x0, y0      : centro en t = 0 [m].
    t_ramp      : duracion de la rampa de encendido [s]. 0 = escalon.
    p0          : presion de fondo [Pa].
    """
    amplitude: float
    sigma: float
    speed: float
    heading_deg: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    t_ramp: float = 0.0
    p0: float = TITAN.p_surface

    @property
    def velocity(self) -> tuple[float, float]:
        """Componentes (Ux, Uy) de la velocidad de traslacion [m/s]."""
        th = np.radians(self.heading_deg)
        return self.speed * np.cos(th), self.speed * np.sin(th)

    def center(self, t: float) -> tuple[float, float]:
        """Posicion del centro de la perturbacion en el instante t [m]."""
        ux, uy = self.velocity
        return self.x0 + ux * t, self.y0 + uy * t

    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xc, yc = self.center(t)
        r2 = (x - xc) ** 2 + (y - yc) ** 2
        return self.p0 + self.amplitude * _ramp(t, self.t_ramp) * \
            np.exp(-0.5 * r2 / self.sigma ** 2)


@dataclass
class MovingPressureFront2D:
    """
    FRENTE de presion 2D: cresta gaussiana infinita en la direccion transversal
    que avanza perpendicular a si misma.

        s = (x - x0) cos(theta) + (y - y0) sin(theta) - U t
        P = p0 + A r(t) exp( -s^2 / (2 sigma^2) )

    Es la idealizacion adecuada de un frente convectivo (una linea de racha, no
    una burbuja) y es el forzamiento de la compuerta 3: con theta = 0 el campo es
    INVARIANTE EN y, de modo que el problema 2D se reduce exactamente al 1D.

    Parametros: como MovingGaussianPressure2D; (x0, y0) es un punto por el que
    pasa el frente en t = 0.
    """
    amplitude: float
    sigma: float
    speed: float
    heading_deg: float = 0.0
    x0: float = 0.0
    y0: float = 0.0
    t_ramp: float = 0.0
    p0: float = TITAN.p_surface

    @property
    def normal(self) -> tuple[float, float]:
        """Vector unitario normal al frente (direccion de avance)."""
        th = np.radians(self.heading_deg)
        return float(np.cos(th)), float(np.sin(th))

    def signed_distance(self, x: np.ndarray, y: np.ndarray,
                        t: float) -> np.ndarray:
        """Distancia con signo de cada punto al plano del frente [m]."""
        nx_, ny_ = self.normal
        return (np.asarray(x, dtype=float) - self.x0) * nx_ + \
               (np.asarray(y, dtype=float) - self.y0) * ny_ - self.speed * t

    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        s = self.signed_distance(x, y, t)
        return self.p0 + self.amplitude * _ramp(t, self.t_ramp) * \
            np.exp(-0.5 * (s / self.sigma) ** 2)


@dataclass
class StaticGaussianPressure:
    """
    Perturbacion gaussiana fija en el espacio, encendida con rampa temporal.
    Es el caso limite U = 0 y el forzamiento de la compuerta 4 (barometro
    inverso estatico).
    """
    amplitude: float
    sigma: float
    x0: float = 0.0
    t_ramp: float = 0.0
    p0: float = TITAN.p_surface

    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        arg = (x - self.x0) / self.sigma
        return self.p0 + self.amplitude * _ramp(t, self.t_ramp) * \
            np.exp(-0.5 * arg * arg)
