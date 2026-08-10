"""
validation/thacker.py
=====================
Solucion analitica de Thacker: oscilacion de superficie PLANA en una cuenca
parabolica, con frontera movil (la orilla se desplaza). Es el banco de pruebas
canonico del wetting-and-drying, porque tiene solucion cerrada EXACTA incluyendo
el frente de inundacion.

DERIVACION (no se copia de memoria: se deriva y se verifica numericamente)
--------------------------------------------------------------------------
Cota de fondo parabolica, con vertice a profundidad h0 y orilla de reposo en
r = a:

    b(x,y) = h0 * ( (x^2 + y^2)/a^2 - 1 )

Se busca una solucion con velocidad UNIFORME en el espacio y superficie libre
PLANA:

    u = u(t),   v = 0,   eta(x,t) = A(t) + B(t) x

Sustituyendo en las NLSWE (h = eta - b):

  Momento x:  u' + g B = 0
  Momento y:  0 = 0                       (eta no depende de y, v = 0)
  Continuidad: A' + B' x + u B - 2 u h0 x / a^2 = 0  para todo x
               =>  A' + u B = 0   y   B' = 2 h0 u / a^2

De u' = -gB y B' = 2 h0 u/a^2 sale u'' = -(2 g h0/a^2) u, luego

    omega = sqrt(2 g h0) / a
    u(t)  = U cos(omega t)
    B(t)  = (U omega / g) sin(omega t)
    A(t)  = (U^2 / (4 g)) cos(2 omega t) + A0

La solucion es EXACTA en la region mojada; la orilla es el lugar eta = b y se
desplaza con el tiempo. Es la solucion "planar" de Thacker (1981).

VERIFICACION: `residual()` evalua el residuo de las NLSWE por diferencias
finitas de alta precision. `tests/test_s3_gate2_thacker.py` comprueba que es
nulo, de modo que un error de algebra en este modulo se delata solo y el test de
wetting-drying no puede medir contra una solucion equivocada.

CONVENCION DE SIGNO: `bed_elevation` devuelve b (cota, negativa en el vaso);
`rest_depth` devuelve h0_campo = -b, que es lo que espera Domain.bathymetry.
Fuera de r = a la "profundidad de reposo" es negativa: el fondo emerge. Es
intencionado: ahi esta la tierra que la ola inunda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants.titan_params import TITAN

__all__ = ["ThackerPlanar"]


@dataclass(frozen=True)
class ThackerPlanar:
    """
    Oscilacion de superficie plana en cuenca parabolica.

    h0        : profundidad en el vertice de la cuenca en reposo [m].
    a         : radio de la orilla en reposo [m].
    u_amp     : amplitud de la velocidad U [m/s]. Controla la excursion de la
                orilla; debe mantenerse bien por debajo de sqrt(2 g h0) para que
                la cuenca no se vacie.
    g         : gravedad [m/s^2]. Por defecto TITAN.g.
    center    : centro de la cuenca (xc, yc) [m].
    eta0      : constante A0 de la superficie libre [m].
    """
    h0: float
    a: float
    u_amp: float
    g: float = TITAN.g
    center: tuple[float, float] = (0.0, 0.0)
    eta0: float = 0.0

    def __post_init__(self) -> None:
        if self.h0 <= 0 or self.a <= 0:
            raise ValueError("h0 y a deben ser positivos.")
        if abs(self.u_amp) >= np.sqrt(2.0 * self.g * self.h0):
            raise ValueError("u_amp demasiado grande: la cuenca se vaciaria.")

    # --- parametros derivados --------------------------------------------
    @property
    def omega(self) -> float:
        """Frecuencia angular de la oscilacion [rad/s]."""
        return np.sqrt(2.0 * self.g * self.h0) / self.a

    @property
    def period(self) -> float:
        """Periodo de la oscilacion [s]."""
        return 2.0 * np.pi / self.omega

    @property
    def shore_excursion(self) -> float:
        """
        Excursion maxima de la orilla respecto de r = a [m], en la aproximacion
        de pendiente pequena: la orilla se mueve donde eta cruza b.
        """
        return self.a * abs(self.u_amp) * self.omega / (2.0 * self.g * self.h0
                                                        / self.a)

    # --- campos ------------------------------------------------------------
    def _A(self, t: float) -> float:
        return self.eta0 + (self.u_amp ** 2 / (4.0 * self.g)) * \
            np.cos(2.0 * self.omega * t)

    def _B(self, t: float) -> float:
        return (self.u_amp * self.omega / self.g) * np.sin(self.omega * t)

    def bed_elevation(self, x, y) -> np.ndarray:
        """Cota de fondo b [m] (negativa dentro del vaso)."""
        xc, yc = self.center
        r2 = (np.asarray(x, float) - xc) ** 2 + (np.asarray(y, float) - yc) ** 2
        return self.h0 * (r2 / self.a ** 2 - 1.0)

    def rest_depth(self, x, y) -> np.ndarray:
        """h0_campo = -b. NEGATIVA fuera de r = a (tierra emergida)."""
        return -self.bed_elevation(x, y)

    def surface(self, x, y, t: float) -> np.ndarray:
        """Superficie libre eta(x,t) [m]. Plana, no depende de y."""
        xc, _ = self.center
        return self._A(t) + self._B(t) * (np.asarray(x, float) - xc)

    def depth(self, x, y, t: float) -> np.ndarray:
        """Altura de columna h = max(0, eta - b) [m]."""
        return np.maximum(0.0, self.surface(x, y, t) - self.bed_elevation(x, y))

    def velocity(self, x, y, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        (u, v). La velocidad analitica es uniforme; se anula donde esta seco,
        que es lo que hace cualquier solver razonable y lo que permite comparar.
        """
        h = self.depth(x, y, t)
        u = np.where(h > 0.0, self.u_amp * np.cos(self.omega * t), 0.0)
        return u, np.zeros_like(u)

    def state(self, x, y, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estado en variables PRIMITIVAS (zeta, u, v), ADR-001.

        zeta = h + b = superficie libre en la zona mojada; en la zona seca se
        toma zeta = -h0_campo (es decir h = 0), que es la convencion del motor.
        """
        h = self.depth(x, y, t)
        b = self.bed_elevation(x, y)
        u, v = self.velocity(x, y, t)
        return h + b, u, v

    def shoreline_radius(self, t: float) -> tuple[float, float]:
        """
        Posicion de la orilla a lo largo del eje x en el instante t [m].
        Resuelve eta = b sobre y = yc: h0(x^2/a^2 - 1) = A + B x.
        """
        A, B = self._A(t), self._B(t)
        k = self.h0 / self.a ** 2
        disc = B * B + 4.0 * k * (A + self.h0)
        if disc < 0.0:
            return (np.nan, np.nan)
        r = np.sqrt(disc)
        xc, _ = self.center
        return (xc + (B - r) / (2.0 * k), xc + (B + r) / (2.0 * k))

    # --- verificacion del algebra ------------------------------------------
    def residual(self, x: float, y: float, t: float,
                 eps_s: float = 1e-2, eps_t: float = 1e-3) -> np.ndarray:
        """
        Residuo de las NLSWE evaluado por diferencias centradas de alta
        precision. Debe ser numericamente nulo en la region mojada.

        Devuelve (R_h, R_hu, R_hv).
        """
        def q(xx, yy, tt):
            h = float(self.depth(np.array([xx]), np.array([yy]), tt)[0])
            u = float(self.velocity(np.array([xx]), np.array([yy]), tt)[0][0])
            return np.array([h, h * u, 0.0])

        def fx(xx, yy, tt):
            h = float(self.depth(np.array([xx]), np.array([yy]), tt)[0])
            u = float(self.velocity(np.array([xx]), np.array([yy]), tt)[0][0])
            return np.array([h * u, h * u * u + 0.5 * self.g * h * h, 0.0])

        def fy(xx, yy, tt):
            return np.array([0.0, 0.0,
                             0.5 * self.g *
                             float(self.depth(np.array([xx]), np.array([yy]),
                                              tt)[0]) ** 2])

        dq = (q(x, y, t + eps_t) - q(x, y, t - eps_t)) / (2 * eps_t)
        dfx = (fx(x + eps_s, y, t) - fx(x - eps_s, y, t)) / (2 * eps_s)
        dfy = (fy(x, y + eps_s, t) - fy(x, y - eps_s, t)) / (2 * eps_s)

        h = float(self.depth(np.array([x]), np.array([y]), t)[0])
        bx = float((self.bed_elevation(np.array([x + eps_s]), np.array([y]))[0]
                    - self.bed_elevation(np.array([x - eps_s]),
                                         np.array([y]))[0]) / (2 * eps_s))
        by = float((self.bed_elevation(np.array([x]), np.array([y + eps_s]))[0]
                    - self.bed_elevation(np.array([x]),
                                         np.array([y - eps_s]))[0]) / (2 * eps_s))
        return dq + dfx + dfy + np.array([0.0, self.g * h * bx,
                                          self.g * h * by])
