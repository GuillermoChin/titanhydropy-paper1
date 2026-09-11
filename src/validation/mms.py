"""
validation/mms.py
=================
Metodo de Soluciones Manufacturadas (MMS) para medir el ORDEN FORMAL de
exactitud del esquema sobre soluciones SUAVES.

POR QUE HACE FALTA
------------------
El dam-break mide ~1 en norma L1 porque contiene un choque, y ese es el limite
teorico para cualquier esquema conservativo sobre discontinuidades. No dice nada
sobre el orden formal. El Paper 1 afirma que el esquema es de orden 2 en zonas
suaves; esa afirmacion necesita su propia prueba, y MMS es la unica que no
depende de conocer una solucion analitica del problema fisico.

COMO FUNCIONA
-------------
Se ELIGE una solucion suave arbitraria (h, u, v) sobre un fondo b suave, se
sustituye en las NLSWE y se calcula ANALITICAMENTE el residuo S. Añadiendo S
como termino fuente, la solucion elegida pasa a ser solucion EXACTA del sistema
modificado. El error del solver frente a ella mide el error de discretizacion
puro, sin contaminacion de condiciones de frontera (dominio periodico) ni de
ondas de choque.

SOLUCION MANUFACTURADA (declarada explicitamente)
-------------------------------------------------
    phi   = kx*x + ky*y - w*t
    h     = H  + a  * sin(phi)
    u     = u0 * cos(phi)
    v     = v0 * sin(phi)
    b     = -Hb + b0 * cos(kbx*x) * cos(kby*y)

Todo es C-infinito y periodico. Se exige H > a (columna siempre mojada) y
Hb > |b0| (profundidad de reposo siempre positiva).

Las derivadas se escriben a mano, no se aproximan: si se aproximaran
numericamente, el test mediria el orden de esa aproximacion y no el del esquema.

NOTA SOBRE EL ORDEN ESPERADO Y LOS LIMITADORES
----------------------------------------------
Los limitadores TVD (minmod, MC) degradan el orden en los EXTREMOS suaves: es
una propiedad conocida y demostrada de la clase TVD, no un defecto de esta
implementacion. Por eso el barrido se corre con los tres limitadores:
  * 'none'   -> pendiente centrada sin limitar: mide el orden formal del
                esquema base. Debe dar 2.
  * 'mc'     -> limitador menos disipativo.
  * 'minmod' -> el de produccion, el mas disipativo.
Reportar los tres es lo honesto: el esquema base es de orden 2 y el limitador
de produccion cobra un peaje cuantificado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants.titan_params import TITAN

__all__ = ["ManufacturedSolution", "default_mms_1d", "default_mms_2d"]


@dataclass(frozen=True)
class ManufacturedSolution:
    """
    Solucion manufacturada suave y periodica para las NLSWE.

    Parametros geometricos: lx, ly definen el periodo espacial; los numeros de
    onda se derivan de ellos para que la solucion sea exactamente periodica en
    el dominio, condicion necesaria para que las fronteras periodicas no
    introduzcan error.
    """
    lx: float
    ly: float = 1.0
    H: float = 100.0        # profundidad media de columna [m]
    a: float = 5.0          # amplitud de la oscilacion de h [m]
    u0: float = 1.0         # amplitud de u [m/s]
    v0: float = 0.6         # amplitud de v [m/s]
    Hb: float = 100.0       # profundidad de reposo media [m]
    b0: float = 10.0        # amplitud del relieve del fondo [m]
    g: float = TITAN.g
    n_modes_x: int = 1
    n_modes_y: int = 1
    two_d: bool = True

    def __post_init__(self) -> None:
        if self.a >= self.H:
            raise ValueError("a < H: la columna debe permanecer mojada.")
        if abs(self.b0) >= self.Hb:
            raise ValueError("|b0| < Hb: la profundidad de reposo debe ser > 0.")

    # --- numeros de onda y frecuencia ------------------------------------
    @property
    def kx(self) -> float:
        return 2.0 * np.pi * self.n_modes_x / self.lx

    @property
    def ky(self) -> float:
        return (2.0 * np.pi * self.n_modes_y / self.ly) if self.two_d else 0.0

    @property
    def omega(self) -> float:
        """Frecuencia. Se fija a c*|k| para que la escala temporal sea la de
        una onda de gravedad del problema, no un valor arbitrario."""
        k = np.hypot(self.kx, self.ky)
        return np.sqrt(self.g * self.H) * k

    # --- campos exactos ---------------------------------------------------
    def _phi(self, x, y, t):
        return self.kx * np.asarray(x, dtype=float) + \
            self.ky * np.asarray(y, dtype=float) - self.omega * t

    def bed_elevation(self, x, y) -> np.ndarray:
        """Cota de fondo b [m] (negativa bajo el datum)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cy = np.cos(self.ky * y) if self.two_d else np.ones_like(y)
        return -self.Hb + self.b0 * np.cos(self.kx * x) * cy

    def rest_depth(self, x, y) -> np.ndarray:
        """Profundidad de reposo h0 = -b [m], siempre positiva."""
        return -self.bed_elevation(x, y)

    def exact(self, x, y, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solucion exacta (h, u, v) en el instante t."""
        phi = self._phi(x, y, t)
        h = self.H + self.a * np.sin(phi)
        u = self.u0 * np.cos(phi)
        v = (self.v0 * np.sin(phi)) if self.two_d else np.zeros_like(h)
        return h, u, v

    def exact_primitive(self, x, y, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solucion exacta en variables PRIMITIVAS (zeta, u, v)."""
        h, u, v = self.exact(x, y, t)
        return h + self.bed_elevation(x, y), u, v

    # --- termino fuente manufacturado -------------------------------------
    def source(self, x, y, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Residuo S = (S_h, S_hu, S_hv) que hace exacta a la solucion elegida.
        Todas las derivadas son analiticas.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        kx, ky, w, g = self.kx, self.ky, self.omega, self.g
        a, u0, v0 = self.a, self.u0, (self.v0 if self.two_d else 0.0)

        phi = self._phi(x, y, t)
        sp, cp = np.sin(phi), np.cos(phi)

        h = self.H + a * sp
        h_t = -a * w * cp
        h_x = a * kx * cp
        h_y = a * ky * cp

        u = u0 * cp
        u_t = u0 * w * sp
        u_x = -u0 * kx * sp
        u_y = -u0 * ky * sp

        v = v0 * sp
        v_t = -v0 * w * cp
        v_x = v0 * kx * cp
        v_y = v0 * ky * cp

        # Derivadas del fondo (analiticas).
        cy = np.cos(ky * y) if self.two_d else np.ones_like(y)
        sy = np.sin(ky * y) if self.two_d else np.zeros_like(y)
        b_x = -self.b0 * kx * np.sin(kx * x) * cy
        b_y = (-self.b0 * ky * np.cos(kx * x) * sy) if self.two_d \
            else np.zeros_like(x)

        # Continuidad: d_t h + d_x(h u) + d_y(h v)
        s_h = h_t + (h_x * u + h * u_x) + (h_y * v + h * v_y)

        # Momento x: d_t(hu) + d_x(h u^2 + g h^2/2) + d_y(h u v) + g h d_x b
        s_hu = (h_t * u + h * u_t) \
            + (h_x * u * u + 2.0 * h * u * u_x + g * h * h_x) \
            + (h_y * u * v + h * u_y * v + h * u * v_y) \
            + g * h * b_x

        # Momento y: d_t(hv) + d_x(h u v) + d_y(h v^2 + g h^2/2) + g h d_y b
        s_hv = (h_t * v + h * v_t) \
            + (h_x * u * v + h * u_x * v + h * u * v_x) \
            + (h_y * v * v + 2.0 * h * v * v_y + g * h * h_y) \
            + g * h * b_y

        return s_h, s_hu, s_hv

    # --- adaptadores a la firma extra_source de cada motor ----------------
    def source_1d(self):
        """Callable (x, t) -> (S_h, S_hu, S_hv) para NativeFVMSolver."""
        def _f(x, t):
            return self.source(x, np.zeros_like(np.asarray(x, dtype=float)), t)
        return _f

    def source_2d(self):
        """Callable (X, Y, t) -> (S_h, S_hu, S_hv) para NativeFVMSolver2D."""
        def _f(X, Y, t):
            return self.source(X, Y, t)
        return _f


def default_mms_1d(lx: float = 1000.0) -> ManufacturedSolution:
    """Solucion manufacturada 1D de referencia (sin dependencia en y)."""
    return ManufacturedSolution(lx=lx, ly=lx, two_d=False, v0=0.0)


def default_mms_2d(lx: float = 1000.0, ly: float = 1000.0
                   ) -> ManufacturedSolution:
    """Solucion manufacturada 2D de referencia."""
    return ManufacturedSolution(lx=lx, ly=ly, two_d=True)
