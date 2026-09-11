"""
core/reconstruction.py
======================
Reconstruccion MUSCL con limitador minmod y reconstruccion hidrostatica
(Audusse et al. 2004) para preservar el equilibrio de reposo ("well-balanced").

SUPUESTOS NUMERICOS EXPLICITOS
------------------------------
* Se reconstruyen linealmente la SUPERFICIE LIBRE eta = h + b, la velocidad u,
  la velocidad transversal v y el FONDO b. Nunca se reconstruye h directamente:
  hacerlo destruye el equilibrio de reposo sobre batimetria variable.
* Limitador minmod (TVD, orden 2 en zonas suaves, orden 1 en extremos).
* Con eta constante y u = v = 0 todas las pendientes de eta y u son nulas, de
  modo que el esquema recupera el equilibrio de reposo de forma EXACTA.
* order=1 desactiva la reconstruccion (pendientes nulas): util para verificar
  el orden de convergencia del esquema base.
"""

from __future__ import annotations
import numpy as np

__all__ = ["minmod_slope", "mc_slope", "centered_slope", "slope",
           "muscl_edges", "hydrostatic_reconstruction"]


def minmod_slope(q: np.ndarray) -> np.ndarray:
    """
    Pendiente limitada minmod (por unidad de celda) a lo largo del ULTIMO EJE.

    Funciona igual para arrays 1D (nx+2*NG,) y 2D (ny+2*NG, nx+2*NG): el motor
    2D recorre el eje y transponiendo, de modo que un unico kernel direccional
    sirve para ambas direcciones (ver core/fvm_kernel.py).

    Devuelve un array del mismo tamano; las dos celdas de los extremos reciben
    pendiente nula por falta de vecinos (no se usan en el calculo de flujos).
    """
    s = np.zeros_like(q)
    dq_back = q[..., 1:-1] - q[..., :-2]     # diferencia hacia atras
    dq_fwd = q[..., 2:] - q[..., 1:-1]       # diferencia hacia adelante
    # minmod(a, b): 0 si tienen signo distinto, si no el de menor magnitud.
    s[..., 1:-1] = np.where(
        dq_back * dq_fwd > 0.0,
        np.sign(dq_back) * np.minimum(np.abs(dq_back), np.abs(dq_fwd)),
        0.0)
    return s


def mc_slope(q: np.ndarray) -> np.ndarray:
    """
    Pendiente limitada monotonized-central (MC, van Leer) al ultimo eje.

    Menos disipativa que minmod, sigue siendo TVD. Se ofrece como alternativa
    para el test de orden formal (compuerta 4, 2D): los limitadores TVD
    degradan el orden en los extremos suaves, y MC lo degrada menos que minmod.
    """
    s = np.zeros_like(q)
    dq_back = q[..., 1:-1] - q[..., :-2]
    dq_fwd = q[..., 2:] - q[..., 1:-1]
    dq_cen = 0.5 * (dq_back + dq_fwd)
    lim = np.minimum(np.minimum(2.0 * np.abs(dq_back), 2.0 * np.abs(dq_fwd)),
                     np.abs(dq_cen))
    s[..., 1:-1] = np.where(dq_back * dq_fwd > 0.0, np.sign(dq_cen) * lim, 0.0)
    return s


def centered_slope(q: np.ndarray) -> np.ndarray:
    """
    Pendiente centrada SIN limitar. NO es TVD: produce oscilaciones en choques.

    Se usa exclusivamente para demostrar el orden formal 2 del esquema base en
    soluciones suaves (compuerta 4). NUNCA en produccion.
    """
    s = np.zeros_like(q)
    s[..., 1:-1] = 0.5 * (q[..., 2:] - q[..., :-2])
    return s


_LIMITERS = {"minmod": minmod_slope, "mc": mc_slope, "none": centered_slope}


def slope(q: np.ndarray, limiter: str = "minmod") -> np.ndarray:
    """Pendiente segun el limitador seleccionado."""
    try:
        return _LIMITERS[limiter](q)
    except KeyError:
        raise ValueError(f"Limitador desconocido: {limiter!r}. "
                         f"Validos: {tuple(_LIMITERS)}") from None


def muscl_edges(q: np.ndarray, order: int = 2,
                limiter: str = "minmod") -> tuple[np.ndarray, np.ndarray]:
    """
    Valores en los bordes izquierdo y derecho de cada celda: (q_L, q_R).

    order=1 -> reconstruccion constante a trozos (q_L = q_R = q).
    order=2 -> lineal con el limitador indicado.
    """
    if order == 1:
        return q.copy(), q.copy()
    s = slope(q, limiter)
    return q - 0.5 * s, q + 0.5 * s


def hydrostatic_reconstruction(hL: np.ndarray, bL: np.ndarray,
                               hR: np.ndarray, bR: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruccion hidrostatica de Audusse en una interfaz.

    Entrada: alturas y cotas de fondo a ambos lados de la interfaz.
    Salida:  alturas corregidas (h*_L, h*_R) que comparten la cota b* = max(bL,bR).

    Con eta_L == eta_R (reposo) se obtiene h*_L == h*_R == eta - b*, lo que hace
    que el flujo de Riemann sea el de un estado uniforme y el termino fuente
    cancele exactamente el gradiente de presion hidrostatica.
    """
    b_star = np.maximum(bL, bR)
    hs_L = np.maximum(0.0, hL + bL - b_star)
    hs_R = np.maximum(0.0, hR + bR - b_star)
    return hs_L, hs_R
