"""
physics/wetting_drying.py
=========================
Tratamiento de la frontera movil (inundacion y desecacion costera).

El problema no es "poner h = 0 donde no hay agua". Un esquema de volumenes
finitos de segundo orden falla de tres maneras distintas en un frente mojado/seco
y cada una necesita su remedio:

1. POSITIVIDAD. La reconstruccion MUSCL puede extrapolar h < 0 en una celda
   parcialmente mojada, y de ahi salen sqrt(g h) complejos y NaN. Remedio:
   REDUCCION DE ORDEN en las celdas del frente (pendiente nula), que es
   incondicionalmente positiva porque el valor de celda ya lo es.

2. VELOCIDADES ESPURIAS. En una lamina delgada, u = hu/h divide dos cantidades
   que tienden a cero y produce velocidades enormes que colapsan el paso de
   tiempo o revientan la corrida. Remedio: desingularizacion de
   Kurganov-Petrova, que tiende suavemente a cero en vez de saturar.

3. EQUILIBRIO DE REPOSO CON ORILLA SECA. Un lago en reposo cuya superficie corta
   el fondo debe permanecer en reposo. Es una condicion MAS FUERTE que el lago
   en reposo completamente mojado de la compuerta 1, y es la que hay que
   comprobar ANTES de creerse cualquier resultado de inundacion.

CONSERVACION DE MASA. Ninguno de los remedios anteriores crea ni destruye masa:
la reduccion de orden actua sobre la reconstruccion (el valor medio de celda no
cambia), y la desingularizacion actua sobre la VELOCIDAD, no sobre h. El unico
punto donde se podria perder masa es el recorte h = max(h, 0); por eso el motor
lo contabiliza y los tests miden la deriva.

Todos los umbrales salen de SolverConfig.min_depth: no se introducen constantes
nuevas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WettingDryingConfig", "classify_cells", "front_mask",
           "desingularized_velocity_kp", "dry_out"]


@dataclass(frozen=True)
class WettingDryingConfig:
    """
    Parametros del tratamiento de frontera movil.

    min_depth        : umbral de celda seca [m]. Viene de SolverConfig.min_depth.
    reduce_order     : reducir a orden 1 en las celdas del frente. Recomendado.
    desingularize    : usar la desingularizacion de Kurganov-Petrova para u.
    front_stencil    : radio en celdas del entorno del frente que se degrada.
                       1 basta para MUSCL de 2 puntos.
    """
    min_depth: float = 1e-3
    reduce_order: bool = True
    desingularize: bool = True
    front_stencil: int = 1


def classify_cells(h: np.ndarray, min_depth: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (mascara_mojada, mascara_seca)."""
    wet = h > min_depth
    return wet, ~wet


def _dilate(mask: np.ndarray) -> np.ndarray:
    """
    Dilatacion de un paso, eje a eje, SIN ENVOLVER los extremos.

    NO se usa np.roll: envuelve el array, de modo que una celda seca en un
    extremo del dominio marcaria como frente a la celda del extremo OPUESTO.
    Sobre el array extendido con celdas fantasma eso degradaba a orden 1 la
    frontera contraria sin motivo. Se usa relleno por replica del borde
    ('edge'), que es la extension coherente con las celdas fantasma.
    """
    out = mask.copy()
    for eje in range(mask.ndim):
        pad = [(1, 1) if i == eje else (0, 0) for i in range(mask.ndim)]
        p = np.pad(mask, pad, mode="edge")
        lo = [slice(None)] * mask.ndim
        hi = [slice(None)] * mask.ndim
        lo[eje] = slice(0, -2)
        hi[eje] = slice(2, None)
        out |= p[tuple(lo)] | p[tuple(hi)]
    return out


def front_mask(h: np.ndarray, min_depth: float, stencil: int = 1
               ) -> np.ndarray:
    """
    Celdas del FRENTE: mojadas con al menos una vecina seca dentro del radio
    `stencil`, mas las propias secas. Son las celdas donde la reconstruccion se
    degrada a orden 1.

    Funciona igual en 1D (nx,) y en 2D (ny, nx): la dilatacion se hace eje a eje.
    """
    wet, dry = classify_cells(h, min_depth)
    vecino_seco = dry.copy()
    for _ in range(max(1, stencil)):
        vecino_seco = _dilate(vecino_seco)
    return dry | (wet & vecino_seco)


def desingularized_velocity_kp(h: np.ndarray, m: np.ndarray,
                               min_depth: float) -> np.ndarray:
    """
    Desingularizacion de Kurganov-Petrova:

        u = sqrt(2) * h * m / sqrt(h^4 + max(h^4, eps^4))

    Para h >> eps se reduce a m/h con error relativo O((eps/h)^4); para h -> 0
    tiende a cero suavemente en vez de explotar. eps = min_depth.

    Frente al umbral duro `u = m/h if h > eps else 0`, evita el salto de
    velocidad en el frente, que es fuente de ruido y de reduccion drastica del
    paso de tiempo.
    """
    eps4 = min_depth ** 4
    h2 = h * h
    den = np.sqrt(h2 * h2 + np.maximum(h2 * h2, eps4))
    seguro = den > 0.0
    out = np.zeros_like(h)
    out[seguro] = np.sqrt(2.0) * h[seguro] * m[seguro] / den[seguro]
    return out


def dry_out(h: np.ndarray, hu: np.ndarray, hv: np.ndarray,
            min_depth: float) -> None:
    """
    Impone el estado seco IN PLACE: h >= 0 y momento nulo bajo el umbral.

    NO toca h salvo para recortar negativos: la masa recortada la contabiliza el
    motor a traves de sus diagnosticos, de modo que una perdida de masa se ve en
    el test en vez de esconderse aqui.
    """
    np.maximum(h, 0.0, out=h)
    seca = h <= min_depth
    if np.any(seca):
        hu[seca] = 0.0
        hv[seca] = 0.0


def mass_clipped(h_antes: np.ndarray, h_despues: np.ndarray,
                 cell_measure: float) -> float:
    """
    Masa creada por el recorte de positividad en un paso [kg/rho, es decir m^3].
    Positiva = se ha CREADO masa (h negativo elevado a cero). Es el diagnostico
    honesto de cuanto miente el recorte.
    """
    return float(np.sum(np.maximum(0.0, h_despues - h_antes)) * cell_measure)
