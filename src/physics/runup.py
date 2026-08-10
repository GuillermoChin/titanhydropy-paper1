"""
physics/runup.py
================
Diagnosticos de shoaling (asomeramiento) y run-up (remonte costero).

DEFINICIONES USADAS (explicitas, porque en la literatura conviven varias)
------------------------------------------------------------------------
* SHOALING: crecimiento de la amplitud de una onda larga al disminuir la
  profundidad, a flujo de energia constante. Ley de Green:

        zeta_2 / zeta_1 = (h_1 / h_2)^(1/4)

  Es una relacion de la teoria LINEAL de onda larga sin refraccion ni
  disipacion. Aqui se usa como CONTRASTE: si la amplificacion simulada sobre la
  batimetria real se separa mucho de Green, la diferencia la explican la
  refraccion, la resonancia o la no linealidad, no un error del modelo.

* RUN-UP: cota maxima alcanzada por la lamina de agua sobre el nivel de reposo,
  medida como la elevacion del terreno en el punto mojado mas alto.
  `runup_height` la mide en metros de cota.

* INUNDACION (inundation distance): distancia horizontal maxima entre la linea
  de costa en reposo y la linea de costa instantanea. Es la magnitud que
  interesa para "hasta donde llega el agua".

Ambas se calculan sobre la MASCARA MOJADA con el mismo `min_depth` del solver,
para que el diagnostico no use un criterio de mojado distinto del de la fisica.

TODO valor fisico (g) se importa de constants/titan_params.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants.titan_params import TITAN

__all__ = ["greens_law_amplitude", "shoaling_factor", "RunupResult",
           "compute_runup", "shoreline_mask", "track_max_envelope"]


# ---------------------------------------------------------------------------
# Shoaling
# ---------------------------------------------------------------------------
def greens_law_amplitude(amplitude_ref: float, depth_ref: float,
                         depth: np.ndarray) -> np.ndarray:
    """
    Amplitud predicha por la ley de Green al pasar de `depth_ref` a `depth`.

        zeta(h) = zeta_ref * (h_ref / h)^(1/4)

    Las celdas con h <= 0 devuelven NaN: la ley de onda larga no esta definida
    en seco y devolver un numero seria inventarlo.
    """
    h = np.asarray(depth, dtype=float)
    out = np.full(h.shape, np.nan)
    ok = h > 0.0
    out[ok] = amplitude_ref * (depth_ref / h[ok]) ** 0.25
    return out


def shoaling_factor(depth_ref: float, depth: np.ndarray) -> np.ndarray:
    """Factor de shoaling (h_ref/h)^(1/4), adimensional."""
    return greens_law_amplitude(1.0, depth_ref, depth)


# ---------------------------------------------------------------------------
# Run-up
# ---------------------------------------------------------------------------
@dataclass
class RunupResult:
    """Diagnosticos de remonte e inundacion en un instante o sobre una envolvente."""
    runup_height_m: float          # cota maxima mojada sobre el nivel de reposo
    inundation_distance_m: float   # avance horizontal maximo de la orilla
    wet_area_m2: float             # area mojada
    wet_area_change_m2: float      # cambio respecto de la de reposo
    max_zeta_m: float              # maxima elevacion de superficie libre
    n_newly_wet: int               # celdas inundadas respecto del reposo
    n_newly_dry: int               # celdas desecadas respecto del reposo

    def summary(self) -> str:
        return (f"run-up = {self.runup_height_m:.3f} m,  "
                f"inundacion = {self.inundation_distance_m/1e3:.3f} km,  "
                f"max|zeta| = {self.max_zeta_m:.3f} m,  "
                f"celdas inundadas = {self.n_newly_wet}")


def shoreline_mask(wet: np.ndarray) -> np.ndarray:
    """Celdas mojadas con al menos una vecina seca: la linea de orilla."""
    seca = ~wet
    borde = np.zeros_like(wet)
    for eje in range(wet.ndim):
        borde |= np.roll(seca, 1, axis=eje)
        borde |= np.roll(seca, -1, axis=eje)
    return wet & borde


def compute_runup(zeta: np.ndarray, rest_depth: np.ndarray,
                  X: np.ndarray, Y: np.ndarray, dx: float, dy: float,
                  min_depth: float = 1e-3) -> RunupResult:
    """
    Diagnosticos de remonte para un campo zeta dado.

    `rest_depth` es h0(x,y): positiva bajo el agua en reposo, negativa en tierra.
    La cota del terreno es z_terreno = -h0, de modo que el run-up es la cota
    maxima del terreno que esta mojado.

    La distancia de inundacion se mide como la distancia maxima de una celda
    mojada-y-antes-seca a la orilla de reposo mas proxima. Se calcula por fuerza
    bruta sobre las celdas de orilla; para mallas de este tamano es despreciable.
    """
    zeta = np.asarray(zeta, dtype=float)
    h0 = np.asarray(rest_depth, dtype=float)
    h = np.maximum(zeta + h0, 0.0)

    wet = h > min_depth
    wet_reposo = h0 > min_depth
    terreno = -h0                       # cota del fondo [m]

    if np.any(wet):
        runup = float(np.max(terreno[wet]))
        max_zeta = float(np.max(np.abs(zeta[wet])))
    else:
        runup, max_zeta = float("nan"), 0.0

    nuevas = wet & ~wet_reposo
    secas = wet_reposo & ~wet

    dist = 0.0
    if np.any(nuevas) and np.any(wet_reposo):
        orilla = shoreline_mask(wet_reposo)
        if np.any(orilla):
            ox, oy = X[orilla], Y[orilla]
            nx_, ny_ = X[nuevas], Y[nuevas]
            mejor = np.full(nx_.shape, np.inf)
            bloque = max(1, int(2.0e7 / max(nx_.size, 1)))
            for i in range(0, ox.size, bloque):
                d2 = (nx_[:, None] - ox[None, i:i + bloque]) ** 2 + \
                     (ny_[:, None] - oy[None, i:i + bloque]) ** 2
                mejor = np.minimum(mejor, np.min(d2, axis=1))
            dist = float(np.sqrt(np.max(mejor)))

    area = float(np.count_nonzero(wet) * dx * dy)
    area_reposo = float(np.count_nonzero(wet_reposo) * dx * dy)
    return RunupResult(
        runup_height_m=runup, inundation_distance_m=dist,
        wet_area_m2=area, wet_area_change_m2=area - area_reposo,
        max_zeta_m=max_zeta, n_newly_wet=int(np.count_nonzero(nuevas)),
        n_newly_dry=int(np.count_nonzero(secas)))


def track_max_envelope(envelope: np.ndarray | None,
                       zeta: np.ndarray) -> np.ndarray:
    """
    Envolvente de maximos de |zeta| a lo largo de una corrida.

    El run-up y el mapa de amplificacion se calculan sobre la ENVOLVENTE, no
    sobre el instante final: el instante final no tiene por que ser el de maxima
    inundacion, y usarlo subestimaria sistematicamente el remonte.
    """
    z = np.abs(np.asarray(zeta, dtype=float))
    return z.copy() if envelope is None else np.maximum(envelope, z)
