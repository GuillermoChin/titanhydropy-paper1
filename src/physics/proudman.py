"""
physics/proudman.py
===================
Numero de Froude atmosferico-marino y respuesta de Proudman.

TEORIA (SWE lineal 1D, sin friccion, profundidad constante h, fondo plano)
--------------------------------------------------------------------------
Para una perturbacion de presion P'(x - U t) que viaja a velocidad U sobre una
capa de profundidad h, la respuesta ESTACIONARIA de la superficie libre es

        zeta(x) = - P'(x) / (rho g (1 - F^2)),      F = U / sqrt(g h)

es decir, la respuesta de barometro inverso amplificada por 1/|1 - F^2|. El
factor diverge en F -> 1: es la RESONANCIA DE PROUDMAN. En F = 1 no existe
solucion estacionaria; la amplitud crece linealmente con el tiempo (o con la
distancia recorrida), con tasa

        d zeta / dt  ~  (U / (2 rho g L_p)) * P'_amp      (escala)

donde L_p es la escala espacial de la perturbacion. Aqui solo se expone la
relacion asintotica exacta y la tasa se deja al diagnostico numerico.

Referencia conceptual: Proudman (1929). NO se citan DOIs no verificados.

Todos los valores fisicos (g, rho) provienen de constants/titan_params.py.
"""

from __future__ import annotations

import numpy as np

from constants.titan_params import (TITAN, DEFAULT_FLUID, CryogenicFluid,
                                    inverse_barometer, resonant_depth,
                                    resonant_depth_band, shallow_water_speed)

__all__ = [
    "atmospheric_froude",
    "resonance_speed",
    "static_ib_response",
    "proudman_amplification",
    "amplification_from_field",
    "bound_amplification",
    "froude_field",
    "resonance_mask",
    "resonant_depth",
    "resonant_depth_band",
    "resonant_region_summary",
    "amplification_depth_profile",
    "classify_field_shape",
]


def resonance_speed(depth: float, g: float = TITAN.g) -> float:
    """Velocidad de tormenta que resuena, U_res = c = sqrt(g h) [m/s]."""
    return shallow_water_speed(depth, g=g)


def atmospheric_froude(storm_speed: float, depth: float,
                       g: float = TITAN.g) -> float:
    """
    Numero de Froude atmosferico-marino F = U / sqrt(g h) [adimensional].

    F < 1 subcritico, F = 1 resonancia de Proudman, F > 1 supercritico.
    """
    return storm_speed / shallow_water_speed(depth, g=g)


def static_ib_response(delta_p: float, fluid: CryogenicFluid = DEFAULT_FLUID,
                       g: float = TITAN.g) -> float:
    """
    Respuesta estatica de barometro inverso zeta = -dP/(rho g) [m].
    Delega en constants.titan_params.inverse_barometer (fuente unica).
    """
    return inverse_barometer(delta_p, fluid=fluid, g=g)


def proudman_amplification(froude: float | np.ndarray) -> float | np.ndarray:
    """
    Factor de amplificacion estacionario respecto al barometro inverso:

        R(F) = 1 / |1 - F^2|

    Devuelve np.inf en F = 1 exactamente (no hay estado estacionario).
    """
    f = np.asarray(froude, dtype=float)
    d = np.abs(1.0 - f * f)
    out = np.where(d > 0.0, 1.0 / np.where(d > 0.0, d, 1.0), np.inf)
    return float(out) if np.ndim(froude) == 0 else out


def amplification_from_field(zeta: np.ndarray, delta_p: float,
                             fluid: CryogenicFluid = DEFAULT_FLUID,
                             g: float = TITAN.g) -> float:
    """
    Factor de amplificacion medido en una simulacion:

        R_num = max|zeta| / |zeta_IB|,   zeta_IB = -dP/(rho g)

    Es la magnitud que la compuerta 5 barre frente a F.
    """
    ib = abs(static_ib_response(delta_p, fluid=fluid, g=g))
    if ib <= 0.0:
        raise ValueError("delta_p debe ser no nulo para normalizar.")
    return float(np.max(np.abs(np.asarray(zeta, dtype=float)))) / ib


def bound_amplification(zeta: np.ndarray, x: np.ndarray, x_center: float,
                        delta_p: float, half_width_sigmas: float = 3.0,
                        sigma: float = 1.0,
                        fluid: CryogenicFluid = DEFAULT_FLUID,
                        g: float = TITAN.g) -> float:
    """
    Amplificacion de la RESPUESTA LIGADA (bound response): maximo de |zeta|
    restringido a una ventana centrada en la perturbacion de presion.

    POR QUE HACE FALTA
    ------------------
    La metrica global A = max|zeta| / |zeta_IB| sobre todo el dominio mezcla dos
    cosas fisicamente distintas:
      * la respuesta LIGADA, que viaja pegada a la perturbacion y es la que
        describe la teoria estacionaria 1/|1-F^2|;
      * la onda LIBRE de arranque, emitida al encender el forzamiento, que viaja
        a la celeridad c.
    En regimen SUBCRITICO (F<1) la onda libre adelanta a la perturbacion y acaba
    saliendo por la frontera abierta, asi que ambas metricas coinciden. En
    SUPERCRITICO (F>1) la onda libre queda REZAGADA dentro del dominio y el
    maximo global la recoge, sesgando A al alza (hasta ~12% en F = 1.36).

    Esta funcion aisla la respuesta ligada para poder compararla con la teoria.
    La metrica global sigue siendo la magnitud reportada como A: aqui no se
    sustituye, se complementa.
    """
    z = np.asarray(zeta, dtype=float)
    xx = np.asarray(x, dtype=float)
    ventana = np.abs(xx - x_center) < half_width_sigmas * sigma
    if not np.any(ventana):
        raise ValueError("La ventana de la respuesta ligada esta vacia.")
    ib = abs(static_ib_response(delta_p, fluid=fluid, g=g))
    if z.ndim == 1:
        sel = z[ventana]
    else:
        sel = z[..., ventana]
    return float(np.max(np.abs(sel))) / ib


def resonant_growth_estimate(delta_p: float, storm_speed: float, sigma: float,
                             elapsed: float,
                             fluid: CryogenicFluid = DEFAULT_FLUID,
                             g: float = TITAN.g) -> float:
    """
    Estimacion de ORDEN DE MAGNITUD de la amplitud en resonancia exacta (F=1):

        |zeta| ~ |zeta_IB| * (U * t) / (2 * sigma)

    Es una escala, NO una solucion exacta: la constante depende de la forma de
    la perturbacion. Se usa solo para comprobar que el crecimiento numerico en
    F -> 1 es del orden esperado, nunca como valor publicable.
    """
    return abs(static_ib_response(delta_p, fluid=fluid, g=g)) * \
        (storm_speed * elapsed) / (2.0 * sigma)


# ---------------------------------------------------------------------------
# Extension 2D: campo de Froude local e identificacion de F -> 1
# ---------------------------------------------------------------------------
def froude_field(storm_speed: float, depth: np.ndarray,
                 g: float = TITAN.g, min_depth: float = 1e-3) -> np.ndarray:
    """
    Campo de Froude atmosferico-marino LOCAL:

        F(x,y) = U / sqrt(g h(x,y))

    Es la magnitud que localiza la resonancia sobre una batimetria real: la
    tormenta tiene una sola velocidad U, pero la celeridad de onda varia punto a
    punto, de modo que la condicion F = 1 se cumple en una BANDA de profundidad,
    no en toda la cuenca.

    Las celdas secas (h <= min_depth) devuelven np.inf: alli la onda larga no
    esta definida y no deben contarse como resonantes.
    """
    h = np.asarray(depth, dtype=float)
    wet = h > min_depth
    out = np.full(h.shape, np.inf, dtype=float)
    out[wet] = storm_speed / np.sqrt(g * h[wet])
    return out


def resonance_mask(froude: np.ndarray, tol: float = 0.1) -> np.ndarray:
    """
    Mascara booleana de las celdas en resonancia: |F - 1| <= tol.

    `tol` es una DECISION DE ANALISIS, no un umbral fisico: define cuan cerca de
    F = 1 se considera "resonante". El valor por defecto 0.1 corresponde a una
    amplificacion estacionaria 1/|1-F^2| >= 5 aproximadamente.
    """
    f = np.asarray(froude, dtype=float)
    return np.isfinite(f) & (np.abs(f - 1.0) <= tol)


def resonant_region_summary(storm_speed: float, depth: np.ndarray,
                            cell_area: float, g: float = TITAN.g,
                            tol: float = 0.1, min_depth: float = 1e-3) -> dict:
    """
    Resumen cuantitativo de la region resonante para una velocidad de tormenta.

    Devuelve area resonante, fraccion del area mojada, rango de profundidad
    resonante y la profundidad exacta de resonancia h = U^2/g.
    """
    h = np.asarray(depth, dtype=float)
    f = froude_field(storm_speed, h, g=g, min_depth=min_depth)
    mask = resonance_mask(f, tol)
    wet = h > min_depth
    n_wet = int(np.count_nonzero(wet))
    n_res = int(np.count_nonzero(mask))
    return {
        "storm_speed_m_s": storm_speed,
        "resonant_depth_m": resonant_depth(storm_speed, g),
        "froude_tolerance": tol,
        "resonant_area_m2": n_res * cell_area,
        "wet_area_m2": n_wet * cell_area,
        "resonant_area_fraction": (n_res / n_wet) if n_wet else 0.0,
        "resonant_depth_range_m": ((float(h[mask].min()), float(h[mask].max()))
                                   if n_res else None),
        "froude_min": float(np.min(f[wet])) if n_wet else None,
        "froude_max": float(np.max(f[wet])) if n_wet else None,
    }


# ---------------------------------------------------------------------------
# Diagnostico de FORMA del campo de amplificacion
# ---------------------------------------------------------------------------
def amplification_depth_profile(depth: np.ndarray, amplification: np.ndarray,
                                n_bins: int = 20, min_depth: float = 1e-3,
                                min_cells: int = 20, statistic: str = "mean"
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perfil A(h): estadistico de la amplificacion por banda de profundidad.

    `statistic` : 'mean' | 'median' | 'p75' | 'p90' | 'p99' | 'max'.

    POR QUE EL ESTADISTICO IMPORTA (y no es una eleccion inocente). Sobre una
    costa muy recortada, la respuesta en agua somera es BIMODAL: unas pocas
    celdas expuestas amplifican mucho y la mayoria, en ensenadas resguardadas,
    responden por DEBAJO del barometro inverso. La media y la mediana quedan
    dominadas por las segundas; la cola alta describe a las primeras. Las dos
    cosas son ciertas y describen fenomenos distintos, asi que hay que reportar
    ambas en vez de elegir la que convenga.

    Devuelve (centros_de_banda, estadistico, n_celdas). Las bandas con menos de
    `min_cells` celdas se descartan: un estadistico sobre cuatro celdas no es un
    punto del perfil, es ruido.
    """
    h = np.asarray(depth, dtype=float)
    a = np.asarray(amplification, dtype=float)
    wet = h > min_depth
    if not np.any(wet):
        raise ValueError("No hay celdas mojadas.")
    hw, aw = h[wet], a[wet]

    def _stat(x: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(x))
        if statistic == "median":
            return float(np.median(x))
        if statistic == "max":
            return float(np.max(x))
        if statistic.startswith("p"):
            return float(np.percentile(x, float(statistic[1:])))
        raise ValueError(f"Estadistico desconocido: {statistic!r}")

    bordes = np.linspace(0.0, float(hw.max()), n_bins + 1)
    centros, valores, cuentas = [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (hw > lo) & (hw <= hi)
        n = int(np.count_nonzero(m))
        if n >= min_cells:
            centros.append(0.5 * (lo + hi))
            valores.append(_stat(aw[m]))
            cuentas.append(n)
    return np.array(centros), np.array(valores), np.array(cuentas)


def _pico_resonante(centros: np.ndarray, valores: np.ndarray, h_res: float,
                    resonance_tol: float, peak_prominence: float):
    """Maximo local INTERIOR del perfil dentro de `resonance_tol` de h_res."""
    for i in range(1, centros.size - 1):
        if abs(centros[i] - h_res) / h_res > resonance_tol:
            continue
        vecino = max(valores[i - 1], valores[i + 1])
        p = (valores[i] - vecino) / max(vecino, 1e-30)
        if valores[i] > valores[i - 1] and valores[i] > valores[i + 1] \
                and p > peak_prominence:
            return True, i, float(p)
    return False, None, 0.0


def classify_field_shape(depth: np.ndarray, amplification: np.ndarray,
                         storm_speed: float, g: float = TITAN.g,
                         n_bins: int = 20, min_depth: float = 1e-3,
                         resonance_tol: float = 0.25,
                         peak_prominence: float = 0.10,
                         statistics: tuple[str, ...] = ("mean", "median", "p90"),
                         shore_fraction: float = 0.20) -> dict:
    """
    Clasifica la FORMA del campo de amplificacion. Es el diagnostico que decide
    si se sostiene que F = 1 es necesaria pero NO suficiente para la resonancia.

    DOS AFIRMACIONES DISTINTAS
    --------------------------
    Esa hipotesis dice dos cosas que hay que evaluar por separado, porque no son
    igualmente robustas:

      (A) DECISIVA: no hay pico interior en la franja F ~ 1. Es la afirmacion
          que sostiene "F = 1 es necesaria pero NO suficiente". Se comprueba
          sobre TODOS los estadisticos pedidos: si ninguno muestra pico
          resonante, la conclusion no depende de como se resuma el campo.

      (B) DESCRIPTIVA: el campo decae hacia mar adentro desde un maximo costero.
          Esta SI depende del estadistico sobre una costa recortada, asi que
          se reporta por estadistico y NO se usa para decidir sobre la
          hipotesis.

    DOS CRITERIOS PARA (B), y hay que quedarse con el segundo:
      `monotonic_shoreward` exige que el maximo caiga EXACTAMENTE en la primera
          banda. Es demasiado estricto: la banda mas somera contiene celdas en
          secado donde la respuesta esta recortada, asi que un perfil de
          shoaling de manual puede tener su maximo en la SEGUNDA banda. Se
          conserva solo como informacion.
      `shoreward_decaying` exige que el maximo caiga dentro del `shore_fraction`
          mas somero del rango de profundidad (por defecto el 20%). Es el
          criterio que se usa, y esta declarado.

    `shape` refleja (A), que es lo decisivo:
      'no_resonant_peak'        ningun estadistico muestra pico en F ~ 1.
      'interior_resonant_peak'  al menos uno lo muestra. La hipotesis SE DEBILITA.

    `peak_prominence` = 0.10: el pico debe superar en al menos un 10% a la mayor
    de sus dos vecinas. Es una DECISION DE ANALISIS declarada, no un umbral
    fisico: sin ella cualquier rizado del perfil contaria como pico.
    """
    h_res = resonant_depth(storm_speed, g)
    perfiles, pico_en = {}, {}
    for est in statistics:
        c, v, n = amplification_depth_profile(
            depth, amplification, n_bins=n_bins, min_depth=min_depth,
            statistic=est)
        if c.size < 3:
            raise ValueError("Perfil demasiado corto para clasificar la forma.")
        hay, idx, prom = _pico_resonante(c, v, h_res, resonance_tol,
                                         peak_prominence)
        cerca = np.abs(c - h_res) / h_res <= resonance_tol
        h_span = float(c[-1])
        h_max_perfil = float(c[int(np.argmax(v))])
        perfiles[est] = {
            "bin_centers_m": c, "values": v, "n_cells": n,
            "i_max": int(np.argmax(v)),
            "depth_of_max_m": h_max_perfil,
            "monotonic_shoreward": bool(int(np.argmax(v)) == 0),
            "shoreward_decaying": bool(h_max_perfil <= shore_fraction * h_span),
            "depth_of_max_fraction": h_max_perfil / h_span if h_span else np.nan,
            "interior_peak_at_resonance": bool(hay),
            "peak_index": idx, "peak_prominence": prom,
            "A_resonant": float(np.mean(v[cerca])) if np.any(cerca) else np.nan,
            "A_shore": float(v[0]),
        }
        perfiles[est]["A_resonant_over_shore"] = (
            perfiles[est]["A_resonant"] / perfiles[est]["A_shore"]
            if perfiles[est]["A_shore"] else np.nan)
        pico_en[est] = hay

    algun_pico = any(pico_en.values())
    base = perfiles[statistics[0]]
    return {
        "shape": ("interior_resonant_peak" if algun_pico else "no_resonant_peak"),
        "h_resonant_m": h_res,
        "statistics": statistics,
        "profiles": perfiles,
        "interior_peak_any_statistic": bool(algun_pico),
        "interior_peak_by_statistic": pico_en,
        "monotonic_shoreward_by_statistic": {
            e: perfiles[e]["monotonic_shoreward"] for e in statistics},
        "shoreward_decaying_by_statistic": {
            e: perfiles[e]["shoreward_decaying"] for e in statistics},
        "shore_fraction": shore_fraction,
        "depth_of_max_by_statistic": {
            e: perfiles[e]["depth_of_max_m"] for e in statistics},
        "A_resonant_over_shore_by_statistic": {
            e: perfiles[e]["A_resonant_over_shore"] for e in statistics},
        # Atajos al estadistico principal, por compatibilidad de lectura.
        "bin_centers_m": base["bin_centers_m"],
        "A_mean": base["values"],
        "n_cells": base["n_cells"],
        "depth_of_max_m": base["depth_of_max_m"],
        "resonance_tol": resonance_tol,
        "prominence_threshold": peak_prominence,
    }


def summary(depth: float, storm_speed: float, delta_p: float,
            fluid: CryogenicFluid = DEFAULT_FLUID, g: float = TITAN.g) -> dict:
    """Resumen de los numeros de Proudman para un caso dado."""
    c = shallow_water_speed(depth, g=g)
    fr = storm_speed / c
    return {
        "depth_m": depth,
        "wave_speed_m_s": c,
        "storm_speed_m_s": storm_speed,
        "froude": fr,
        "ib_response_m": static_ib_response(delta_p, fluid=fluid, g=g),
        "amplification_steady": proudman_amplification(fr),
        "fluid": fluid.name,
        "g_m_s2": g,
        "rho_kg_m3": fluid.rho,
    }
