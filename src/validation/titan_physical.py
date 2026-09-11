"""
validation/titan_physical.py
============================
Verificaciones de coherencia FÍSICA (Nivel C) sobre Titán.

Nivel A = invariantes numéricos (conservation.py).
Nivel B = soluciones analíticas (analytic.py).
Nivel C = "¿tiene sentido esto para Titán?". Este módulo.

Todo valor físico se importa de constants/titan_params.py. Aquí NO se escribe
ningún número físico nuevo; solo relaciones y rangos de plausibilidad, y estos
últimos se declaran explícitamente como criterios de cordura, no como datos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED,
                                    REFERENCE_DEPTHS, SWEEP_SPEED_RANGE, TITAN,
                                    CryogenicFluid, audit_provenance,
                                    inverse_barometer, resonant_depth,
                                    resonant_depth_band, sanity_check,
                                    shallow_water_speed)

__all__ = ["run_sanity", "pending_constants", "proudman_overlap",
           "check_inverse_barometer", "check_zeta_magnitude",
           "check_velocity_magnitude", "PhysicalCheck", "run_all_checks"]


# ---------------------------------------------------------------------------
# Criterios de cordura (NO son datos de Titán; son cotas de plausibilidad)
# ---------------------------------------------------------------------------
# Se declaran aquí, y no en titan_params.py, precisamente porque no son
# constantes físicas medidas: son el rango fuera del cual un resultado indica
# un error de código o de montaje, no un descubrimiento.
ZETA_MAX_PLAUSIBLE = 50.0      # [m]  elevación de superficie libre
SPEED_MAX_PLAUSIBLE = 30.0     # [m/s] velocidad de corriente
FROUDE_FLOW_MAX = 1.0          # el flujo debe permanecer subcrítico


@dataclass
class PhysicalCheck:
    """Resultado de una verificación de Nivel C."""
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'OK ' if self.passed else 'FALLA'}] {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# 1. Coherencia de las constantes
# ---------------------------------------------------------------------------
def run_sanity() -> None:
    """Ejecuta sanity_check() de la fuente única de constantes."""
    sanity_check()


def pending_constants() -> list[str]:
    """Constantes aún en TO_VERIFY. No se pueden publicar resultados con ellas."""
    return audit_provenance()


def proudman_overlap() -> dict:
    """
    El 'linchpin' del proyecto, evaluado con OBSERVED_FRONT_SPEED (el DATO).

    Condición correcta de resonancia alcanzable: la banda resonante
    h_res = U^2/g debe INTERSECAR el rango de profundidades [0, h_max] de la
    cuenca. No se exige que c(h_max) caiga dentro del rango de velocidades: esa
    condición es demasiado fuerte, porque la resonancia exige igualar la
    celeridad en ALGÚN punto del recorrido, no en el más profundo.
    """
    v_min, v_max, status, src = OBSERVED_FRONT_SPEED
    h_res_min, h_res_max = resonant_depth_band()
    out = {
        "front_speed_range": (v_min, v_max),
        "front_status": status,
        "front_source": src,
        "resonant_depth_band_m": (h_res_min, h_res_max),
        "sweep_speed_range": SWEEP_SPEED_RANGE,
        "basins": {},
    }
    for name, (depth, dstatus, dsrc) in REFERENCE_DEPTHS.items():
        c = shallow_water_speed(depth)
        out["basins"][name] = {
            "depth_m": depth,
            "wave_speed_m_s": c,
            "froude_at_min_front": v_min / c,
            "froude_at_max_front": v_max / c,
            # ¿Resuena a profundidad MÁXIMA? (condición fuerte, que NO es la correcta)
            "resonance_at_max_depth": bool(v_min <= c <= v_max),
            # ¿Resuena en ALGÚN punto de la cuenca? (condición correcta)
            "resonance_reachable": bool(h_res_min < depth),
            "resonant_depth_band_m": (h_res_min, min(h_res_max, depth)),
            "depth_status": dstatus,
            "depth_source": dsrc,
        }
    return out


# ---------------------------------------------------------------------------
# 2. Verificaciones sobre campos simulados
# ---------------------------------------------------------------------------
def check_inverse_barometer(zeta_response: float, delta_p: float,
                            fluid: CryogenicFluid = DEFAULT_FLUID,
                            rtol: float = 0.05) -> PhysicalCheck:
    """
    La respuesta estática medida debe coincidir con inverse_barometer().
    `zeta_response` es la diferencia de elevación entre la zona forzada y la
    zona de referencia; `delta_p` la diferencia de presión correspondiente.
    """
    teorico = inverse_barometer(delta_p, fluid=fluid)
    err = abs(zeta_response - teorico) / max(abs(teorico), 1e-30)
    return PhysicalCheck(
        "barómetro inverso", err < rtol,
        f"zeta={zeta_response:.6f} m vs {teorico:.6f} m teórico "
        f"(rho={fluid.rho}, g={TITAN.g}); error {err:.3%} "
        f"{'<' if err < rtol else '>='} {rtol:.0%}")


def check_zeta_magnitude(zeta: np.ndarray,
                         limite: float = ZETA_MAX_PLAUSIBLE,
                         wet: np.ndarray | None = None) -> PhysicalCheck:
    """
    |zeta| debe quedar dentro de un rango físicamente razonable para Titán.
    Un valor fuera de esta cota no es un descubrimiento: es un error de montaje.

    `wet` es OBLIGATORIO en dominios con wetting-drying. En las celdas SECAS el
    motor almacena zeta = -h0 por convención, que en tierra emergida es la
    **cota del terreno**, no una elevación de superficie libre. Sin la máscara,
    esta verificación leía cotas de 179 m en las esquinas del dominio de Ligeia
    y fallaba por un motivo falso.
    """
    z = np.asarray(zeta, dtype=float)
    if wet is not None:
        z = z[np.asarray(wet, dtype=bool)]
    zmax = float(np.max(np.abs(z))) if z.size else 0.0
    finito = bool(np.all(np.isfinite(z)))
    ok = finito and zmax < limite
    return PhysicalCheck(
        "magnitud de zeta", ok,
        f"max|zeta| = {zmax:.4f} m (cota de cordura {limite} m), "
        f"finito={finito}, celdas evaluadas={z.size}")


def check_velocity_magnitude(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                             limite: float = SPEED_MAX_PLAUSIBLE,
                             open_water_depth: float = 1.0) -> PhysicalCheck:
    """
    La corriente debe ser razonable y el flujo permanecer subcrítico
    (Fr_flujo = |U|/sqrt(g h) < 1) EN MAR ABIERTO.

    `open_water_depth` [m] separa el mar abierto de la zona de batida (swash).
    En la lámina delgada de un frente de inundación el flujo ES supercrítico:
    eso es física correcta de run-up, no un fallo. Aplicar el criterio a la
    lámina delgada convertiría un comportamiento esperado en una falsa alarma.
    Por eso el criterio se aplica solo donde h > open_water_depth, y el valor de
    la zona de batida se reporta aparte, sin hacer fallar la verificación.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    h = np.asarray(depth, dtype=float)
    speed = np.sqrt(u * u + v * v)
    abierto = h > open_water_depth
    mojado = h > 1e-3

    smax = float(np.max(speed[mojado])) if np.any(mojado) else 0.0
    fr_sw = (speed[mojado] / np.sqrt(TITAN.g * np.maximum(h[mojado], 1e-12))) \
        if np.any(mojado) else np.array([0.0])

    if not np.any(abierto):
        # NO CONCLUYENTE, y por tanto NO se aprueba. Si no hay ninguna celda de
        # mar abierto, el criterio de Froude no tiene donde aplicarse; devolver
        # "OK" seria aprobar por ausencia de evidencia. Un test de control
        # negativo detecto exactamente este agujero.
        return PhysicalCheck(
            "magnitud de velocidad", False,
            f"NO CONCLUYENTE: no hay celdas de mar abierto (h > "
            f"{open_water_depth} m) donde aplicar el criterio de Froude. "
            f"max|U| = {smax:.4f} m/s; Fr máx en la lámina mojada = "
            f"{float(np.max(fr_sw)):.4f}")

    frmax = float(np.max(speed[abierto] / np.sqrt(TITAN.g * h[abierto])))
    ok = (smax < limite and frmax < FROUDE_FLOW_MAX
          and np.all(np.isfinite(speed)))
    return PhysicalCheck(
        "magnitud de velocidad", ok,
        f"max|U| = {smax:.4f} m/s (cota {limite} m/s); "
        f"Fr_flujo máx en mar abierto (h>{open_water_depth} m) = {frmax:.4f} "
        f"(debe ser < {FROUDE_FLOW_MAX}); Fr máx incluyendo la zona de batida "
        f"= {float(np.max(fr_sw)):.4f} (informativo)")


def check_resonant_band_intersects_basins() -> PhysicalCheck:
    """
    Con OBSERVED_FRONT_SPEED, la banda resonante debe intersecar el rango de
    profundidades de cada cuenca de referencia. Es la forma correcta del
    linchpin.
    """
    h_min, h_max = resonant_depth_band()
    fallos = [n for n, (d, _, _) in REFERENCE_DEPTHS.items() if h_min >= d]
    return PhysicalCheck(
        "banda resonante vs cuencas", not fallos,
        f"banda resonante = [{h_min:.1f}, {h_max:.1f}] m con "
        f"U ∈ {OBSERVED_FRONT_SPEED[:2]} m/s; "
        f"cuencas sin intersección: {fallos or 'ninguna'}")


def check_wave_speed_consistency() -> PhysicalCheck:
    """c = sqrt(g h) y h_res = U^2/g deben ser mutuamente inversas."""
    errores = []
    for u in (2.0, 5.0, 10.0, 14.7, 20.0):
        h = resonant_depth(u)
        errores.append(abs(shallow_water_speed(h) - u) / u)
    err = max(errores)
    return PhysicalCheck(
        "consistencia c(h) <-> h_res(U)", err < 1e-12,
        f"error relativo máximo = {err:.3e}")


def run_all_checks(zeta: np.ndarray | None = None,
                   u: np.ndarray | None = None,
                   v: np.ndarray | None = None,
                   depth: np.ndarray | None = None,
                   zeta_response: float | None = None,
                   delta_p: float | None = None,
                   wet: np.ndarray | None = None) -> list[PhysicalCheck]:
    """
    Ejecuta todas las verificaciones de Nivel C aplicables a lo que se le pase.
    Las verificaciones de constantes se ejecutan siempre.

    `wet` : máscara de celdas mojadas. Si no se pasa pero sí `depth`, se deriva
    de ella. Es imprescindible en dominios con tierra emergida (ver
    check_zeta_magnitude).
    """
    if wet is None and depth is not None:
        wet = np.asarray(depth, dtype=float) > 1e-3
    checks = [check_wave_speed_consistency(),
              check_resonant_band_intersects_basins()]
    if zeta is not None:
        checks.append(check_zeta_magnitude(zeta, wet=wet))
    if u is not None and v is not None and depth is not None:
        checks.append(check_velocity_magnitude(u, v, depth))
    if zeta_response is not None and delta_p is not None:
        checks.append(check_inverse_barometer(zeta_response, delta_p))
    return checks
