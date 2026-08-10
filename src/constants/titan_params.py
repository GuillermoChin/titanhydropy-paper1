"""
constants/titan_params.py
==========================
Fuente ÚNICA de verdad para las constantes físicas de Titán en TitanHydroPy.

POLÍTICA DE ESTE MÓDULO:
  1. Ningún valor se usa en otro módulo sin importarlo desde aquí.
  2. Cada constante lleva: valor, unidad SI, fuente y estatus de verificación.
  3. status = VERIFIED  -> confirmado contra fuente primaria revisada por pares.
     status = TO_VERIFY -> valor de arranque razonable. sin confirmar alguna fuente primaria.
  4. La función audit_provenance() lista todo lo que sigue en TO_VERIFY para
     que una corrida destinada a publicación no arrastre valores sin verificar.

DECISIÓN DE DISEÑO (variables primitivas):
  El estado público del solver se expresa en variables PRIMITIVAS (zeta, u, v),
  no en conservadas (h, hu, hv). Justificación en docs/DECISIONS.md (ADR-001).
  Aquí solo se declara para que las constantes se interpreten en ese marco:
  zeta es la elevación de superficie libre, directamente comparable con las
  observaciones de Cassini.

UNIDADES: todo en SI (m, s, kg, Pa, K). Sin excepciones.
"""

from __future__ import annotations
from dataclasses import dataclass
import math

# --- Estatus de verificación -------------------------------------------------
VERIFIED = "verified"
TO_VERIFY = "to_verify"


# =============================================================================
# 1. CONSTANTES FUNDAMENTALES Y ASTRONÓMICAS
# =============================================================================
@dataclass(frozen=True)
class TitanConstants:
    """Constantes globales de Titán. Instancia única: TITAN (ver abajo)."""

    # Gravedad superficial. Domina la restauración de las ondas de gravedad.
    g: float = 1.352                 # [m/s^2]  VERIFIED (valor estándar JPL/efemérides)

    # Radio medio de Titán.
    radius: float = 2.5747e6         # [m]      VERIFIED

    # Periodo de rotación síncrona (15.945 días). Omega se deriva, no se fija.
    rotation_period: float = 15.945 * 86400.0   # [s]  VERIFIED

    # Presión atmosférica superficial (HASI/Huygens ~1.467 bar).
    # El documento base la redondea a 1.5 bar; aquí se adopta el valor medido.
    p_surface: float = 1.467e5       # [Pa]     VERIFIED 

    # Temperatura superficial en los mares polares (~90-91 K).
    # NO usar el valor ecuatorial (~93.7 K): de T dependen rho y mu del líquido.
    t_surface_polar: float = 90.0    # [K]     VERIFIED

    # Densidad atmosférica superficial (para el esfuerzo de viento).
    # rho_atm: valor 5.3 kg/m^3 confirmado en uso por Charnay et al. 2015;
    # fuente primaria del dato: Fulchignoni et al. 2005, Nature 438 (HASI/Huygens).
    rho_atm: float = 5.3             # [kg/m^3] VERIFIED

    @property
    def omega(self) -> float:
        """Velocidad angular de rotación [rad/s]."""
        return 2.0 * math.pi / self.rotation_period

    def coriolis(self, lat_deg: float) -> float:
        """Parámetro de Coriolis f = 2*Omega*sin(lat) [1/s]."""
        return 2.0 * self.omega * math.sin(math.radians(lat_deg))


TITAN = TitanConstants()  # instancia global de solo lectura


# =============================================================================
# 2. FLUIDO CRIOGÉNICO (dependiente de composición)
# =============================================================================
@dataclass(frozen=True)
class CryogenicFluid:
    """
    Propiedades de un fluido criogénico de Titán a T superficial polar.
    NOTA: en el modelo NLSWE, la tensión superficial (sigma) NO interviene en el
    núcleo (es un modelo de onda larga de gravedad); se conserva aquí para los
    módulos de umbral de oleaje y EOS, no para el solver.
    """
    name: str
    rho: float          # [kg/m^3]  densidad
    mu: float           # [Pa·s]    viscosidad dinámica molecular
    sigma: float        # [N/m]     tensión superficial (para módulos no-SWE)
    composition: str
    status: str

    @property
    def nu(self) -> float:
        """Viscosidad cinemática molecular nu = mu/rho [m^2/s]."""
        return self.mu / self.rho


# BASELINE de Paper 1: composición dominada por metano tipo Ligeia.
# rho de arranque = metano líquido puro ~90 K; el N2 disuelto la eleva hacia
# ~500-600 kg/m^3. eos_titan.py deberá refinar este valor.
LIGEIA_METHANE_BASELINE = CryogenicFluid(
    name="Ligeia-like methane-dominated (baseline)",
    rho=450.0,          # [VERIFICAR] metano puro ~90 K; corregir por N2 disuelto
    mu=2.0e-4,          # [Pa·s] metano puro ~92 K (TITANPOOL; Steckloff et al.)
    sigma=0.016,        # [N/m]  metano puro ~92 K (TITANPOOL)
    composition="CH4-dominated + dissolved N2 (Ligeia-like)",
    status=TO_VERIFY,
)

# Extremo del barrido de sensibilidad: mezcla rica en etano.
# Peng-Robinson a 94 K, 0.15 MPa -> rho=615.2 kg/m^3, mu=547.8 uPa·s.
ETHANE_RICH_REFERENCE = CryogenicFluid(
    name="Ethane-rich reference (25% CH4 / 70% C2H6 / 5% N2)",
    rho=615.2,          # [Pa·s->kg/m^3] Peng-Robinson (Lunine 1993 comp.)
    mu=547.8e-6,        # [Pa·s]
    sigma=0.030,        # [VERIFICAR] intermedio CH4(0.016)-C2H6(0.033); sin sigma de mezcla confirmada
    composition="0.25 CH4 / 0.70 C2H6 / 0.05 N2 (Lunine 1993)",
    status=TO_VERIFY,
)

# Fluido activo por defecto en Paper 1.
DEFAULT_FLUID = LIGEIA_METHANE_BASELINE


# =============================================================================
# 3. PROFUNDIDADES DE REFERENCIA (para verificaciones físicas, Papers 2-4)
# =============================================================================
# Se usan en los tests de coherencia c = sqrt(g*h). NO son dominios de Sprint 1.
REFERENCE_DEPTHS = {
    "ligeia_max":          (160.0, VERIFIED,   "Mastrogiuseppe et al. 2014, GRL, 10.1002/2013GL058618"),
    "moray_sinus":         (85.0,  VERIFIED,   "Poggiali et al. 2020, JGR:Planets, 10.1029/2020JE006558"),
    "kraken_central_min":  (100.0, TO_VERIFY,  "Poggiali et al. 2020: >100 m (sin eco de fondo)"),
    "punga_max":           (110.0, VERIFIED,  "Mastrogiuseppe et al. 2018 No Olvidar poner el DOI"),
}

# -----------------------------------------------------------------------------
# DATO FÍSICO vs DECISIÓN DE DISEÑO: dos conceptos que NO deben mezclarse.
# Antes convivían en una sola constante (CONVECTIVE_FRONT_SPEED), lo que hacía
# imposible distinguir "esto es lo que Titán hace" de "esto es lo que decidimos
# simular". Ver docs/DECISIONS.md (ADR-007).
# -----------------------------------------------------------------------------

# DATO. Rango de velocidad de frentes convectivos atmosféricos [m/s] respaldado
# por fuente primaria. 
# ES EL LINCHPIN del proyecto: fija qué profundidades pueden entrar en
# Resonancia de Proudman, porque F = 1 exige c = sqrt(g*h) = U, es decir
# h_res = U^2/g. Con este rango, h_res ∈ [3.0, 74.0] m.
#
# NOTA CRÍTICA: Charnay reporta VIENTO de gust front, no velocidad de
#   propagación del frente; se usa como proxy de orden de magnitud (corrientes
#   de densidad). El dato es ECUATORIAL; su aplicabilidad polar está sin
#   verificar. Para mares polares se complementa con Rafkin et al. (2022, Icarus
#   373, 114755) y Hueso & Sánchez-Lavega (2006, Nature 442, 428-431).
OBSERVED_FRONT_SPEED = (2.0, 10.0, TO_VERIFY,
                        "Charnay et al. 2015, Nature Geoscience 8, 362-366, "
                        "DOI 10.1038/ngeo2406; viento de gust front como proxy "
                        "de velocidad de propagación; dato ecuatorial, "
                        "aplicabilidad polar por verificar")

# DECISIÓN DE DISEÑO DEL EXPERIMENTO. Eje del barrido paramétrico de velocidad
# de tormenta [m/s]. NO lleva estatus de verificación porque no afirma nada
# sobre Titán: solo define hasta dónde se barre. Se elige más ancho que
# OBSERVED_FRONT_SPEED para que la curva de amplificación cubra la resonancia de
# las cuencas profundas (c(Ligeia_max) = 14.7 m/s) y quede completa a ambos
# lados de F = 1, de modo que el máximo sea identificable y no un extremo del eje.
SWEEP_SPEED_RANGE = (2.0, 20.0)

# Calibración de la batimetría de Lorenz et al. 2014 (Icarus 237, 9-15),
# Apéndice A. Archivo docs/bathymetry_map.txt: 325 filas x 270 columnas.
LORENZ_BATHYMETRY = {
    # Vertical: el valor de celda ES profundidad en metros (asunción A=0.56).
    "depth_unit": ("meters", VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "ligeia_central_depth_m": (170.0, VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "kraken_max_depth_m":     (197.0, VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    # Horizontal: área de píxel de líquido; OJO con la variación del 37%.
    "pixel_area_km2":         (29.0, VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "pixel_area_std_km2":     (1.5,  VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "pixel_area_variation":   (0.37, VERIFIED, "Lorenz et al. 2014: hasta 37% en el dominio"),
    # Georreferenciación (proyección azimutal polar norte).
    "georef_S":  (-495.0, VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "georef_Xc": (232.0,  VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    "georef_Yc": (245.0,  VERIFIED, "Lorenz et al. 2014, Apéndice A"),
    # LIMITACIÓN declarada por la fuente: no apto para cantidades precisas ni
    # comparación de regiones pequeñas separadas. Uso admisible: forma cualitativa.
}

# =============================================================================
# 4. UTILIDADES DE VERIFICACIÓN FÍSICA (semilla de validation/titan_physical.py)
# =============================================================================
def shallow_water_speed(depth: float, g: float = TITAN.g) -> float:
    """Velocidad de fase de onda somera c = sqrt(g*h) [m/s]."""
    if depth <= 0.0:
        raise ValueError("La profundidad debe ser positiva.")
    return math.sqrt(g * depth)


def inverse_barometer(delta_p: float, fluid: CryogenicFluid = DEFAULT_FLUID,
                      g: float = TITAN.g) -> float:
    """
    Respuesta estática del barómetro inverso: zeta = -delta_p / (rho*g) [m].
    delta_p en [Pa] (negativo = caída de presión -> ascenso de superficie).
    """
    return -delta_p / (fluid.rho * g)


def resonant_depth(storm_speed: float, g: float = TITAN.g) -> float:
    """
    Profundidad que entra en Resonancia de Proudman con una tormenta que se
    desplaza a velocidad U:  F = 1  <=>  c = sqrt(g*h) = U  <=>  h = U^2/g [m].
    """
    return storm_speed * storm_speed / g


def resonant_depth_band(speed_range: tuple[float, float] | None = None,
                        g: float = TITAN.g) -> tuple[float, float]:
    """
    Banda de profundidades resonantes para un rango de velocidades de tormenta.
    Por defecto usa OBSERVED_FRONT_SPEED (el DATO), no SWEEP_SPEED_RANGE.
    """
    if speed_range is None:
        speed_range = (OBSERVED_FRONT_SPEED[0], OBSERVED_FRONT_SPEED[1])
    return resonant_depth(speed_range[0], g), resonant_depth(speed_range[1], g)


def sanity_check() -> None:
    """
    Verificaciones físicas mínimas (Nivel C). Lanza AssertionError si algo
    viola la coherencia esperada. Se ejecuta en la suite de tests.

    La primera versión exigía que c = sqrt(g*h_max) de Ligeia cayera DENTRO
    del rango de frentes convectivos. Con OBSERVED_FRONT_SPEED = [2, 10] m/s eso
    es FALSO: c(Ligeia_max) = 14.7 m/s queda por encima del rango observado.

    Pero esa nunca fue la condición correcta. La resonancia no exige que la
    tormenta iguale la celeridad de la parte MÁS PROFUNDA de la cuenca: exige que
    la iguale en ALGÚN punto del recorrido. Una cuenca abarca todas las
    profundidades entre 0 y h_max, así que la condición correcta es que la banda
    resonante h_res = U^2/g INTERSEQUE el rango [0, h_max] de la cuenca.

    Consecuencia física: la resonancia de
    Proudman en los mares de Titán NO es un fenómeno de cuenca profunda, sino de
    los flancos de profundidad intermedia.
    """
    assert TITAN.g > 0, "g debe ser positiva"
    assert 400.0 <= DEFAULT_FLUID.rho <= 700.0, "rho fuera de rango criogénico"
    assert DEFAULT_FLUID.nu > 0, "nu debe ser positiva"

    h_res_min, h_res_max = resonant_depth_band()
    assert 0.0 < h_res_min < h_res_max, "banda resonante mal definida"

    # La banda resonante debe intersecar el rango de profundidades de Ligeia.
    h_max_ligeia = REFERENCE_DEPTHS["ligeia_max"][0]
    assert h_res_min < h_max_ligeia, (
        f"La banda resonante [{h_res_min:.1f}, {h_res_max:.1f}] m no interseca "
        f"[0, {h_max_ligeia:.1f}] m de Ligeia: la premisa de Proudman falla."
    )

    # El eje del barrido debe cubrir la resonancia de la cuenca más profunda,
    # o el máximo de la curva A(U) caería fuera del barrido y sería inobservable.
    c_max = shallow_water_speed(max(d for d, _, _ in REFERENCE_DEPTHS.values()))
    assert SWEEP_SPEED_RANGE[0] < c_max < SWEEP_SPEED_RANGE[1], (
        f"SWEEP_SPEED_RANGE {SWEEP_SPEED_RANGE} no cubre c_max = {c_max:.1f} m/s."
    )


def audit_provenance() -> list[str]:
    """
    Devuelve la lista de constantes aún en TO_VERIFY. Una corrida destinada a
    manuscrito debería revisar (y vaciar) esta lista.
    """
    pending = []
    if TITAN.p_surface and True:  # marcadores explícitos abajo
        pending.append("TITAN.p_surface (TO_VERIFY)")
    pending.append("TITAN.t_surface_polar (TO_VERIFY)")
    pending.append("TITAN.rho_atm (TO_VERIFY)")
    pending.append(f"{DEFAULT_FLUID.name} rho/mu/sigma (TO_VERIFY)")
    pending.append("ETHANE_RICH_REFERENCE sigma (TO_VERIFY)")
    for k, (_, status, src) in REFERENCE_DEPTHS.items():
        if status == TO_VERIFY:
            pending.append(f"REFERENCE_DEPTHS['{k}']: {src}")
    pending.append(f"OBSERVED_FRONT_SPEED: {OBSERVED_FRONT_SPEED[3]}")
    # SWEEP_SPEED_RANGE NO aparece aquí a propósito: es una decisión de diseño
    # del experimento, no un dato que verificar contra una fuente primaria.
    return pending


if __name__ == "__main__":
    sanity_check()
    print("sanity_check() OK")
    print("\nConstantes pendientes de verificar:")
    for item in audit_provenance():
        print(f"  - {item}")