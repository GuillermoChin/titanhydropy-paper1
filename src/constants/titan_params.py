"""
constants/titan_params.py
==========================
Fuente ÚNICA de verdad para las constantes físicas de Titán en TitanHydroPy.

POLÍTICA DE ESTE MÓDULO (blindaje anti-alucinación):
  1. Ningún valor se usa en otro módulo sin importarlo desde aquí.
  2. Cada constante lleva: valor, unidad SI, fuente y estatus de verificación.
  3. status = VERIFIED  -> confirmado contra fuente primaria revisada por pares.
     status = TO_VERIFY -> valor de arranque razonable; NO usar en manuscrito
                           sin confirmar la fuente primaria y el número exacto.
  4. La función audit_provenance() lista todo lo que sigue en TO_VERIFY para
     que una corrida destinada a publicación no arrastre valores sin verificar.

DECISIÓN DE DISEÑO (variables primitivas):
  El estado público del solver se expresa en variables PRIMITIVAS (zeta, u, v),
  no en conservadas (h, hu, hv), para que la frontera pública no dependa del
  motor que haya debajo. Aquí solo se declara para que las constantes se
  interpreten en ese marco:
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

    # Presión atmosférica superficial medida in situ por HASI/Huygens.
    # 1467 +/- 1 hPa = 1.467e5 Pa: coincide EXACTAMENTE con el valor adoptado.
    # El documento base la redondeaba a 1.5 bar; aquí se usa el valor medido.
    p_surface: float = 1.467e5       # [Pa]     VERIFIED (ver TITAN_PROVENANCE)

    # Temperatura superficial en los mares polares del norte.
    # NO usar el valor ecuatorial (93.65 +/- 0.25 K, Fulchignoni et al. 2005):
    # de T dependen rho y mu del líquido.
    #
    # ATENCION: 90.0 K es un valor ADOPTADO, no medido, y queda POR DEBAJO del
    # rango que da la fuente primaria disponible (Jennings et al. 2016 mide
    # 90.7 +/- 0.5 a 91.5 +/- 0.2 K en el polo norte). Por eso sigue TO_VERIFY.
    t_surface_polar: float = 90.0    # [K]      TO_VERIFY (ver TITAN_PROVENANCE)

    # Densidad atmosférica superficial (solo interviene en el esfuerzo de viento;
    # el Paper 1 no usa forzamiento de viento).
    # Es una cantidad DERIVADA, no medida: HASI infiere la densidad de P y T bajo
    # equilibrio hidrostático y ley de gases reales. Sigue TO_VERIFY.
    rho_atm: float = 5.3             # [kg/m^3] TO_VERIFY (ver TITAN_PROVENANCE)

    @property
    def omega(self) -> float:
        """Velocidad angular de rotación [rad/s]."""
        return 2.0 * math.pi / self.rotation_period

    def coriolis(self, lat_deg: float) -> float:
        """Parámetro de Coriolis f = 2*Omega*sin(lat) [1/s]."""
        return 2.0 * self.omega * math.sin(math.radians(lat_deg))


TITAN = TitanConstants()  # instancia global de solo lectura


# Procedencia LEGIBLE POR MÁQUINA de cada constante de TitanConstants.
#
# Existe para que el estatus de verificación sea UN SOLO DATO y no varias
# copias que puedan desincronizarse: el comentario de cada constante,
# `audit_provenance()` y `PROVENANCE.md` tienen que decir lo mismo. La
# auditoría ITERA sobre este diccionario en vez de llevar una lista escrita a
# mano, y un test comprueba que el documento dice exactamente lo mismo.
#
# Regla: ninguna entrada VERIFIED sin cita completa con DOI.
TITAN_PROVENANCE: dict[str, tuple[str, str]] = {
    "g": (VERIFIED,
          "Gravedad superficial estándar (efemérides JPL)"),
    "radius": (VERIFIED,
               "Radio medio de Titán (efemérides JPL)"),
    "rotation_period": (VERIFIED,
                        "Periodo de rotación síncrona de Titán, 15.945 d, igual "
                        "a su periodo orbital (efemérides JPL)"),
    "p_surface": (VERIFIED,
                  "Fulchignoni, M. et al., 2005. In situ measurements of the "
                  "physical characteristics of Titan's environment. Nature 438, "
                  "785-791, DOI 10.1038/nature04314. HASI/Huygens mide "
                  "1467 +/- 1 hPa en superficie = 1.467e5 Pa: coincidencia "
                  "exacta con el valor adoptado"),
    "t_surface_polar": (TO_VERIFY,
                        "Valor ADOPTADO (90.0 K), no medido. La fuente primaria "
                        "disponible, Jennings, D. E. et al., 2016. Surface "
                        "temperatures on Titan during northern winter and "
                        "spring. ApJL 816, L17, DOI 10.3847/2041-8205/816/1/L17, "
                        "mide en el polo norte 90.7 +/- 0.5 K subiendo a "
                        "91.5 +/- 0.2 K. El valor del código queda POR DEBAJO de "
                        "ese rango, asi que la cita NO lo respalda: o se adopta "
                        "el valor medido o se justifica la diferencia"),
    "rho_atm": (TO_VERIFY,
                "Cantidad DERIVADA, no medida: HASI infiere la densidad de P y T "
                "bajo equilibrio hidrostático y ley de gases reales "
                "(Fulchignoni et al. 2005). No se ha localizado fuente primaria "
                "que publique 5.3 kg/m^3 con esos dígitos; el gas ideal con la "
                "P y T verificadas da ~5.2 kg/m^3. Solo interviene en el "
                "esfuerzo de viento, que el Paper 1 no usa"),
}


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
# Se usan en los tests de coherencia c = sqrt(g*h).
REFERENCE_DEPTHS = {
    "ligeia_max":          (160.0, VERIFIED,   "Mastrogiuseppe et al. 2014, GRL, 10.1002/2013GL058618"),
    "moray_sinus":         (85.0,  VERIFIED,   "Poggiali et al. 2020, JGR:Planets, 10.1029/2020JE006558"),
    # COTA INFERIOR, NO MEDICIÓN. El radar no recibió eco de fondo, de modo que
    # la profundidad es MAYOR que la penetración alcanzada; 100 m es el límite
    # por debajo del cual se puede descartar, no la profundidad de la cuenca.
    # Usarla como si fuera una medida sobreestima la certeza y subestima h.
    "kraken_central_min":  (100.0, TO_VERIFY,
                            "COTA INFERIOR (no medición): Poggiali et al. 2020, "
                            "JGR:Planets, DOI 10.1029/2020JE006558, reporta "
                            ">100 m por AUSENCIA DE ECO DE FONDO. La profundidad "
                            "real es mayor y sigue sin medirse"),
    "punga_max":           (110.0, VERIFIED,
                            "Mastrogiuseppe, M., Poggiali, V., Hayes, A. G., "
                            "Lunine, J. I., Seu, R., Di Achille, G., Lorenz, "
                            "R. D., 2018. Cassini radar observation of Punga "
                            "Mare and environs: bathymetry and composition. "
                            "Earth Planet. Sci. Lett. 496, 89-95, "
                            "DOI 10.1016/j.epsl.2018.05.033. Profundidad máxima "
                            "MEDIDA de 110 m a lo largo de la traza de altimetría "
                            "del sobrevuelo T108"),
}

# -----------------------------------------------------------------------------
# DATO FÍSICO vs DECISIÓN DE DISEÑO: dos conceptos que NO deben mezclarse.
# Antes convivían en una sola constante (CONVECTIVE_FRONT_SPEED), lo que hacía
# imposible distinguir "esto es lo que Titán hace" de "esto es lo que decidimos
# simular".
# -----------------------------------------------------------------------------

# DATO. Rango de velocidad de frentes convectivos atmosféricos [m/s] respaldado
# por fuente primaria. Lleva estatus de verificación.
# ES EL LINCHPIN del proyecto: fija qué profundidades pueden entrar en
# Resonancia de Proudman, porque F = 1 exige c = sqrt(g*h) = U, es decir
# h_res = U^2/g. Con este rango, h_res ∈ [3.0, 74.0] m.
#
# NOTA CRÍTICA: Charnay reporta VIENTO de gust front, no velocidad de
#   propagación del frente; se usa como proxy de orden de magnitud (corrientes
#   de densidad). El dato es ECUATORIAL; su aplicabilidad polar está sin
#   verificar. Para mares polares complementar con Rafkin et al. (2022, Icarus
#   373, 114755) y Hueso & Sánchez-Lavega (2006, Nature 442, 428-431).
#   CORRECCIÓN DE CITA (8 de agosto del 2026). Las páginas eran 344-348 y son
#   362-366. La revista (Nature Geoscience 8) y el DOI (10.1038/ngeo2406) ya
#   eran correctos. Se corrige la CITA, no la verificación: el estatus sigue
#   siendo TO_VERIFY, porque lo que está sin verificar no es de dónde sale el
#   número sino su aplicabilidad — viento de gust front ECUATORIAL usado como
#   proxy de velocidad de propagación POLAR.
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
# Apéndice A: mapa de 325 filas x 270 columnas.
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

    EL LINCHPIN, EN SU FORMA CORRECTA
    ---------------------------------
    Una condición más estricta exigiría que c = sqrt(g*h_max) de Ligeia cayera
    DENTRO del rango de frentes convectivos. Con OBSERVED_FRONT_SPEED =
    [2, 10] m/s eso es FALSO: c(Ligeia_max) = 14.7 m/s queda por encima del rango observado.

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
    # Las constantes de Titán se ITERAN desde TITAN_PROVENANCE en vez de
    # escribirse a mano aquí: una lista a mano deja de coincidir con el estatus
    # real en cuanto alguien cambia uno sin tocar el otro.
    for nombre, (status, src) in TITAN_PROVENANCE.items():
        if status == TO_VERIFY:
            pending.append(f"TITAN.{nombre}: {src}")
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