# PROVENANCE — Paper 1 (snapshot v2)

Procedencia de **cada** dato físico usado en este snapshot. Ningún número
físico se escribe fuera de `src/constants/titan_params.py`; este documento es su
trazabilidad.

Estados: `VERIFIED` = confirmado contra fuente primaria revisada por pares.
`TO_VERIFY` = valor de arranque razonable, **no publicable** sin cerrar la fuente.

---

## 1. Constantes fundamentales de Titán

| Constante | Valor | Unidad | Estado | Fuente |
|---|---:|---|---|---|
| `TITAN.g` | 1.352 | m s⁻² | VERIFIED | Gravedad superficial estándar (efemérides JPL) |
| `TITAN.radius` | 2.5747×10⁶ | m | VERIFIED | Radio medio |
| `TITAN.rotation_period` | 15.945 d | s | VERIFIED | Rotación síncrona |
| `TITAN.p_surface` | 1.467×10⁵ | Pa | VERIFIED | HASI/Huygens |
| `TITAN.t_surface_polar` | 90.0 | K | VERIFIED | DOI: 10.3847/2041-8205/816/1/L17 |
| `TITAN.rho_atm` | 5.3 | kg m⁻³ | VERIFIED | HASI/Huygens |

`omega` y el parámetro de Coriolis `f = 2Ω sin(lat)` se **derivan**, no se fijan.

## 2. Fluido criogénico

Fluido activo del Paper 1: `LIGEIA_METHANE_BASELINE`, estado **TO_VERIFY**.

| Propiedad | Valor | Unidad | Nota |
|---|---:|---|---|
| `rho` | 450.0 | kg m⁻³ | Metano puro ~90 K. El N₂ disuelto la eleva hacia ~500–600; `eos_titan.py` deberá refinarlo |
| `mu` | 2.0×10⁻⁴ | Pa s | Metano puro ~92 K (TITANPOOL; Steckloff et al.) |
| `sigma` | 0.016 | N m⁻¹ | No interviene en el núcleo NLSWE (modelo de onda larga) |

Extremo del barrido de sensibilidad: `ETHANE_RICH_REFERENCE`
(25 % CH₄ / 70 % C₂H₆ / 5 % N₂, Lunine 1993), `rho` = 615.2 kg m⁻³ y
`mu` = 547.8 µPa s por Peng-Robinson a 94 K y 0.15 MPa; su `sigma` = 0.030 N m⁻¹
(intermedio interpolado, sin σ de mezcla confirmada).

**Impacto de `rho` en los resultados:** `rho` entra únicamente a través de la
respuesta de barómetro inverso `ζ_IB = −ΔP/(ρg)`. La **amplificación A**, que es
el resultado del Paper 1, está normalizada por `ζ_IB` y por tanto es
**independiente de `rho`**. Solo las elevaciones absolutas en metros escalan con
1/ρ. Es la razón por la que un `rho` en TO_VERIFY no invalida la conclusión
metodológica.

## 3. Profundidades de referencia

| Cuenca | h_max [m] | Estado | Fuente |
|---|---:|---|---|
| Ligeia Mare | 160 | VERIFIED | Mastrogiuseppe et al. 2014, GRL, DOI 10.1002/2013GL058618 |
| Moray Sinus | 85 | VERIFIED | Poggiali et al. 2020, JGR:Planets, DOI 10.1029/2020JE006558 |
| Kraken (centro) | 100 | **TO_VERIFY** | Poggiali et al. 2020: «> 100 m» (sin eco de fondo). El valor es una **cota inferior**, no una medida |
| Punga Mare | 110 | VERIFIED | Mastrogiuseppe et al. 2018, EPSL, DOI 10.1016/j.epsl.2018.05.033 |

## 4. Velocidad de frentes convectivos — EL LINCHPIN

Separada en Sprint 2 en dos conceptos que antes estaban mezclados (ADR-007):

**`OBSERVED_FRONT_SPEED = (2.0, 10.0)` m s⁻¹ — DATO. Estado: TO_VERIFY.**

Charnay reporta **viento de gust front**, no velocidad de propagación del frente. Se usa como proxy de orden de magnitud (corrientes de densidad). El dato es **ecuatorial**. Su aplicabilidad a los mares polares del norte no está establecida, sin embargo es razonable y apoyado directamente por los resultados de Rafkin et al. 2022 (*Icarus* 373, 114755) y Hueso & Sánchez-Lavega 2006 (*Nature* 442, 428–431).

**`SWEEP_SPEED_RANGE = (2.0, 20.0)` m s⁻¹ — DECISIÓN DE DISEÑO. Sin estado.**

No afirma nada sobre Titán: define el eje del barrido. Se elige más ancho que el
rango observado para que la curva `A(U)` quede completa a ambos lados de `F = 1`
y el máximo sea identificable en vez de caer en un extremo del eje.

**Consecuencia (H-002):** la banda de profundidad resonante es
`h_res = U²/g ∈ [3.0, 74.0] m`, que queda **por debajo** de la profundidad máxima
de las cuatro cuencas. La resonancia es alcanzable en sus flancos intermedios,
no en su centro profundo. **Esta conclusión depende enteramente de un valor
TO_VERIFY y no es publicable hasta cerrarlo.**

## 5. Parámetros que NO son datos físicos

Estos se **barren**, no se citan. Viven en `src/forcing/`, no en
`titan_params.py`, precisamente porque no existe medida para Titán (ADR-006):

| Parámetro | Valor usado | Dónde |
|---|---:|---|
| Fricción lineal `r` | 2×10⁻⁴ s⁻¹ | Solo en el test de barómetro inverso, para amortiguar el seiche. En el equilibrio `u = 0`, así que **no sesga** el valor de equilibrio |
| Manning `n` | — | No usado en Paper 1 |
| Arrastre de viento `C_d` | 1.5×10⁻³ | No usado en Paper 1 (sin forzamiento de viento) |

Igualmente son decisiones, no datos: CFL = 0.45, ΔP = 100 Pa, σ = 10 km,
recorrido D = 400 km, resoluciones de malla, tolerancias de los tests. Todas
están declaradas en el docstring del test o del experimento correspondiente.

## 6. Datos observacionales redistribuidos

**Ninguno.** El Paper 1 trabaja exclusivamente sobre **geometría idealizada**
(canales y cuencas rectangulares). Los únicos números observacionales usados son
escalares publicados (profundidades máximas, velocidades de frente, gravedad),
citados arriba. No se redistribuye ningún producto de datos de PDS ni de
ninguna otra misión. 

## 7. Auditoría automática

`src/constants/titan_params.py::audit_provenance()` devuelve la lista viva de
constantes en TO_VERIFY. Está embebida en `expected_outputs/numbers.json` bajo
`meta.to_verify`, y un test (`test_los_numeros_publicados_declaran_su_procedencia_pendiente`)
falla si desaparece. Estado en el momento de congelar este snapshot: **8
entradas pendientes**.
