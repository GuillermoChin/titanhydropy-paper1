# PROVENANCE — Paper 1

Procedencia de **cada** dato físico usado en este snapshot. Ningún número
físico se escribe fuera de `src/constants/titan_params.py`; este documento es su
trazabilidad.

Estados: `VERIFIED` = confirmado contra fuente primaria revisada por pares, con
cita completa y DOI. `TO_VERIFY` = valor de arranque razonable, **no publicable**
sin cerrar la fuente.

> **Este documento no es prosa libre.** `tests/test_provenance_coherence.py`
> comprueba que el estatus que aparece aquí para cada constante es **exactamente**
> el que declara `titan_params.py`, y que ninguna entrada `VERIFIED` carece de
> DOI. Si diverge, la suite falla.

---

## 1. Constantes fundamentales de Titán

| Constante | Valor | Unidad | Estado | Fuente |
|---|---:|---|---|---|
| `g` | 1.352 | m s⁻² | `VERIFIED` | Gravedad superficial estándar (efemérides JPL) |
| `radius` | 2.5747×10⁶ | m | `VERIFIED` | Radio medio (efemérides JPL) |
| `rotation_period` | 15.945 d | s | `VERIFIED` | Rotación síncrona |
| `p_surface` | 1.467×10⁵ | Pa | `VERIFIED` | Fulchignoni et al. 2005 (ver abajo) |
| `t_surface_polar` | 90.0 | K | `TO_VERIFY` | Valor adoptado; la fuente da un rango más alto (ver abajo) |
| `rho_atm` | 5.3 | kg m⁻³ | `TO_VERIFY` | Cantidad derivada, no medida (ver abajo) |

`omega` y el parámetro de Coriolis `f = 2Ω sin(lat)` se **derivan**, no se fijan.

### `p_surface` — cerrada

> Fulchignoni, M. et al., 2005. In situ measurements of the physical
> characteristics of Titan's environment. *Nature* **438**, 785–791,
> DOI 10.1038/nature04314.

HASI/Huygens midió **1467 ± 1 hPa** en superficie, es decir `1.467×10⁵ Pa`:
coincidencia **exacta** con el valor adoptado, no un redondeo. El documento base
del proyecto la redondeaba a 1.5 bar; aquí se usa el valor medido.

### `t_surface_polar` — sigue abierta, y la cita NO respalda el valor

> Jennings, D. E. et al., 2016. Surface temperatures on Titan during northern
> winter and spring. *ApJL* **816**, L17, DOI 10.3847/2041-8205/816/1/L17.

Esa fuente mide en el **polo norte** una temperatura superficial que sube de
**90.7 ± 0.5 K** a **91.5 ± 0.2 K** a lo largo de la misión. El valor del código,
**90.0 K, queda por debajo de ese rango**: es un valor **adoptado**, no medido, y
la cita no lo respalda. Se mantiene `TO_VERIFY` hasta que se adopte el valor
medido o se justifique explícitamente la diferencia.

No debe usarse el valor **ecuatorial** (93.65 ± 0.25 K, Fulchignoni et al. 2005):
de `T` dependen `rho` y `mu` del líquido.

### `rho_atm` — sigue abierta, y además es derivada

HASI **no mide** densidad: la **infiere** de `P` y `T` bajo equilibrio
hidrostático y ley de gases reales. No se ha localizado fuente primaria que
publique `5.3 kg m⁻³` con esos dígitos, y el gas ideal con la `P` y la `T` ya
verificadas da ≈ 5.2 kg m⁻³. Se mantiene `TO_VERIFY`.

**Exposición nula en el Paper 1**: `rho_atm` solo interviene en el esfuerzo de
viento, y el Paper 1 no usa forzamiento de viento.

## 2. Fluido criogénico

Fluido activo del Paper 1: `LIGEIA_METHANE_BASELINE`, estado **`TO_VERIFY`** en
sus tres propiedades (`rho`, `mu`, `sigma`).

| Propiedad | Valor | Unidad | Nota |
|---|---:|---|---|
| `rho` | 450.0 | kg m⁻³ | Metano puro ~90 K. El N₂ disuelto la eleva hacia ~500–600; `eos_titan.py` deberá refinarlo |
| `mu` | 2.0×10⁻⁴ | Pa s | Metano puro ~92 K (TITANPOOL; Steckloff et al.) |
| `sigma` | 0.016 | N m⁻¹ | No interviene en el núcleo NLSWE (modelo de onda larga) |

Extremo del barrido de sensibilidad: `ETHANE_RICH_REFERENCE`
(25 % CH₄ / 70 % C₂H₆ / 5 % N₂, Lunine 1993), `rho` = 615.2 kg m⁻³ y
`mu` = 547.8 µPa s por Peng-Robinson a 94 K y 0.15 MPa.

Su `sigma` = 0.030 N m⁻¹ está **`TO_VERIFY`**: es un valor **interpolado** entre
el del metano (0.016) y el del etano (0.033), sin σ de mezcla confirmada contra
fuente primaria. El estado y el motivo van juntos a propósito: declarar el motivo
y omitir el estado deja al lector sin saber si el valor es publicable.

**Impacto de `rho` en los resultados:** `rho` entra únicamente a través de la
respuesta de barómetro inverso `ζ_IB = −ΔP/(ρg)`. La **amplificación A**, que es
el resultado del Paper 1, está normalizada por `ζ_IB` y por tanto es
**independiente de `rho`**. Solo las elevaciones absolutas en metros escalan con
1/ρ. Es la razón por la que un `rho` en `TO_VERIFY` no invalida la conclusión
metodológica.

## 3. Profundidades de referencia

| Cuenca | h_max [m] | Estado | Fuente |
|---|---:|---|---|
| Ligeia Mare | 160 | `VERIFIED` | Mastrogiuseppe et al. 2014, *GRL*, DOI 10.1002/2013GL058618 |
| Moray Sinus | 85 | `VERIFIED` | Poggiali et al. 2020, *JGR: Planets*, DOI 10.1029/2020JE006558 |
| Kraken (centro) | 100 | `TO_VERIFY` | **Cota inferior, no medición** (ver abajo) |
| Punga Mare | 110 | `VERIFIED` | Mastrogiuseppe et al. 2018 (ver abajo) |

### Punga Mare — cerrada

> Mastrogiuseppe, M., Poggiali, V., Hayes, A. G., Lunine, J. I., Seu, R.,
> Di Achille, G., Lorenz, R. D., 2018. Cassini radar observation of Punga Mare
> and environs: bathymetry and composition. *Earth Planet. Sci. Lett.* **496**,
> 89–95, DOI 10.1016/j.epsl.2018.05.033.

Profundidad máxima **medida** de 110 m a lo largo de la traza de altimetría del
sobrevuelo T108.

### Kraken central — es una COTA INFERIOR, no una medida

Poggiali et al. 2020 reporta «> 100 m» porque **no recibió eco de fondo**: el
radar no alcanzó el lecho. La profundidad real es **mayor** que 100 m y sigue sin
medirse. Usar esta cifra como si fuera una medición sobreestima la certeza y
**subestima** la profundidad, lo que a su vez desplaza cualquier conclusión sobre
qué tramos de la cuenca caen en la banda resonante.

## 4. Velocidad de frentes convectivos — EL LINCHPIN

Aquí conviven dos cantidades con la misma unidad y significados **incompatibles**,
y se declaran por separado a propósito: una afirma algo sobre Titán y la otra solo
define el alcance de un experimento. Mezclarlas haría imposible distinguir «esto es
lo que Titán hace» de «esto es lo que decidimos simular».

**`OBSERVED_FRONT_SPEED = (2.0, 10.0)` m s⁻¹ — DATO. Estado: `TO_VERIFY`.**

> Charnay, B. et al., 2015. Methane storms as a driver of Titan's dune
> orientation. *Nature Geoscience* **8**, 362–366, DOI 10.1038/ngeo2406.

Tres reservas que deben resolverse antes de publicar:

1. Charnay reporta **viento de gust front**, no velocidad de propagación del
   frente. Se usa como proxy de orden de magnitud (corrientes de densidad).
2. El dato es **ecuatorial**. Su aplicabilidad a los mares polares del norte no
   está establecida.
3. Fuentes **a incorporar** para el régimen polar: Rafkin et al. 2022
   (*Icarus* 373, 114755) y Hueso & Sánchez-Lavega 2006 (*Nature* 442, 428–431).

**`SWEEP_SPEED_RANGE = (2.0, 20.0)` m s⁻¹ — DECISIÓN DE DISEÑO. Sin estado.**

No afirma nada sobre Titán: define el eje del barrido. Se elige más ancho que el
rango observado para que la curva `A(U)` quede completa a ambos lados de `F = 1`
y el máximo sea identificable en vez de caer en un extremo del eje.

**Consecuencia:** la banda de profundidad resonante es
`h_res = U²/g ∈ [3.0, 74.0] m`, que queda **por debajo** de la profundidad máxima
de las cuatro cuencas. La resonancia es alcanzable en sus flancos intermedios,
no en su centro profundo. **Esta conclusión depende enteramente de un valor
`TO_VERIFY` y no es publicable hasta cerrarlo.** La sensibilidad es **cuadrática**
en `U`.

## 5. Parámetros que NO son datos físicos

Estos se **barren**, no se citan. Viven en `src/forcing/`, no en
`titan_params.py`, precisamente porque no existe medida para Titán: un parámetro
sin fuente no puede convivir con las constantes que sí la tienen, o el lector no
sabría cuáles descansan en una medida y cuáles en una elección.

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
ninguna otra misión. La ingesta de datos altimétricos reales corresponde al
Paper 2.

## 7. Auditoría automática

`src/constants/titan_params.py::audit_provenance()` devuelve la lista viva de
constantes en `TO_VERIFY`, **iterando** sobre `TITAN_PROVENANCE` y
`REFERENCE_DEPTHS` en vez de llevarla escrita a mano. Está embebida en
`expected_outputs/numbers.json` bajo `meta.to_verify`, y un test falla si
desaparece.

Estado al congelar este snapshot: **6 entradas pendientes**.

1. `TITAN.t_surface_polar`
2. `TITAN.rho_atm`
3. `LIGEIA_METHANE_BASELINE` (`rho`/`mu`/`sigma`)
4. `ETHANE_RICH_REFERENCE` (`sigma`)
5. `REFERENCE_DEPTHS['kraken_central_min']`
6. `OBSERVED_FRONT_SPEED`

De las seis, **solo la última es peligrosa** para las conclusiones: de ella
cuelga por completo la localización de la banda de profundidad resonante, con
sensibilidad cuadrática. Las dos profundidades son
secundarias, y `t_surface_polar`, `rho_atm` y las propiedades del fluido tienen
exposición nula o ya declarada sobre el resultado del Paper 1, que es una
amplificación adimensional.

## 8. Garantía de coherencia

El estatus de verificación de cada constante es **un solo dato**, no una
afirmación repetida en varios sitios. Vive en `TITAN_PROVENANCE` y
`REFERENCE_DEPTHS` dentro de `src/constants/titan_params.py`;
`audit_provenance()` **itera** sobre ellos en vez de llevar una lista escrita a
mano, y la lista resultante viaja embebida en `expected_outputs/numbers.json`
bajo `meta.to_verify`.

`tests/test_provenance_coherence.py` comprueba que este documento declara para
cada constante **exactamente** el estatus que declara el código, que ninguna
entrada `VERIFIED` carece de cita con DOI, y que el recuento de pendientes de la
§7 es el real. Si alguno de los tres se aparta, la suite falla.

Un documento de procedencia que pueda contradecir al código que describe no
documenta nada: por eso la coherencia se comprueba, no se confía.
