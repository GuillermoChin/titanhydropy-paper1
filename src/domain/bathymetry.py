"""
domain/bathymetry.py
====================
Reconstruccion de batimetria 2D de Ligeia Mare a partir de un perfil
altimetrico 1D (traza T91 de Cassini RADAR) y del contorno de costa.

===========================================================================
                    ADVERTENCIA CIENTIFICA - LEER ANTES DE USAR
===========================================================================
NO EXISTE batimetria 2D medida de Ligeia Mare. Lo que existe es:
  * un PERFIL 1D a lo largo de la traza T91 (Mastrogiuseppe et al. 2014,
    GRL, DOI 10.1002/2013GL058618), con profundidad maxima ~160 m;
  * el CONTORNO DE COSTA, cartografiado por radar.

Pasar de eso a un campo h0(x,y) EXIGE un modelo de interpolacion, es decir,
supuestos que no estan en los datos. Este modulo hace esos supuestos
EXPLICITOS y los deja intercambiables, en vez de esconderlos en una funcion.

Consecuencia para el Paper 2: los resultados sobre esta geometria son una
DEMOSTRACION DE METODO sobre una batimetria reconstruida, no una medida de lo
que ocurre en Ligeia. La sensibilidad del resultado a la estrategia de
reconstruccion es, ella misma, un resultado que hay que reportar.

PROCEDENCIA DE LAS ENTRADAS
---------------------------
Este modulo NO trae datos empotrados. Dos vias:
  1. `load_track_profile(path)` y `load_coastline(path)` leen los datos reales
     digitalizados, en el formato documentado en cada funcion. Es la via para
     resultados publicables.
  2. `idealized_track_profile()` e `idealized_coastline()` construyen una
     geometria IDEALIZADA a partir del UNICO escalar publicado que tenemos
     (h_max = 160 m, via REFERENCE_DEPTHS['ligeia_max']) mas una forma
     declarada. Estan etiquetadas como NO-DATO y su uso queda registrado en los
     metadatos de toda salida.
===========================================================================

LAS TRES ESTRATEGIAS DE RECONSTRUCCION
--------------------------------------
Se implementan las tres y son intercambiables por nombre. Elegida por defecto:
`coast_distance`.

1. `nearest_track` - INTERPOLACION RADIAL DESDE LA TRAZA.
   A cada punto del mar se le asigna la profundidad del punto MAS CERCANO de la
   traza T91, con una atenuacion hacia cero al acercarse a la costa.
   * Supuesto: la cuenca es localmente simetrica respecto de la traza.
   * Ventaja: respeta el dato medido exactamente SOBRE la traza.
   * Limitacion grave: fuera del entorno de la traza no hay informacion y la
     extrapolacion es arbitraria; genera artefactos de "cresta" a lo largo de la
     mediatriz entre segmentos de traza. Es la mas fiel al dato y la menos
     fiable lejos de el.

2. `coast_distance` - PROFUNDIDAD EN FUNCION DE LA DISTANCIA A COSTA.
   h0(x,y) = h_max * Phi( d(x,y) / d_max ), donde d es la distancia al contorno
   de costa y Phi es la forma normalizada CALIBRADA para reproducir el perfil
   medido a lo largo de la traza.
   * Supuesto: la profundidad depende solo de la distancia a la costa; la cuenca
     tiene seccion transversal auto-similar.
   * Ventaja: usa la informacion que SI esta bien observada (la linea de costa,
     cartografiada por radar) para organizar el campo, y honra el perfil medido
     por construccion de Phi. Las franjas de profundidad intermedia siguen la
     geometria real de la costa, que es exactamente lo que hace falta para
     localizar F -> 1.
   * Limitacion: impone simetria respecto de la costa; no puede representar
     canales profundos adosados a un margen ni asimetrias tectonicas.

3. `hypsometric` - CUENCA PARAMETRICA POR HIPSOMETRIA.
   h0 = h_max * (1 - (r/R)^p), con r la distancia radial a un centro y (R, p)
   ajustados para reproducir el perfil medido y el area de la cuenca.
   * Supuesto: la cuenca es un cuerpo de revolucion.
   * Ventaja: totalmente analitica, derivable, sin campos auxiliares; util como
     control de sensibilidad y para tests.
   * Limitacion: ignora por completo la forma de la costa. Para Ligeia, que
     tiene tres brazos muy marcados, es una idealizacion fuerte.

CONVENCION: devuelve h0 (PROFUNDIDAD DE REPOSO, positiva en zona liquida y <= 0
en tierra), de forma (ny, nx), lista para Domain.bathymetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from constants.titan_params import (LORENZ_BATHYMETRY, REFERENCE_DEPTHS, TITAN)

__all__ = ["TrackProfile", "Coastline", "BathymetryResult",
           "load_track_profile", "load_coastline",
           "idealized_track_profile", "idealized_coastline",
           "reconstruct", "STRATEGIES", "distance_to_coast",
           "LorenzMap", "load_lorenz_map", "LIGEIA_LATLON_BOX",
           "lorenz_region_to_bathymetry"]


# ===========================================================================
# 1. Entradas
# ===========================================================================
@dataclass(frozen=True)
class TrackProfile:
    """
    Perfil altimetrico 1D a lo largo de una traza.

    distance : distancia a lo largo de la traza [m], creciente.
    depth    : profundidad medida [m], positiva.
    source   : procedencia textual. Obligatoria.
    is_data  : True solo si proviene de datos reales digitalizados.
    """
    distance: np.ndarray
    depth: np.ndarray
    source: str
    is_data: bool

    def __post_init__(self) -> None:
        if self.distance.shape != self.depth.shape:
            raise ValueError("distance y depth deben tener la misma forma.")
        if np.any(np.diff(self.distance) <= 0):
            raise ValueError("distance debe ser estrictamente creciente.")
        if np.any(self.depth < 0):
            raise ValueError("depth debe ser >= 0 (profundidad, no cota).")

    @property
    def max_depth(self) -> float:
        return float(np.max(self.depth))

    @property
    def length(self) -> float:
        return float(self.distance[-1] - self.distance[0])


@dataclass(frozen=True)
class Coastline:
    """
    Contorno de costa como poligono cerrado en coordenadas del dominio [m].

    x, y    : vertices del poligono. No hace falta repetir el primero al final.
    source  : procedencia textual. Obligatoria.
    is_data : True solo si proviene de cartografia real.
    """
    x: np.ndarray
    y: np.ndarray
    source: str
    is_data: bool

    def __post_init__(self) -> None:
        if self.x.shape != self.y.shape or self.x.size < 3:
            raise ValueError("El poligono necesita al menos 3 vertices.")

    def contains(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Mascara booleana de los puntos interiores al poligono.
        Algoritmo de cruce de rayos (ray casting), vectorizado.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        dentro = np.zeros(X.shape, dtype=bool)
        x1, y1 = self.x, self.y
        x2, y2 = np.roll(self.x, -1), np.roll(self.y, -1)
        for a1, b1, a2, b2 in zip(x1, y1, x2, y2):
            cruza = ((b1 > Y) != (b2 > Y))
            with np.errstate(divide="ignore", invalid="ignore"):
                x_int = (a2 - a1) * (Y - b1) / (b2 - b1) + a1
            dentro ^= cruza & (X < x_int)
        return dentro


def load_track_profile(path: str | Path,
                       source: str | None = None) -> TrackProfile:
    """
    Lee un perfil real digitalizado.

    FORMATO: CSV o texto con dos columnas numericas, `distance_m, depth_m`, con
    cabecera opcional que empiece por '#'. La distancia debe ser creciente.

    Esta es la via para resultados PUBLICABLES: sustituir la geometria
    idealizada por el perfil real es cambiar una linea del script de escenario.
    """
    path = Path(path)
    datos = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if datos.shape[1] < 2:
        raise ValueError(f"{path}: se esperaban 2 columnas "
                         f"(distance_m, depth_m).")
    return TrackProfile(distance=datos[:, 0].astype(float),
                        depth=datos[:, 1].astype(float),
                        source=source or f"fichero: {path}",
                        is_data=True)


def load_coastline(path: str | Path, source: str | None = None) -> Coastline:
    """
    Lee un contorno de costa real.

    FORMATO: CSV o texto con dos columnas numericas, `x_m, y_m`, un vertice por
    linea, en orden a lo largo del contorno. Cabecera opcional con '#'.
    """
    path = Path(path)
    datos = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if datos.shape[1] < 2:
        raise ValueError(f"{path}: se esperaban 2 columnas (x_m, y_m).")
    return Coastline(x=datos[:, 0].astype(float), y=datos[:, 1].astype(float),
                     source=source or f"fichero: {path}", is_data=True)


# ---------------------------------------------------------------------------
# Geometria IDEALIZADA (NO son datos)
# ---------------------------------------------------------------------------
def idealized_track_profile(n: int = 200, length: float = 3.0e5,
                            max_depth: float | None = None,
                            shape_exponent: float = 2.0) -> TrackProfile:
    """
    Perfil IDEALIZADO. **NO ES UN DATO.**

    Se construye a partir del UNICO escalar publicado que tenemos,
    h_max = REFERENCE_DEPTHS['ligeia_max'] = 160 m (Mastrogiuseppe et al. 2014),
    mas una FORMA DECLARADA: perfil en U suave,

        h(s) = h_max * [1 - |2s/L - 1|^p],   p = shape_exponent

    p = 2 da una parabola (fondo redondeado); p -> 1 da un perfil en V; p grande
    da un fondo plano con taludes abruptos. El valor de p es un PARAMETRO LIBRE
    del estudio de sensibilidad, no una medida.

    `is_data = False`: toda salida que use esto queda marcada como no-dato.
    """
    h_max = max_depth if max_depth is not None \
        else REFERENCE_DEPTHS["ligeia_max"][0]
    s = np.linspace(0.0, length, n)
    t = 2.0 * s / length - 1.0
    h = h_max * (1.0 - np.abs(t) ** shape_exponent)
    return TrackProfile(
        distance=s, depth=np.maximum(h, 0.0),
        source=(f"IDEALIZADO (NO ES DATO): perfil en U con exponente "
                f"p={shape_exponent}, escalado a h_max={h_max} m de "
                f"REFERENCE_DEPTHS['ligeia_max'] "
                f"(Mastrogiuseppe et al. 2014, 10.1002/2013GL058618). "
                f"Sustituir por la traza T91 digitalizada antes de publicar."),
        is_data=False)


def idealized_coastline(n: int = 720, lx: float = 4.7e5, ly: float = 4.2e5,
                        center: tuple[float, float] | None = None
                        ) -> Coastline:
    """
    Contorno IDEALIZADO de tres brazos. **NO ES UN DATO.**

    Ligeia Mare tiene tres brazos principales; aqui se representa esa topologia
    con una forma trilobulada analitica

        r(theta) = r0 * [1 + e1*cos(3*theta + f1) + e2*cos(theta + f2)]

    cuyas dimensiones globales (~470 x 420 km) son del orden de las publicadas
    para Ligeia. NO reproduce la linea de costa real: reproduce su TOPOLOGIA
    (cuenca central con brazos alargados y bahias someras), que es lo que el
    escenario necesita para que la reconstruccion por distancia a costa genere
    franjas de profundidad intermedia realistas en su geometria.

    `is_data = False`.
    """
    cx, cy = center if center is not None else (0.5 * lx, 0.5 * ly)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r0 = 0.36 * min(lx, ly)
    r = r0 * (1.0 + 0.34 * np.cos(3.0 * th + 0.6)
              + 0.12 * np.cos(th - 1.1)
              + 0.06 * np.cos(5.0 * th + 2.2))
    return Coastline(
        x=cx + r * np.cos(th), y=cy + r * np.sin(th),
        source=("IDEALIZADO (NO ES DATO): contorno trilobulado analitico de "
                "dimensiones del orden de Ligeia (~470x420 km). Reproduce la "
                "TOPOLOGIA (cuenca central + brazos + bahias), no la linea de "
                "costa cartografiada. Sustituir por el contorno digitalizado "
                "antes de publicar."),
        is_data=False)


# ===========================================================================
# 2. Utilidades geometricas
# ===========================================================================
def distance_to_coast(mask_wet: np.ndarray, dx: float, dy: float,
                      coast: Coastline | None = None,
                      X: np.ndarray | None = None,
                      Y: np.ndarray | None = None) -> np.ndarray:
    """
    Distancia euclidea de cada celda mojada a la costa [m].

    Si se pasa `coast` (y X, Y), se calcula la distancia EXACTA a los vertices
    del poligono, densificados a media celda. Si no, se usa la distancia a la
    celda seca mas proxima de la mascara, que discretiza la costa a la malla.

    Implementacion: fuerza bruta vectorizada sobre los puntos de costa. Es
    O(n_celdas * n_costa); para mallas de ~10^5 celdas y contornos de ~10^3
    puntos son ~10^8 operaciones en bloques, del orden de un segundo. Se evita a
    proposito una dependencia de scipy: el proyecto solo exige NumPy.
    GANCHO: sustituible por scipy.ndimage.distance_transform_edt o un KD-tree si
    el coste llegara a importar.
    """
    d = np.full(mask_wet.shape, np.inf, dtype=float)
    if not np.any(mask_wet):
        return d

    if coast is not None and X is not None and Y is not None:
        px, py = _densificar_poligono(coast, 0.5 * min(dx, dy))
    else:
        # Costa discreta: bordes de la mascara mojada.
        seca = ~mask_wet
        borde = np.zeros_like(seca)
        borde[1:, :] |= seca[:-1, :]
        borde[:-1, :] |= seca[1:, :]
        borde[:, 1:] |= seca[:, :-1]
        borde[:, :-1] |= seca[:, 1:]
        borde &= mask_wet
        if X is None or Y is None:
            ny, nx = mask_wet.shape
            X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx,
                               (np.arange(ny) + 0.5) * dy)
        px, py = X[borde], Y[borde]
        if px.size == 0:
            return d

    xs, ys = X[mask_wet], Y[mask_wet]
    mejor = np.full(xs.shape, np.inf)
    # Bloques para acotar la memoria del producto exterior.
    bloque = max(1, int(4.0e7 / max(xs.size, 1)))
    for i in range(0, px.size, bloque):
        dxs = xs[:, None] - px[None, i:i + bloque]
        dys = ys[:, None] - py[None, i:i + bloque]
        mejor = np.minimum(mejor, np.min(dxs * dxs + dys * dys, axis=1))
    d[mask_wet] = np.sqrt(mejor)
    return d


def _densificar_poligono(coast: Coastline, paso: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Inserta puntos a lo largo de cada arista para que ninguna supere `paso`."""
    xs, ys = [], []
    x2, y2 = np.roll(coast.x, -1), np.roll(coast.y, -1)
    for a1, b1, a2, b2 in zip(coast.x, coast.y, x2, y2):
        n = max(2, int(np.hypot(a2 - a1, b2 - b1) / max(paso, 1e-9)) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        xs.append(a1 + (a2 - a1) * t)
        ys.append(b1 + (b2 - b1) * t)
    return np.concatenate(xs), np.concatenate(ys)


# ===========================================================================
# 3. Resultado
# ===========================================================================
@dataclass
class BathymetryResult:
    """Batimetria reconstruida junto con TODOS los supuestos que la produjeron."""
    depth: np.ndarray                # h0 (ny, nx) [m]; <= 0 en tierra
    x: np.ndarray
    y: np.ndarray
    strategy: str
    assumptions: dict = field(default_factory=dict)
    profile_source: str = ""
    coast_source: str = ""
    is_data_based: bool = False

    @property
    def wet(self) -> np.ndarray:
        return self.depth > 0.0

    @property
    def max_depth(self) -> float:
        return float(np.max(self.depth))

    def area(self, dx: float, dy: float) -> float:
        return float(np.count_nonzero(self.wet) * dx * dy)

    def volume(self, dx: float, dy: float) -> float:
        return float(np.sum(np.maximum(self.depth, 0.0)) * dx * dy)

    def provenance(self) -> dict:
        """Metadatos para embeber en toda salida (titan_io)."""
        return {
            "bathymetry_strategy": self.strategy,
            "bathymetry_assumptions": self.assumptions,
            "bathymetry_profile_source": self.profile_source,
            "bathymetry_coast_source": self.coast_source,
            "bathymetry_is_data_based": self.is_data_based,
            "bathymetry_max_depth_m": self.max_depth,
        }

    def summary(self) -> str:
        marca = "DATOS REALES" if self.is_data_based else "IDEALIZADA (NO-DATO)"
        return (f"Batimetria [{marca}] estrategia='{self.strategy}'  "
                f"h_max={self.max_depth:.1f} m  "
                f"celdas mojadas={np.count_nonzero(self.wet)}")


# ===========================================================================
# 4. Las tres estrategias
# ===========================================================================
def _perfil_normalizado(profile: TrackProfile, n: int = 400
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae la FORMA normalizada Phi del perfil: profundidad relativa en funcion
    de la distancia relativa a la costa mas proxima a lo largo de la traza.

    Se asume que la traza cruza la cuenca de orilla a orilla, de modo que la
    distancia a costa de un punto de la traza es min(s, L-s). Con eso, el perfil
    1D medido se convierte en una curva Phi(d_rel) directamente utilizable por
    la estrategia `coast_distance`.
    """
    s = profile.distance - profile.distance[0]
    d_costa = np.minimum(s, s[-1] - s)
    d_rel = d_costa / max(d_costa.max(), 1e-12)
    h_rel = profile.depth / max(profile.max_depth, 1e-12)
    orden = np.argsort(d_rel)
    d_rel, h_rel = d_rel[orden], h_rel[orden]
    # Promedia los dos flancos en una rejilla comun y fuerza Phi(0) = 0.
    rejilla = np.linspace(0.0, 1.0, n)
    phi = np.interp(rejilla, d_rel, h_rel)
    phi[0] = 0.0
    phi = np.maximum.accumulate(np.clip(phi, 0.0, 1.0))
    return rejilla, phi


def _strategy_coast_distance(X, Y, dx, dy, wet, profile, coast, **kw):
    """h0 = h_max * Phi(d/d_max), Phi calibrada con el perfil medido."""
    d = distance_to_coast(wet, dx, dy, coast=coast, X=X, Y=Y)
    d_fin = d[np.isfinite(d)]
    d_max = float(d_fin.max()) if d_fin.size else 1.0
    rejilla, phi = _perfil_normalizado(profile)
    h = np.zeros_like(X)
    h[wet] = profile.max_depth * np.interp(d[wet] / d_max, rejilla, phi)
    supuestos = {
        "model": "depth is a function of distance-to-shore only",
        "shape_function": "Phi calibrated from the measured track profile",
        "d_max_m": d_max,
        "limitation": ("impone simetria respecto de la costa; no representa "
                       "canales profundos adosados a un margen"),
    }
    return h, supuestos


def _strategy_nearest_track(X, Y, dx, dy, wet, profile, coast,
                            track_xy=None, **kw):
    """
    Profundidad del punto mas cercano de la traza, atenuada hacia la costa.

    `track_xy` = (x_traza, y_traza) con la MISMA longitud que profile.distance.
    Si no se pasa, la traza se coloca como una recta que cruza el centroide de
    la region mojada a lo largo de x: es una idealizacion mas, declarada como tal.
    """
    if track_xy is None:
        xs = X[wet]
        ys = Y[wet]
        y0 = float(np.mean(ys))
        x_min, x_max = float(xs.min()), float(xs.max())
        tx = np.linspace(x_min, x_max, profile.distance.size)
        ty = np.full_like(tx, y0)
        colocacion = ("traza IDEALIZADA: recta en x por el centroide de la "
                      "region mojada")
    else:
        tx, ty = np.asarray(track_xy[0], float), np.asarray(track_xy[1], float)
        if tx.size != profile.distance.size:
            raise ValueError("track_xy debe tener la longitud del perfil.")
        colocacion = "traza georreferenciada suministrada"

    xs, ys = X[wet], Y[wet]
    idx = np.empty(xs.size, dtype=int)
    bloque = max(1, int(4.0e7 / max(xs.size, 1)))
    mejor = np.full(xs.shape, np.inf)
    for i in range(0, tx.size, bloque):
        d2 = (xs[:, None] - tx[None, i:i + bloque]) ** 2 + \
             (ys[:, None] - ty[None, i:i + bloque]) ** 2
        loc = np.argmin(d2, axis=1)
        val = d2[np.arange(xs.size), loc]
        actualiza = val < mejor
        mejor[actualiza] = val[actualiza]
        idx[actualiza] = i + loc[actualiza]

    h = np.zeros_like(X)
    h_traza = profile.depth[idx]
    # Atenuacion hacia la costa: sin ella el metodo dejaria un escalon de
    # profundidad finita justo en la orilla, fisicamente imposible.
    d_costa = distance_to_coast(wet, dx, dy, coast=coast, X=X, Y=Y)
    d_fin = d_costa[np.isfinite(d_costa)]
    esc = float(np.percentile(d_fin, 90)) if d_fin.size else 1.0
    atenua = np.clip(d_costa[wet] / max(esc, 1e-9), 0.0, 1.0)
    h[wet] = h_traza * atenua
    supuestos = {
        "model": "nearest-point extrapolation from the altimetric track",
        "track_placement": colocacion,
        "shore_taper_scale_m": esc,
        "limitation": ("sin informacion fuera del entorno de la traza; genera "
                       "artefactos en la mediatriz entre segmentos"),
    }
    return h, supuestos


def _strategy_hypsometric(X, Y, dx, dy, wet, profile, coast,
                          exponent: float | None = None, **kw):
    """h0 = h_max * (1 - (r/R)^p), cuerpo de revolucion ajustado al perfil."""
    xs, ys = X[wet], Y[wet]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    r = np.hypot(X - cx, Y - cy)
    R = float(np.max(np.hypot(xs - cx, ys - cy)))
    p = exponent if exponent is not None else _ajustar_exponente(profile)
    h = np.zeros_like(X)
    h[wet] = profile.max_depth * np.maximum(
        0.0, 1.0 - (r[wet] / max(R, 1e-9)) ** p)
    supuestos = {
        "model": "axisymmetric parametric basin h = h_max (1 - (r/R)^p)",
        "center_m": (cx, cy), "R_m": R, "exponent_p": float(p),
        "limitation": ("ignora la forma de la costa; para una cuenca "
                       "trilobulada como Ligeia es una idealizacion fuerte"),
    }
    return h, supuestos


def _ajustar_exponente(profile: TrackProfile) -> float:
    """
    Ajusta p minimizando el error cuadratico entre 1-(1-d_rel)^p y la forma
    normalizada del perfil. Barrido directo: el espacio es 1D y pequeno.
    """
    d_rel, phi = _perfil_normalizado(profile)
    ps = np.linspace(0.5, 6.0, 111)
    err = [np.sum((1.0 - (1.0 - d_rel) ** p - phi) ** 2) for p in ps]
    return float(ps[int(np.argmin(err))])


STRATEGIES = {
    "coast_distance": _strategy_coast_distance,
    "nearest_track": _strategy_nearest_track,
    "hypsometric": _strategy_hypsometric,
}


def reconstruct(x: np.ndarray, y: np.ndarray, profile: TrackProfile,
                coast: Coastline, strategy: str = "coast_distance",
                smooth_passes: int = 2, land_slope: float = 1.0e-3,
                **kwargs) -> BathymetryResult:
    """
    Reconstruye h0(x,y) con la estrategia indicada.

    smooth_passes : pasadas de suavizado de 5 puntos aplicadas SOLO al interior
        mojado. Elimina el ruido de discretizacion de la costa sin mover la
        linea de orilla. 0 lo desactiva.

    land_slope : pendiente del TERRENO EMERGIDO [m/m]. Fuera del contorno de
        costa se impone h0 = -land_slope * d, con d la distancia a la costa, de
        modo que el terreno asciende al alejarse del mar.

        POR QUE HACE FALTA Y POR QUE ES UN SUPUESTO. Sin topografia emergida el
        "terreno" seria una plataforma plana a la cota del datum y el run-up
        seria identicamente cero mientras el agua se extenderia sin freno, y
        la inundacion resultante no tendria significado fisico.

        La topografia costera de Ligeia NO esta medida con la resolucion que
        este calculo necesita. `land_slope` es por tanto un PARAMETRO LIBRE del
        estudio de sensibilidad, como los coeficientes de friccion, y
        NO vive en titan_params.py. El valor por defecto 1e-3 (1 m por km) es
        del orden de los margenes muy tendidos que muestra el radar, pero no es
        una medida: el run-up escala como 1/land_slope y hay que barrerlo.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Estrategia desconocida: {strategy!r}. "
                         f"Validas: {tuple(STRATEGIES)}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X, Y = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    wet = coast.contains(X, Y)
    if not np.any(wet):
        raise ValueError("El contorno de costa no encierra ninguna celda: "
                         "revisa las coordenadas del dominio.")

    h, supuestos = STRATEGIES[strategy](X, Y, dx, dy, wet, profile, coast,
                                        **kwargs)
    h = np.where(wet, h, 0.0)
    for _ in range(max(0, smooth_passes)):
        h = _suavizar(h, wet)
    supuestos["smooth_passes"] = smooth_passes

    # --- Topografia emergida (SUPUESTO declarado, no dato) ------------------
    if land_slope > 0.0 and np.any(~wet):
        d_tierra = distance_to_coast(~wet, dx, dy, coast=coast, X=X, Y=Y)
        h = np.where(wet, h, -land_slope * np.where(np.isfinite(d_tierra),
                                                    d_tierra, 0.0))
    supuestos["land_slope_m_per_m"] = land_slope
    supuestos["land_topography"] = (
        "SUPUESTO, NO DATO: terreno emergido con pendiente constante desde la "
        "costa. El run-up escala como 1/land_slope; barrer en sensibilidad.")

    return BathymetryResult(
        depth=h, x=x, y=y, strategy=strategy, assumptions=supuestos,
        profile_source=profile.source, coast_source=coast.source,
        is_data_based=bool(profile.is_data and coast.is_data))


# ===========================================================================
# 5. BATIMETRIA PUBLICADA DE LORENZ et al. 2014 (Icarus 237, 9-15), Apendice A
# ===========================================================================
"""
ADVERTENCIA DE LA FUENTE, REPRODUCIDA LITERALMENTE EN SU ALCANCE
-----------------------------------------------------------------
Lorenz et al. (2014) declaran que este mapa NO es apto para cantidades precisas
ni para comparar regiones pequenas separadas entre si. Es una construccion
ANALITICA: la profundidad se asigna de forma proporcional a la distancia a la
costa, calibrada con unos pocos sondeos de radar. En consecuencia:

  * Las unicas conclusiones admisibles son CUALITATIVAS: la FORMA del campo.
  * NO se pueden reportar volumenes, profundidades locales ni comparaciones
    punto a punto entre cuencas.
  * El area de pixel varia hasta un 37% en el dominio (proyeccion), asi que
    toda escala horizontal arrastra esa incertidumbre. Por eso el diagnostico
    de forma se repite perturbando la escala +-37%.

CONSECUENCIA METODOLOGICA CRITICA para el uso que se le da aqui: como la
profundidad es proporcional a la distancia a la costa POR CONSTRUCCION, este
mapa casi no contiene tramos de PROFUNDIDAD UNIFORME. Y el fetch de profundidad
casi uniforme es exactamente la condicion necesaria para que la resonancia de
Proudman se sostenga: sobre fondo inclinado la onda ligada radia hacia
profundidades no resonantes y se desintoniza. Por tanto, encontrar aqui
ausencia de resonancia
es ESPERABLE y NO demuestra que no la haya en la Ligeia real: demuestra que no
la hay en la batimetria analitica estandar.
"""


@dataclass(frozen=True)
class LorenzMap:
    """
    Mapa batimetrico de Lorenz et al. 2014 con su georreferenciacion.

    depth : (325, 270) int/float, PROFUNDIDAD EN METROS por celda; 0 = tierra.
    S, Xc, Yc : parametros de la proyeccion azimutal polar norte del Apendice A.

    CONVENCION DE EJES, determinada por VALIDACION y no por suposicion:
        X = indice de FILA,  Y = indice de COLUMNA
    Es la convencion (X, Y) = (primer eje, segundo eje) tipica del codigo
    Fortran/IDL de la epoca. Se eligio porque es la unica de las dos que
    reproduce el criterio que da la propia fuente (Ligeia ~170 m a ~79 N):
    con X=fila el maximo de Ligeia cae en 79.53 N; con X=columna, en 81.44 N.
    Ver `validation_report()` para las tres comprobaciones independientes.

    PROYECCION: azimutal polar norte. El radio en pixeles es proporcional a la
    colatitud, r = |S| * colat_rad. Se valida de forma INDEPENDIENTE contra la
    escala derivada de pixel_area_km2: R_Titan / sqrt(29 km^2) = 478.1 px/rad,
    frente al |S| = 495 declarado; concuerdan al 3.5%, holgadamente dentro de la
    variacion de area de pixel del 37% que la propia fuente declara.

    AZIMUT: el signo negativo de S implica una rotacion de 180 grados, de modo
    que la longitud este es lon_E = atan2(Y - Yc, X - Xc) + 180 grados. Con esa
    convencion las tres cuencas caen en sus posiciones conocidas (ver
    validation_report). El origen de azimut queda determinado con ~5-10 grados
    de holgura, suficiente para un recorte por caja pero NO para cartografia.
    """
    depth: np.ndarray
    S: float
    Xc: float
    Yc: float
    source: str

    @property
    def pixel_km(self) -> float:
        """
        Lado nominal de pixel [km], DERIVADO de pixel_area_km2. No se fija a
        mano: sqrt(29 km^2) = 5.385 km.
        """
        return float(np.sqrt(LORENZ_BATHYMETRY["pixel_area_km2"][0]))

    @property
    def pixel_m(self) -> float:
        return self.pixel_km * 1.0e3

    def _dxdy(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.depth.shape
        R, C = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return R - self.Xc, C - self.Yc

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat_N, lon_E) en grados para cada celda de la malla."""
        dx, dy = self._dxdy()
        colat = np.degrees(np.hypot(dx, dy) / abs(self.S))
        lat = 90.0 - colat
        lon = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 360.0
        return lat, lon

    def validation_report(self) -> dict:
        """
        Las tres comprobaciones independientes que validan la georreferenciacion.
        Ninguna usa un valor inventado: todas contrastan contra constantes ya
        registradas en titan_params.py o contra el criterio de la propia fuente.
        """
        lat, lon = self.latlon()
        liq = self.depth > 0
        out = {"pixel_km": self.pixel_km,
               "scale_declared_px_per_rad": abs(self.S),
               "scale_derived_px_per_rad": TITAN.radius * 1e-3 / self.pixel_km,
               "lat_range_liquid": (float(lat[liq].min()), float(lat[liq].max()))}
        out["scale_mismatch"] = abs(
            out["scale_declared_px_per_rad"] - out["scale_derived_px_per_rad"]
        ) / out["scale_derived_px_per_rad"]
        # 1) maximo global = Kraken
        i = np.unravel_index(np.argmax(self.depth), self.depth.shape)
        out["global_max"] = {"depth_m": float(self.depth[i]),
                             "lat_N": float(lat[i]), "lon_E": float(lon[i])}
        # 2) maximo de la caja de Ligeia
        m = self.box_mask(*LIGEIA_LATLON_BOX)
        j = np.argmax(np.where(m, self.depth, -1))
        j = np.unravel_index(j, self.depth.shape)
        out["ligeia_max"] = {"depth_m": float(self.depth[j]),
                             "lat_N": float(lat[j]), "lon_E": float(lon[j])}
        return out

    def box_mask(self, lat_min: float, lat_max: float,
                 lon_min: float, lon_max: float) -> np.ndarray:
        """Mascara de la caja lat/lon (grados). Solo celdas liquidas."""
        lat, lon = self.latlon()
        return ((self.depth > 0) & (lat >= lat_min) & (lat <= lat_max)
                & (lon >= lon_min) & (lon <= lon_max))

    def extract_box(self, lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float,
                    margin: int = 4) -> tuple[np.ndarray, dict]:
        """
        Recorta la subregion delimitada por la caja lat/lon.

        Devuelve (subgrilla_de_profundidad, metadatos). Las celdas liquidas
        FUERA de la caja se ponen a 0 (tierra): asi el estrecho que une Ligeia
        con Kraken queda cerrado por el corte geografico, que es exactamente el
        criterio pedido (recorte por lat/lon, NO por componente conectado, que
        no serviria porque en este mapa las dos cuencas estan fusionadas).

        `margin` anade unas celdas de tierra alrededor para que la frontera del
        dominio no corte la costa.
        """
        m = self.box_mask(lat_min, lat_max, lon_min, lon_max)
        if not np.any(m):
            raise ValueError("La caja lat/lon no contiene celdas liquidas.")
        ys, xs = np.where(m)
        y0 = max(0, ys.min() - margin)
        y1 = min(self.depth.shape[0], ys.max() + margin + 1)
        x0 = max(0, xs.min() - margin)
        x1 = min(self.depth.shape[1], xs.max() + margin + 1)
        sub = np.where(m, self.depth, 0)[y0:y1, x0:x1].astype(float)
        lat, lon = self.latlon()
        meta = {
            "box_lat_N": (lat_min, lat_max), "box_lon_E": (lon_min, lon_max),
            "rows": (int(y0), int(y1)), "cols": (int(x0), int(x1)),
            "shape": tuple(int(v) for v in sub.shape),
            "pixel_km": self.pixel_km,
            "n_liquid": int(np.count_nonzero(sub > 0)),
            "liquid_fraction": float(np.mean(sub > 0)),
            "max_depth_m": float(sub.max()),
            "lat_range_N": (float(lat[m].min()), float(lat[m].max())),
            "lon_range_E": (float(lon[m].min()), float(lon[m].max())),
            "lat_of_max_N": float(lat[y0:y1, x0:x1][np.unravel_index(
                np.argmax(sub), sub.shape)]),
            "source": self.source,
        }
        return sub, meta


# CRITERIO DE RECORTE DE LIGEIA.
# Caja generosa centrada en el maximo de Ligeia (79.5 N, 109 E). Los limites se
# eligen para (a) contener la cuenca entera, (b) cortar el estrecho que la une a
# Kraken y (c) excluir el maximo global de 197 m, que es de Kraken. La holgura
# es deliberada: el origen de azimut solo esta determinado con ~5-10 grados.
LIGEIA_LATLON_BOX = (74.0, 86.0, 75.0, 145.0)   # (lat_min, lat_max, lon_min, lon_max)


def load_lorenz_map(path: str | Path) -> LorenzMap:
    """
    Carga el mapa ASCII de Lorenz et al. 2014, Apendice A.

    `path` es OBLIGATORIO. El fichero de datos no forma parte del paquete, y una
    ruta por defecto relativa se resolveria contra el directorio de trabajo, de
    modo que la misma llamada leeria un fichero u otro -o ninguno- segun desde
    donde se ejecutase. Quien llama debe construir la ruta de forma explicita,
    derivandola de la raiz de su proyecto.

    FORMATO: 325 filas x 270 columnas, un entero por celda con la profundidad en
    metros (0 = tierra). Terminadores CR/LF: se abre en modo texto universal, de
    modo que tanto CRLF como LF se normalizan. Los campos van separados por
    espacios de ancho variable (la primera fila no lleva el mismo relleno que las
    demas), asi que se parsea por separacion en blancos y NO por columnas fijas.

    Se valida la forma y que no haya profundidades negativas.
    """
    path = Path(path)
    with path.open("r", newline=None, encoding="ascii") as f:
        filas = [l.split() for l in f.read().splitlines() if l.strip()]
    anchos = {len(r) for r in filas}
    if len(anchos) != 1:
        raise ValueError(f"{path}: filas de anchura distinta: {sorted(anchos)}")
    grid = np.array([[int(v) for v in r] for r in filas], dtype=float)
    if grid.shape != (325, 270):
        raise ValueError(f"{path}: forma {grid.shape}, se esperaba (325, 270).")
    if np.any(grid < 0):
        raise ValueError(f"{path}: hay profundidades negativas.")
    return LorenzMap(
        depth=grid,
        S=LORENZ_BATHYMETRY["georef_S"][0],
        Xc=LORENZ_BATHYMETRY["georef_Xc"][0],
        Yc=LORENZ_BATHYMETRY["georef_Yc"][0],
        source=("Lorenz et al. 2014, Icarus 237, 9-15, Apendice A "
                "(mapa batimetrico analitico; NO apto para cantidades precisas "
                "ni comparacion de regiones pequenas separadas)"))


def lorenz_region_to_bathymetry(sub: np.ndarray, meta: dict,
                                land_slope: float = 1.0e-3,
                                scale_factor: float = 1.0
                                ) -> tuple[BathymetryResult, float]:
    """
    Convierte una subgrilla de Lorenz en un BathymetryResult listo para el motor.

    scale_factor : perturbacion de la ESCALA HORIZONTAL. La fuente declara hasta
        un 37% de variacion del area de pixel en el dominio, asi que el
        diagnostico debe repetirse con scale_factor = 1 +- 0.37 para comprobar
        que la conclusion no depende de la distorsion de proyeccion.
        NOTA: perturba solo dx, no la profundidad; es una distorsion horizontal.

    Devuelve (BathymetryResult, dx_en_metros).
    """
    dx = meta["pixel_km"] * 1.0e3 * scale_factor
    ny, nx = sub.shape
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    h = sub.astype(float).copy()
    wet = h > 0.0

    # Topografia emergida, mismo criterio que la reconstruccion idealizada:
    # PARAMETRO LIBRE, no dato.
    if land_slope > 0.0 and np.any(~wet):
        X, Y = np.meshgrid(x, y)
        d = distance_to_coast(~wet, dx, dx, X=X, Y=Y)
        h = np.where(wet, h, -land_slope * np.where(np.isfinite(d), d, 0.0))

    supuestos = {
        "model": "published analytic bathymetry (Lorenz et al. 2014, App. A)",
        "depth_proportional_to_distance_to_shore": True,
        "pixel_km_nominal": meta["pixel_km"],
        "horizontal_scale_factor": scale_factor,
        "land_slope_m_per_m": land_slope,
        "box_lat_N": meta["box_lat_N"], "box_lon_E": meta["box_lon_E"],
        "limitation": ("la fuente declara que el mapa NO sirve para cantidades "
                       "precisas ni para comparar regiones pequenas separadas; "
                       "solo conclusiones cualitativas de forma de campo"),
    }
    res = BathymetryResult(
        depth=h, x=x, y=y, strategy="lorenz2014_published",
        assumptions=supuestos, profile_source=meta["source"],
        coast_source=meta["source"], is_data_based=True)
    return res, dx


def _suavizar(h: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Media de 5 puntos aplicada solo donde hay agua; la costa no se mueve."""
    acc = h.copy()
    cnt = np.ones_like(h)
    for eje, desp in ((0, 1), (0, -1), (1, 1), (1, -1)):
        vec = np.roll(h, desp, axis=eje)
        vm = np.roll(wet, desp, axis=eje)
        acc += np.where(vm, vec, 0.0)
        cnt += vm.astype(float)
    return np.where(wet, acc / cnt, 0.0)
