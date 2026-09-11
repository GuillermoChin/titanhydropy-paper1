"""
utils/visualizer.py
===================
Figuras del proyecto. TEXTO DE FIGURA EN INGLES (norma del proyecto); los
comentarios y los mensajes de consola, en espanol.

matplotlib es una dependencia OPCIONAL: si no esta instalada, las funciones
avisan y devuelven None en vez de romper una corrida de calculo.

CRITERIOS DE DISENO
-------------------
* Mapa de zeta: colormap DIVERGENTE centrado en cero. La elevacion de superficie
  libre tiene signo fisico (subida vs bajada) y un colormap secuencial lo
  ocultaria.
* Mapa de Froude: colormap divergente centrado en F = 1, que es el valor con
  significado. Se dibuja ademas el contorno F = 1.
* Curva A vs U: se superpone la teoria estacionaria 1/|1-F^2| como referencia,
  y se marca la velocidad resonante c = sqrt(g h).
* Toda figura de resultado lleva anotadas las magnitudes que la hacen
  interpretable (h, dP, resolucion), para que no se pueda leer fuera de su
  contexto numerico.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED, TITAN,
                                    inverse_barometer)
from physics.proudman import proudman_amplification, resonance_speed

__all__ = ["plot_zeta_map", "plot_froude_map", "plot_amplification_curve",
           "plot_bathymetry", "plot_profile_comparison", "matplotlib_available"]


def _mpl():
    """Importa matplotlib con backend no interactivo. None si no esta."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[aviso] matplotlib no instalado; se omite la figura.")
        return None


def matplotlib_available() -> bool:
    return _mpl() is not None


def _guardar(fig, path: Path | str, dpi: int = 300) -> Path:
    """
    Guarda la figura en PNG **y** en PDF vectorial con el mismo nombre base.

    Icarus exige arte vectorial o, en su defecto, 300 dpi como minimo; se dan
    ambas cosas. El nombre del PNG no cambia, para no romper referencias.

    No se usa `bbox_inches="tight"`: las figuras se crean con
    `layout="constrained"`, que ya reserva el espacio de titulos y leyendas.
    Combinar ambos recortaria el lienzo de forma distinta en PNG y en PDF, y las
    dos versiones dejarian de ser la misma figura.

    DETERMINISMO DEL PDF. matplotlib estampa la hora de pared en
    `/CreationDate` dentro del diccionario de metadatos del PDF, de modo que dos
    corridas identicas producian PDF con distinto digest: cuatro bytes, todos
    dentro de la marca de tiempo. Un revisor que regenerase las figuras y
    ejecutase `sha256sum -c` veia cuatro ficheros fallando y tenia que averiguar
    por su cuenta que era un sello temporal y no un cambio de contenido.
    `metadata={"CreationDate": None}` omite el campo y deja el PDF tan
    reproducible como el PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, metadata={"CreationDate": None})
    print(f"[figura] {path}")
    print(f"[figura] {pdf}")
    return path


# ---------------------------------------------------------------------------
# Mapas 2D
# ---------------------------------------------------------------------------
def _titulo(ax, title: str, subtitle: str | None) -> None:
    """
    Titulo + subtitulo sin colision.

    El subtitulo se coloca en y = 1.01 en coordenadas de ejes, asi que el titulo
    necesita `pad` suficiente para no solaparse con el. Sin el pad, ambos caen a
    la misma altura y el texto se pisa, que es lo que ocurre si el subtitulo se
    dibuja a mano sin reservar el hueco. Por eso esta funcion es el unico
    camino admitido para poner titulo + subtitulo.
    """
    ax.set_title(title, pad=22 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=8,
                color="0.35", va="bottom", ha="left")


def plot_zeta_map(x: np.ndarray, y: np.ndarray, zeta: np.ndarray,
                  path: Path | str, title: str | None = None,
                  subtitle: str | None = None,
                  vmax: float | None = None,
                  mask_dry: np.ndarray | None = None,
                  cbar_label: str | None = None):
    """
    Mapa 2D con escala divergente centrada en cero.

    `cbar_label` es OBLIGATORIO en la practica cuando el campo no es zeta. Esta
    funcion se reutiliza para el mapa de AMPLIFICACION, que es adimensional;
    dejar la etiqueta por defecto ("Free-surface elevation [m]") etiquetaria mal
    la figura, que es peor que no tener figura.
    """
    plt = _mpl()
    if plt is None:
        return None
    z = np.asarray(zeta, dtype=float).copy()
    if mask_dry is not None:
        z = np.ma.masked_where(mask_dry, z)
    lim = float(vmax if vmax is not None else np.max(np.abs(z)))
    lim = lim if lim > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    im = ax.pcolormesh(x / 1e3, y / 1e3, z, cmap="RdBu_r",
                       vmin=-lim, vmax=lim, shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label or "Free-surface elevation $\\zeta$ [m]")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal")
    _titulo(ax, title or "Free-surface elevation", subtitle)
    return _guardar(fig, path)


def plot_froude_map(x: np.ndarray, y: np.ndarray, froude: np.ndarray,
                    path: Path | str, storm_speed: float,
                    title: str | None = None, fmax: float = 2.0):
    """
    Mapa del campo de Froude local con escala centrada en F = 1 y contorno de
    resonancia. Las celdas secas (F = inf) se enmascaran.
    """
    plt = _mpl()
    if plt is None:
        return None
    f = np.asarray(froude, dtype=float)
    fm = np.ma.masked_invalid(np.where(np.isfinite(f), f, np.nan))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    im = ax.pcolormesh(x / 1e3, y / 1e3, fm, cmap="RdBu_r",
                       vmin=2.0 - fmax, vmax=fmax, shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Atmospheric-oceanic Froude number $F = U/\\sqrt{gh}$")
    try:
        cs = ax.contour(x / 1e3, y / 1e3, fm, levels=[1.0], colors="k",
                        linewidths=1.4)
        ax.clabel(cs, fmt={1.0: "F = 1"}, fontsize=8)
    except (ValueError, TypeError):
        pass  # sin cruce de F = 1 en el dominio
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal")
    _titulo(ax, title or f"Local Froude field, storm speed U = "
                         f"{storm_speed:.2f} m/s",
            f"Resonant depth $h = U^2/g$ = {storm_speed**2/TITAN.g:.1f} m "
            f"(g = {TITAN.g} m s$^{{-2}}$)")
    return _guardar(fig, path)


def plot_bathymetry(x: np.ndarray, y: np.ndarray, depth: np.ndarray,
                    path: Path | str, title: str | None = None,
                    resonant_band: tuple[float, float] | None = None):
    """
    Mapa de batimetria. Si se pasa `resonant_band`, se sombrea la franja de
    profundidad resonante: es la lectura clave, porque la resonancia solo es
    alcanzable en esa franja y no en toda la cuenca.
    """
    plt = _mpl()
    if plt is None:
        return None
    h = np.ma.masked_less_equal(np.asarray(depth, dtype=float), 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    im = ax.pcolormesh(x / 1e3, y / 1e3, h, cmap="viridis", shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Rest depth $h_0$ [m]")
    if resonant_band is not None:
        lo, hi = resonant_band
        try:
            cs = ax.contour(x / 1e3, y / 1e3, h, levels=[lo, hi],
                            colors="crimson", linewidths=1.2, linestyles="--")
            ax.clabel(cs, fmt=lambda v: f"{v:.0f} m", fontsize=8)
        except (ValueError, TypeError):
            pass
        ax.plot([], [], "--", color="crimson",
                label=f"Resonant band {lo:.0f}-{hi:.0f} m")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal")
    ax.set_title(title or "Reconstructed bathymetry")
    return _guardar(fig, path)


# ---------------------------------------------------------------------------
# Curvas
# ---------------------------------------------------------------------------
def plot_amplification_curve(speeds: np.ndarray, amplification: np.ndarray,
                             depth: float, path: Path | str,
                             delta_p: float = 100.0,
                             bound_amplification: np.ndarray | None = None,
                             extra_curves: dict | None = None,
                             title: str | None = None):
    """
    Curva A vs U, con la teoria estacionaria superpuesta y la banda de
    velocidades OBSERVADAS de frentes convectivos sombreada.

    `extra_curves` permite superponer resoluciones distintas para mostrar la
    convergencia de malla: {'etiqueta': (speeds, A)}.
    """
    plt = _mpl()
    if plt is None:
        return None
    u = np.asarray(speeds, dtype=float)
    a = np.asarray(amplification, dtype=float)
    c = resonance_speed(depth)

    u_teo = np.linspace(max(u.min() * 0.5, 0.05), u.max() * 1.05, 800)
    a_teo = proudman_amplification(u_teo / c)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")

    v_min, v_max, _, _ = OBSERVED_FRONT_SPEED
    ax.axvspan(v_min, v_max, color="0.85", zorder=0,
               label=f"Observed front speeds {v_min:.0f}-{v_max:.0f} m/s")

    # La teoria estacionaria DIVERGE en F = 1, asi que la curva se sale por
    # arriba del marco. Es correcto y no se recorta: se atenua con alpha y se
    # dice en la leyenda, para que se lea como intencional y no como un fallo
    # de trazado.
    ax.plot(u_teo, a_teo, "k-", lw=1.2, alpha=0.6,
            label=r"Steady linear theory $1/|1-F^2|$ (diverges at $F=1$)")
    ax.plot(u, a, "ro-", ms=5, lw=1.3, label="Native FVM (global max)")
    if bound_amplification is not None:
        ax.plot(u, np.asarray(bound_amplification, dtype=float), "s--",
                color="darkorange", ms=4, lw=1.0,
                label="Native FVM (bound response)")
    if extra_curves:
        for etiqueta, (uu, aa) in extra_curves.items():
            ax.plot(uu, aa, ":", lw=1.0, label=etiqueta)

    ax.axvline(c, color="0.4", ls=":", lw=1.0)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax.annotate(f"$c=\\sqrt{{gh}}$ = {c:.2f} m/s", xy=(c, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=8, color="0.35", rotation=90, va="top")

    ax.set_xlim(u.min() * 0.0, u.max() * 1.02)
    ax.set_ylim(0.0, 1.35 * float(np.nanmax(a)))
    ax.set_xlabel("Storm speed $U$ [m s$^{-1}$]")
    ax.set_ylabel(r"Amplification  $A = \max|\zeta| \,/\, |\zeta_{IB}|$")
    # Titulo y subtitulo por _titulo(): dibujarlos a mano era lo que hacia que
    # el subtitulo cayera ENCIMA del titulo en la figura 3 del Paper 1.
    _titulo(ax, title or f"Proudman resonance sweep (h = {depth:.0f} m)",
            f"$g$ = {TITAN.g} m s$^{{-2}}$, $\\rho$ = {DEFAULT_FLUID.rho} "
            f"kg m$^{{-3}}$, $\\Delta P$ = {delta_p:.0f} Pa, "
            f"$|\\zeta_{{IB}}|$ = {abs(inverse_barometer(delta_p)):.4f} m")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _guardar(fig, path)


def plot_profile_comparison(x: np.ndarray, curves: dict, path: Path | str,
                            ylabel: str = "Free-surface elevation $\\zeta$ [m]",
                            title: str | None = None,
                            subtitle: str | None = None):
    """
    Compara varios perfiles 1D. `curves` = {'etiqueta': (y, estilo_dict)} o
    {'etiqueta': y}.
    """
    plt = _mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    for etiqueta, dato in curves.items():
        y, estilo = dato if isinstance(dato, tuple) else (dato, {})
        ax.plot(np.asarray(x) / 1e3, np.asarray(y), label=etiqueta, **estilo)
    ax.set_xlabel("x [km]")
    ax.set_ylabel(ylabel)
    # Ver _titulo(): reserva el hueco del subtitulo con `pad`. La version
    # anterior escribia el subtitulo en y = 1.02 sin pad y se pisaba con el
    # titulo, que es el defecto de la figura 1 del Paper 1.
    _titulo(ax, title or "Profile comparison", subtitle)
    ax.legend(frameon=False, fontsize=8)
    return _guardar(fig, path)
