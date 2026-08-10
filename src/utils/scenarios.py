"""
utils/scenarios.py
==================
Escenarios reproducibles de alto nivel. Vive en `utils/` y no en `experiments/`
para que los TESTS puedan importarlo: una compuerta de validacion tiene que
correr exactamente el mismo escenario que el experimento del paper, no una
version parecida.

ESCENARIO LIGEIA (Sprint 3 / Paper 2)
------------------------------------
Frente convectivo movil de presion sobre la batimetria reconstruida de Ligeia
Mare, con wetting-and-drying activo y Coriolis en el plano f de la latitud
polar norte.

Diagnosticos que produce:
  * ENVOLVENTE de max|zeta| a lo largo de toda la corrida (no el instante
    final: el instante final no tiene por que ser el de maxima inundacion).
  * MAPA DE AMPLIFICACION respecto de la respuesta estatica de barometro
    inverso: A(x,y) = envolvente|zeta| / |zeta_IB|.
  * CAMPO DE FROUDE local F(x,y) = U/sqrt(g h) y la region F -> 1 (H-002).
  * RUN-UP e inundacion sobre la envolvente.

TODO valor fisico se importa de constants/titan_params.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constants.titan_params import (DEFAULT_FLUID, OBSERVED_FRONT_SPEED, TITAN,
                                    inverse_barometer, resonant_depth_band)
from core.native_fvm2d import NativeFVMSolver2D
from core.solver_base import Domain, ForcingBundle, SolverConfig, State
from domain.bathymetry import (BathymetryResult, Coastline, TrackProfile,
                               idealized_coastline, idealized_track_profile,
                               reconstruct)
from forcing.atmospheric import MovingPressureFront2D
from forcing.friction import LinearFriction
from physics.proudman import (froude_field, resonance_mask,
                              resonant_region_summary)
from physics.runup import compute_runup, track_max_envelope

__all__ = ["LigeiaConfig", "LigeiaResult", "build_ligeia_domain",
           "run_ligeia_scenario"]


@dataclass
class LigeiaConfig:
    """
    Todos los supuestos del escenario, en un solo sitio y explicitos.

    LX, LY      : extension del dominio [m]. ~470x420 km, del orden de Ligeia.
    dx          : paso de malla [m]. Es el parametro del estudio de convergencia.
    strategy    : estrategia de reconstruccion batimetrica (ver domain/bathymetry).
    storm_speed : velocidad del frente [m/s]. Por defecto el maximo de
                  OBSERVED_FRONT_SPEED, que es el que maximiza la profundidad
                  resonante alcanzable (h_res = U^2/g).
    heading_deg : direccion de avance del frente.
    delta_p     : amplitud de la perturbacion de presion [Pa].
    sigma       : semiancho del frente [m].
    coriolis    : activar Coriolis en el plano f de lat0_deg.
    friction_r  : coeficiente de friccion lineal [1/s]. PARAMETRO LIBRE, no dato.
    """
    LX: float = 4.7e5
    LY: float = 4.2e5
    dx: float = 2.0e3
    strategy: str = "coast_distance"
    storm_speed: float = OBSERVED_FRONT_SPEED[1]
    heading_deg: float = 0.0
    delta_p: float = 100.0
    sigma: float = 2.0e4
    coriolis: bool = True
    lat0_deg: float = 78.0
    friction_r: float = 1.0e-5
    # Pendiente del terreno emergido [m/m]. PARAMETRO LIBRE, no dato: el run-up
    # escala como 1/land_slope. Ver domain/bathymetry.reconstruct.
    land_slope: float = 1.0e-3
    cfl: float = 0.4
    min_depth: float = 1e-2
    n_envelope_samples: int = 200
    profile: TrackProfile | None = None
    coast: Coastline | None = None
    # Batimetria YA construida. Si se pasa, se usa tal cual y se ignoran
    # profile/coast/strategy/LX/LY/dx, que se reajustan a la malla recibida.
    # Es el enganche del experimento-puente: permite correr EXACTAMENTE el mismo
    # diagnostico de forma de campo sobre la batimetria publicada de Lorenz.
    bathymetry: BathymetryResult | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("profile", "coast", "bathymetry")}
        d["g"] = TITAN.g
        d["rho"] = DEFAULT_FLUID.rho
        d["zeta_ib"] = abs(inverse_barometer(self.delta_p))
        return d


@dataclass
class LigeiaResult:
    """Salida del escenario, con todo lo necesario para figuras y compuertas."""
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray                 # h0 de reposo [m]
    envelope: np.ndarray              # max|zeta| a lo largo de la corrida [m]
    amplification: np.ndarray         # envolvente / |zeta_IB|
    froude: np.ndarray                # F(x,y)
    final_state: State
    bathymetry: BathymetryResult
    runup: object                     # RunupResult sobre la envolvente
    resonance: dict
    diagnostics: dict
    config: LigeiaConfig
    history: dict = field(default_factory=dict)

    @property
    def wet(self) -> np.ndarray:
        return self.depth > self.config.min_depth

    def max_amplification(self) -> float:
        return float(np.max(self.amplification[self.wet]))

    def provenance(self) -> dict:
        p = self.bathymetry.provenance()
        p.update({"scenario": "ligeia", **self.config.as_dict()})
        return p


def build_ligeia_domain(cfg: LigeiaConfig) -> tuple[Domain, BathymetryResult]:
    """
    Malla + batimetria, con la procedencia adjunta.

    Si `cfg.bathymetry` viene dado (experimento-puente con la batimetria
    publicada de Lorenz), se adopta tal cual y se REAJUSTAN cfg.LX, cfg.LY y
    cfg.dx a esa malla, para que el resto del escenario —recorrido del frente,
    posicion inicial, muestreo— quede consistente con el dominio real.
    """
    if cfg.bathymetry is not None:
        b = cfg.bathymetry
        cfg.dx = float(b.x[1] - b.x[0])
        cfg.LX = float(b.x[-1] + 0.5 * cfg.dx)
        cfg.LY = float(b.y[-1] + 0.5 * cfg.dx)
        dom = Domain(x=b.x, y=b.y, bathymetry=b.depth, lat0_deg=cfg.lat0_deg)
        return dom, b

    nx = int(round(cfg.LX / cfg.dx))
    ny = int(round(cfg.LY / cfg.dx))
    x = (np.arange(nx) + 0.5) * cfg.dx
    y = (np.arange(ny) + 0.5) * cfg.dx
    profile = cfg.profile or idealized_track_profile()
    coast = cfg.coast or idealized_coastline(lx=cfg.LX, ly=cfg.LY)
    bathy = reconstruct(x, y, profile, coast, strategy=cfg.strategy,
                        land_slope=cfg.land_slope)
    dom = Domain(x=x, y=y, bathymetry=bathy.depth, lat0_deg=cfg.lat0_deg)
    return dom, bathy


def run_ligeia_scenario(cfg: LigeiaConfig | None = None,
                        verbose: bool = False,
                        progress: "callable | None" = None,
                        progress_every: int = 100) -> LigeiaResult:
    """
    Corre el escenario completo y devuelve todos los diagnosticos.

    El tiempo de integracion se fija por el RECORRIDO del frente: cruza el
    dominio completo mas dos semianchos de margen a cada lado, para que la
    perturbacion entre y salga por entero. Es la misma politica de "recorrido
    fijo" del Paper 1.

    progress : callable(info: dict) -> None, invocado cada `progress_every`
        pasos de tiempo. `info` trae n_step, t, t_end, fraccion, dt, max|zeta|
        y la deriva de masa. EXISTE PARA PODER DISTINGUIR "LENTO" DE "MUERTO":
        una corrida larga sin latido observable es indistinguible de un cuelgue,
        que es exactamente donde quedo la corrida de 18 h que murio sin dejar
        rastro. No tiene coste apreciable: una llamada cada `progress_every`
        pasos.
    """
    cfg = cfg or LigeiaConfig()
    dom, bathy = build_ligeia_domain(cfg)
    X, Y = np.meshgrid(dom.x, dom.y)

    # El frente arranca fuera del dominio y lo cruza entero.
    recorrido = cfg.LX + 6.0 * cfg.sigma
    x0 = -3.0 * cfg.sigma
    t_end = recorrido / cfg.storm_speed

    presion = MovingPressureFront2D(amplitude=cfg.delta_p, sigma=cfg.sigma,
                                    speed=cfg.storm_speed,
                                    heading_deg=cfg.heading_deg,
                                    x0=x0, y0=0.5 * cfg.LY)
    forz = ForcingBundle(pressure=presion,
                         friction=LinearFriction(r=cfg.friction_r),
                         coriolis_enabled=cfg.coriolis)
    scfg = SolverConfig(cfl=cfg.cfl, min_depth=cfg.min_depth,
                        bc={"west": "transmissive", "east": "transmissive",
                            "south": "transmissive", "north": "transmissive"})

    s = NativeFVMSolver2D(order=2, wetting_drying=True)
    # zeta = 0 donde hay agua; donde el fondo emerge, h = 0 (zeta = -h0).
    zeta0 = np.where(bathy.depth > 0.0, 0.0, -bathy.depth)
    cero = np.zeros_like(zeta0)
    s.initialize(dom, State(zeta0, cero.copy(), cero.copy()), forz, scfg)

    envolvente = None
    hist_t, hist_max, hist_masa = [], [], []
    intervalo = t_end / max(cfg.n_envelope_samples, 1)
    siguiente = 0.0
    n_paso = 0
    while s.state.t < t_end:
        dt = min(s.compute_stable_dt(), t_end - s.state.t)
        if dt <= 0.0:
            break
        s.step(dt)
        n_paso += 1
        st = s.state
        if progress is not None and n_paso % progress_every == 0:
            d_now = s.diagnostics()
            # OJO: diagnostics()['zeta_max_abs'] recorre TODO el dominio, y en
            # las celdas secas el motor almacena zeta = -h0, que es la COTA DEL
            # TERRENO (hasta 179 m en las esquinas de Ligeia). Ese valor es
            # constante y como latido no informa de nada. El maximo util es el
            # de las celdas MOJADAS, que es la ola.
            mojado = (st.zeta + bathy.depth) > cfg.min_depth
            zeta_mojado = (float(np.max(np.abs(st.zeta[mojado])))
                           if np.any(mojado) else 0.0)
            progress({"n_step": n_paso, "t": st.t, "t_end": t_end,
                      "fraction": st.t / t_end, "dt": dt,
                      "zeta_max_wet": zeta_mojado,
                      "zeta_max_abs_all": d_now["zeta_max_abs"],
                      "mass_rel_signed": d_now["mass_rel_signed"],
                      "is_finite": d_now["is_finite"]})
        # La envolvente solo cuenta donde hay agua: en tierra seca zeta vale
        # -h0 por convencion y no es una elevacion fisica.
        mojado = (st.zeta + bathy.depth) > cfg.min_depth
        z = np.where(mojado, st.zeta, 0.0)
        envolvente = track_max_envelope(envolvente, z)
        if st.t >= siguiente:
            d = s.diagnostics()
            hist_t.append(st.t)
            hist_max.append(float(np.max(np.abs(z))))
            hist_masa.append(d["mass_rel_signed"])
            siguiente = st.t + intervalo
            if verbose:
                print(f"    t = {st.t/3600:7.2f} h   max|zeta| = "
                      f"{hist_max[-1]:.4f} m   masa = {hist_masa[-1]:+.2e}")

    ib = abs(inverse_barometer(cfg.delta_p))
    amp = envolvente / ib
    f = froude_field(cfg.storm_speed, bathy.depth, min_depth=cfg.min_depth)

    # Run-up sobre la ENVOLVENTE, no sobre el instante final.
    runup = compute_runup(envolvente, bathy.depth, X, Y, cfg.dx, cfg.dx,
                          min_depth=cfg.min_depth)
    reson = resonant_region_summary(cfg.storm_speed, bathy.depth,
                                    cell_area=cfg.dx * cfg.dx,
                                    min_depth=cfg.min_depth)
    reson["resonant_depth_band_observed_m"] = resonant_depth_band()
    reson["mask"] = resonance_mask(f, 0.1) & (bathy.depth > cfg.min_depth)

    return LigeiaResult(
        x=dom.x, y=dom.y, depth=bathy.depth, envelope=envolvente,
        amplification=amp, froude=f, final_state=s.state, bathymetry=bathy,
        runup=runup, resonance=reson, diagnostics=s.diagnostics(), config=cfg,
        history={"t": np.array(hist_t), "max_zeta": np.array(hist_max),
                 "mass_rel_signed": np.array(hist_masa)})
