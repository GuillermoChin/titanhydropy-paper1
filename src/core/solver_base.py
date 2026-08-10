"""
core/solver_base.py
Contrato abstracto de motor hidrodinamico (engine-agnostic) para TitanHydroPy.

Objetivo de diseno:
    Las capas de fisica, forzamiento y diagnostico NO deben saber que motor
    numerico corre por debajo (motor propio FVM, GeoClaw, Thetis, etc.).
    Este modulo define la frontera entre "la ciencia" (neutral al motor) y
    "el numerico" (especifico de cada motor).

    - El forzamiento se expresa como CAMPOS FISICOS neutrales (presion, viento,
      friccion), no como terminos-fuente de un esquema concreto. Cada adaptador
      de motor traduce esos campos a su propia maquinaria interna.
    - El estado se expresa en variables primitivas (zeta, u, v). Cada motor
      convierte internamente a variables conservadas (h, hu, hv) si aplica.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


# ---------------------------------------------------------------------------
# 1. Contenedores de datos neutrales al motor
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Estado hidrodinamico en variables primitivas, promediado en la vertical."""
    zeta: np.ndarray   # elevacion de superficie libre respecto al reposo [m]
    u: np.ndarray      # velocidad horizontal, componente x [m/s]
    v: np.ndarray      # velocidad horizontal, componente y [m/s]
    t: float = 0.0     # tiempo de simulacion [s]

    def copy(self) -> "State":
        return State(self.zeta.copy(), self.u.copy(), self.v.copy(), self.t)


@dataclass
class Domain:
    """Geometria y batimetria del dominio (neutral al motor)."""
    x: np.ndarray            # coordenadas de celda/nodo, eje x [m]
    y: np.ndarray            # coordenadas de celda/nodo, eje y [m]
    bathymetry: np.ndarray   # profundidad de reposo h0 (>0 en zona liquida) [m]
    lat0_deg: float          # latitud de referencia para Coriolis [grados]
    # NOTA: la convencion de signo de la batimetria se documenta aqui de forma
    # explicita para evitar el error clasico de fondo/elevacion invertidos.


@dataclass
class SolverConfig:
    """Parametros numericos. Todo supuesto numerico queda explicito aqui."""
    cfl: float = 0.45               # numero de Courant objetivo
    scheme: str = "hllc"            # solver de Riemann: 'hllc' | 'roe'
    time_integrator: str = "ssprk3" # integrador temporal
    bc: dict = field(default_factory=lambda: {
        "north": "reflective", "south": "reflective",
        "east": "reflective",  "west": "reflective",
    })
    min_depth: float = 1e-3         # umbral de profundidad para wetting-drying [m]
    engine: str = "native"          # 'native' | 'geoclaw' | 'thetis'
    max_dt: float | None = None     # cota superior opcional del paso [s]


# ---------------------------------------------------------------------------
# 2. Protocolos de forzamiento (campos fisicos neutrales al motor)
# ---------------------------------------------------------------------------

@runtime_checkable
class PressureField(Protocol):
    """Campo de presion atmosferica superficial P(x, y, t) [Pa]."""
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray: ...


@runtime_checkable
class WindField(Protocol):
    """Campo de viento a 10 m: devuelve (u10, v10) [m/s]."""
    def evaluate(self, x: np.ndarray, y: np.ndarray,
                 t: float) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class FrictionModel(Protocol):
    """Ley de friccion de fondo. Devuelve el arrastre por unidad de masa (x, y)."""
    def stress(self, state: State,
               depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class ForcingBundle:
    """Agrupa los forzamientos activos. Cualquiera puede ser None (desactivado)."""
    pressure: PressureField | None = None
    wind: WindField | None = None
    friction: FrictionModel | None = None
    coriolis_enabled: bool = True
    # El gradiente de presion, el viento y Coriolis se activan/desactivan aqui
    # para correr los experimentos A/B/C (solo viento / solo presion / acoplado).


# ---------------------------------------------------------------------------
# 3. Contrato abstracto del motor
# ---------------------------------------------------------------------------

class Solver(ABC):
    """
    Interfaz que todo motor hidrodinamico debe implementar.
    Un adaptador concreto (NativeFVMSolver, GeoClawAdapter, ThetisAdapter)
    hereda de esta clase y traduce los campos fisicos a su maquinaria interna.
    """

    @abstractmethod
    def initialize(self, domain: Domain, initial_state: State,
                   forcing: ForcingBundle, config: SolverConfig) -> None:
        """Prepara el motor: mallas internas, memoria, condiciones de frontera."""
        ...

    @abstractmethod
    def compute_stable_dt(self) -> float:
        """Paso de tiempo estable segun CFL (celeridad sqrt(g*h) + flujo)."""
        ...

    @abstractmethod
    def step(self, dt: float | None = None) -> float:
        """
        Avanza UN paso. Si dt es None, usa compute_stable_dt().
        Devuelve el dt efectivamente usado [s].
        """
        ...

    def advance(self, t_end: float) -> None:
        """Integra hasta t_end encadenando step() con recorte de dt final."""
        while self.state.t < t_end:
            dt = self.compute_stable_dt()
            if self.config.max_dt is not None:
                dt = min(dt, self.config.max_dt)
            dt = min(dt, t_end - self.state.t)
            self.step(dt)

    @property
    @abstractmethod
    def state(self) -> State:
        """Estado actual (lectura): zeta, u, v, t vigentes."""
        ...

    @property
    @abstractmethod
    def config(self) -> SolverConfig:
        ...

    @abstractmethod
    def diagnostics(self) -> dict:
        """
        Invariantes para la validacion Nivel A: masa total, momento, energia
        y flags de estabilidad. Consumido por validation/conservation.py.
        """
        ...
