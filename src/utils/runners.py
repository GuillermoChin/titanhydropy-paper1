"""
utils/runners.py
================
Utilidades de orquestacion: montar un solver y avanzarlo muestreando
invariantes. Concentra el codigo repetido de tests y experimentos para que las
compuertas no reimplementen el bucle temporal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.native_fvm import NativeFVMSolver
from core.solver_base import Domain, ForcingBundle, SolverConfig, State
from validation.conservation import ConservationMonitor

__all__ = ["RunResult", "run_to", "l1_error", "l2_error"]


@dataclass
class RunResult:
    """Resultado de una corrida: estado final, monitor y traza de zeta maximo."""
    solver: NativeFVMSolver
    monitor: ConservationMonitor
    zeta_peak_history: np.ndarray
    time_history: np.ndarray

    @property
    def state(self) -> State:
        return self.solver.state


def run_to(domain: Domain, initial: State, forcing: ForcingBundle,
           config: SolverConfig, t_end: float,
           fluid=None, g: float | None = None, order: int = 2,
           sample_every: int = 25) -> RunResult:
    """
    Integra hasta t_end con el motor nativo, muestreando invariantes cada
    `sample_every` pasos (y siempre en t=0 y al final).
    """
    kwargs = {}
    if fluid is not None:
        kwargs["fluid"] = fluid
    solver = NativeFVMSolver(g=g, order=order, **kwargs)
    solver.initialize(domain, initial, forcing, config)

    mon = ConservationMonitor()
    mon.sample(solver)
    peaks = [float(np.max(np.abs(solver.state.zeta)))]
    times = [solver.state.t]

    n = 0
    while solver.state.t < t_end:
        dt = solver.compute_stable_dt()
        if config.max_dt is not None:
            dt = min(dt, config.max_dt)
        dt = min(dt, t_end - solver.state.t)
        if dt <= 0.0:
            break
        solver.step(dt)
        n += 1
        if n % sample_every == 0:
            mon.sample(solver)
            peaks.append(float(np.max(np.abs(solver.state.zeta))))
            times.append(solver.state.t)

    mon.sample(solver)
    peaks.append(float(np.max(np.abs(solver.state.zeta))))
    times.append(solver.state.t)
    return RunResult(solver, mon, np.array(peaks), np.array(times))


def l1_error(numeric: np.ndarray, exact: np.ndarray, dx: float) -> float:
    """Norma L1 discreta del error: sum |e| dx."""
    return float(np.sum(np.abs(numeric - exact)) * dx)


def l2_error(numeric: np.ndarray, exact: np.ndarray, dx: float) -> float:
    """Norma L2 discreta del error: sqrt(sum e^2 dx)."""
    return float(np.sqrt(np.sum((numeric - exact) ** 2) * dx))
