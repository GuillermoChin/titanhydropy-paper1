"""
validation/conservation.py
==========================
Monitor de invariantes (Nivel A). Consume Solver.diagnostics().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.solver_base import Solver

__all__ = ["ConservationMonitor", "convergence_order"]


@dataclass
class ConservationMonitor:
    """
    Registra masa, momento y energia a lo largo de una corrida.

    `mass_drift` es el error relativo maximo de masa respecto al instante
    inicial; en dominio cerrado (fronteras reflectivas) y sin forzamiento debe
    quedar al nivel del redondeo acumulado.
    `energy_growth` es el crecimiento relativo maximo de la energia respecto a
    la inicial; el esquema es disipativo, asi que un valor positivo significativo
    delata inestabilidad, no fisica.
    """
    records: list[dict] = field(default_factory=list)

    def sample(self, solver: Solver) -> dict:
        d = solver.diagnostics()
        self.records.append(d)
        return d

    def _series(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records], dtype=float)

    @property
    def mass_drift(self) -> float:
        m = self._series("mass")
        return float(np.max(np.abs(m - m[0])) / max(abs(m[0]), 1e-300))

    @property
    def energy_growth(self) -> float:
        e = self._series("energy")
        return float((np.max(e) - e[0]) / max(abs(e[0]), 1e-300))

    @property
    def all_finite(self) -> bool:
        return all(r["is_finite"] for r in self.records)

    def report(self) -> str:
        return (f"muestras={len(self.records)}  "
                f"deriva_masa={self.mass_drift:.3e}  "
                f"crecimiento_energia={self.energy_growth:+.3e}  "
                f"finito={self.all_finite}")


def convergence_order(resolutions: np.ndarray,
                      errors: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ordenes de convergencia entre refinamientos consecutivos y ajuste global.

    Devuelve (ordenes_por_par, pendiente_del_ajuste_log-log).
    resolutions = numero de celdas; errors = norma del error.
    """
    n = np.asarray(resolutions, dtype=float)
    e = np.asarray(errors, dtype=float)
    if n.size < 2:
        raise ValueError("Se necesitan al menos dos resoluciones.")
    pairwise = np.log(e[:-1] / e[1:]) / np.log(n[1:] / n[:-1])
    slope = float(-np.polyfit(np.log(n), np.log(e), 1)[0])
    return pairwise, slope
