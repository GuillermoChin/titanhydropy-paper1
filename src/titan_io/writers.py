"""
titan_io/writers.py
===================
Persistencia de resultados. Formato .npz comprimido: autocontenido, sin
dependencias externas y con metadatos de procedencia embebidos.

REGLA DE PROCEDENCIA: toda salida guarda la lista de constantes en TO_VERIFY
vigentes en el momento de la corrida. Un fichero de resultados sin esa lista no
es utilizable para un manuscrito.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from constants.titan_params import audit_provenance
from core.solver_base import Solver, State

__all__ = ["TimeSeriesRecorder", "save_state", "load_state", "save_fields",
           "load_fields"]


class TimeSeriesRecorder:
    """Acumula instantaneas del estado a intervalos fijos."""

    def __init__(self, every_seconds: float) -> None:
        self.every = float(every_seconds)
        self.times: list[float] = []
        self.zeta: list[np.ndarray] = []
        self.u: list[np.ndarray] = []
        self._next = 0.0

    def maybe_sample(self, solver: Solver) -> bool:
        """Toma muestra si toca. Devuelve True si se registro."""
        st = solver.state
        if st.t + 1e-12 < self._next:
            return False
        self.times.append(st.t)
        self.zeta.append(st.zeta.copy())
        self.u.append(st.u.copy())
        self._next = st.t + self.every
        return True

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.array(self.times), np.array(self.zeta), np.array(self.u))


def save_state(path: str | Path, solver: Solver, **metadata) -> Path:
    """Guarda estado + diagnostico + procedencia en un .npz comprimido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    st = solver.state
    meta = dict(metadata)
    meta["diagnostics"] = solver.diagnostics()
    meta["to_verify_constants"] = audit_provenance()
    np.savez_compressed(path, zeta=st.zeta, u=st.u, v=st.v,
                        t=np.array([st.t]),
                        metadata_json=np.array(json.dumps(meta, default=str)))
    return path


def save_fields(path: str | Path, metadata: dict | None = None,
                **fields: np.ndarray) -> Path:
    """
    Guarda un conjunto de campos con nombre en un .npz comprimido, con la misma
    politica de procedencia que save_state: se embebe `audit_provenance()`.

    Existe porque no todo resultado tiene un Solver detras. Un diagnostico
    derivado (envolvente, mapa de amplificacion, campo de Froude) es un producto
    de primera clase y debe salir por titan_io, con su procedencia, igual que un
    estado.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    meta["to_verify_constants"] = audit_provenance()
    np.savez_compressed(
        path, metadata_json=np.array(json.dumps(meta, default=str)),
        **{k: np.asarray(v) for k, v in fields.items()})
    return path


def load_fields(path: str | Path) -> tuple[dict, dict]:
    """Lee un fichero escrito por save_fields. Devuelve (campos, metadatos)."""
    with np.load(Path(path), allow_pickle=False) as z:
        campos = {k: z[k] for k in z.files if k != "metadata_json"}
        meta = json.loads(str(z["metadata_json"]))
    return campos, meta


def load_state(path: str | Path) -> tuple[State, dict]:
    """Lee un fichero escrito por save_state."""
    with np.load(Path(path), allow_pickle=False) as z:
        st = State(zeta=z["zeta"], u=z["u"], v=z["v"], t=float(z["t"][0]))
        meta = json.loads(str(z["metadata_json"]))
    return st, meta
