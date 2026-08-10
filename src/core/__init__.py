"""Nucleo numerico: contrato de motor + implementacion nativa FVM."""

from .solver_base import (  # noqa: F401
    State, Domain, SolverConfig,
    PressureField, WindField, FrictionModel, ForcingBundle,
    Solver,
)
from .native_fvm import NativeFVMSolver  # noqa: F401
