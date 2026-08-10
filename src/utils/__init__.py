"""Utilidades transversales (orquestacion de corridas, normas de error)."""

from .runners import RunResult, run_to, l1_error, l2_error  # noqa: F401
from .visualizer import (  # noqa: F401
    plot_zeta_map, plot_froude_map, plot_amplification_curve,
    plot_bathymetry, plot_profile_comparison, matplotlib_available,
)
