"""Compuertas de validacion: analitica, conservacion y coherencia fisica."""

from .analytic import dam_break_exact, star_depth  # noqa: F401
from .conservation import ConservationMonitor, convergence_order  # noqa: F401
from .titan_physical import (  # noqa: F401
    run_sanity, pending_constants, proudman_overlap,
    PhysicalCheck, run_all_checks, check_inverse_barometer,
    check_zeta_magnitude, check_velocity_magnitude,
)
