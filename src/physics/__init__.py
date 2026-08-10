"""Diagnosticos fisicos neutrales al motor."""

from .proudman import (  # noqa: F401
    atmospheric_froude, resonance_speed, static_ib_response,
    proudman_amplification, amplification_from_field, bound_amplification,
    resonant_growth_estimate, summary,
    froude_field, resonance_mask, resonant_depth, resonant_depth_band,
    resonant_region_summary,
    amplification_depth_profile, classify_field_shape,
)
from .wetting_drying import (  # noqa: F401
    WettingDryingConfig, classify_cells, front_mask,
    desingularized_velocity_kp, dry_out,
)
from .runup import (  # noqa: F401
    greens_law_amplitude, shoaling_factor, RunupResult, compute_runup,
    shoreline_mask, track_max_envelope,
)
