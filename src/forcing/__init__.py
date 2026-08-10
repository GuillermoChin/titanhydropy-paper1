"""Campos de forzamiento neutrales al motor (presion, viento, friccion)."""

from .atmospheric import (  # noqa: F401
    UniformPressure, MovingGaussianPressure, StaticGaussianPressure,
)
from .friction import NoFriction, LinearFriction, ManningFriction  # noqa: F401
from .wind import NoWind, UniformWind  # noqa: F401
