"""NSGA-II optimization backend and support utilities."""

from .adapter import (
    equality_constraints_to_inequality_constraints,
    optimize_acqf_nsgaii,
    validate_discrete_choices,
)
from .strategy import NSGAIIStrategy, build_nsgaii_strategy

__all__ = [
    "NSGAIIStrategy",
    "build_nsgaii_strategy",
    "equality_constraints_to_inequality_constraints",
    "optimize_acqf_nsgaii",
    "validate_discrete_choices",
]
