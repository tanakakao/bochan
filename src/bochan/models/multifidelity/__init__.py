"""Cross-family extension point for shared multi-fidelity infrastructure.

Likelihood-specific multi-fidelity models remain owned by their regression or
classification family. Shared fidelity-axis transforms, adapters, and validation
should be placed here as they are introduced.
"""

from .configured import create_configured_fidelity_surrogate
from .factory import FidelityInputMode, create_fidelity_surrogate
from .optimization import (
    enumerate_discrete_fidelities_into_opt_config,
    merge_target_fidelities_into_opt_config,
    target_fidelity_fixed_features,
)
from .spec import FidelitySpec, ResolvedFidelitySpec

__all__ = [
    "FidelityInputMode",
    "FidelitySpec",
    "ResolvedFidelitySpec",
    "create_configured_fidelity_surrogate",
    "create_fidelity_surrogate",
    "enumerate_discrete_fidelities_into_opt_config",
    "merge_target_fidelities_into_opt_config",
    "target_fidelity_fixed_features",
]
