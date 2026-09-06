"""Cross-family extension point for shared multi-fidelity infrastructure.

Likelihood-specific multi-fidelity models remain owned by their regression or
classification family. Shared fidelity-axis transforms, adapters, and validation
should be placed here as they are introduced.
"""

from .configured import (
    create_configured_correlated_fidelity_surrogate,
    create_configured_fidelity_surrogate,
)
from .correlated import GaussianCorrelatedMultiFidelityGP
from .cost import (
    FidelityCostCallable,
    FidelityCostConfig,
    FidelityCostKind,
    build_fidelity_cost_utility,
    build_learned_fidelity_cost_model,
    evaluate_fidelity_cost_mean,
)
from .factory import FidelityInputMode, create_fidelity_surrogate
from .multioutput import (
    bind_shared_multifidelity_metadata,
    shared_multifidelity_metadata,
)
from .optimization import (
    enumerate_discrete_fidelities_into_opt_config,
    merge_target_fidelities_into_opt_config,
    prepare_continuous_fidelity_optimization,
    target_fidelity_fixed_features,
)
from .spec import FidelitySpec, ResolvedFidelitySpec

__all__ = [
    "FidelityCostCallable",
    "FidelityCostConfig",
    "FidelityCostKind",
    "FidelityInputMode",
    "FidelitySpec",
    "GaussianCorrelatedMultiFidelityGP",
    "ResolvedFidelitySpec",
    "bind_shared_multifidelity_metadata",
    "build_fidelity_cost_utility",
    "build_learned_fidelity_cost_model",
    "create_configured_correlated_fidelity_surrogate",
    "create_configured_fidelity_surrogate",
    "create_fidelity_surrogate",
    "enumerate_discrete_fidelities_into_opt_config",
    "evaluate_fidelity_cost_mean",
    "merge_target_fidelities_into_opt_config",
    "prepare_continuous_fidelity_optimization",
    "shared_multifidelity_metadata",
    "target_fidelity_fixed_features",
]
