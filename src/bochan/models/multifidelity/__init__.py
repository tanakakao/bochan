"""Cross-family extension point for shared multi-fidelity infrastructure.

Likelihood-specific multi-fidelity models remain owned by their regression or
classification family. Shared fidelity-axis transforms, adapters, validation,
and benchmark metrics live here.
"""

from .benchmark import (
    CostNormalizedTrace,
    best_objective_trace,
    cumulative_cost,
    hypervolume_regret_trace,
    hypervolume_trace,
    inference_hv_regret_cost_trace,
    multi_objective_cost_trace,
    single_objective_cost_trace,
)
from .configured import (
    create_configured_correlated_fidelity_surrogate,
    create_configured_fidelity_surrogate,
    create_configured_information_source_surrogate,
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
from .source import (
    GaussianMultiSourceGP,
    InformationSourceSpec,
    ResolvedInformationSourceSpec,
)
from .spec import FidelitySpec, ResolvedFidelitySpec
from .synthetic import (
    SyntheticMultiFidelityProblem,
    augmented_branin_problem,
    augmented_hartmann_problem,
    momf_branin_currin_problem,
)

__all__ = [
    "CostNormalizedTrace",
    "FidelityCostCallable",
    "FidelityCostConfig",
    "FidelityCostKind",
    "FidelityInputMode",
    "FidelitySpec",
    "GaussianCorrelatedMultiFidelityGP",
    "GaussianMultiSourceGP",
    "InformationSourceSpec",
    "ResolvedFidelitySpec",
    "ResolvedInformationSourceSpec",
    "SyntheticMultiFidelityProblem",
    "augmented_branin_problem",
    "augmented_hartmann_problem",
    "best_objective_trace",
    "bind_shared_multifidelity_metadata",
    "build_fidelity_cost_utility",
    "build_learned_fidelity_cost_model",
    "create_configured_correlated_fidelity_surrogate",
    "create_configured_fidelity_surrogate",
    "create_configured_information_source_surrogate",
    "create_fidelity_surrogate",
    "cumulative_cost",
    "enumerate_discrete_fidelities_into_opt_config",
    "evaluate_fidelity_cost_mean",
    "hypervolume_regret_trace",
    "hypervolume_trace",
    "inference_hv_regret_cost_trace",
    "merge_target_fidelities_into_opt_config",
    "momf_branin_currin_problem",
    "multi_objective_cost_trace",
    "prepare_continuous_fidelity_optimization",
    "shared_multifidelity_metadata",
    "single_objective_cost_trace",
    "target_fidelity_fixed_features",
]
