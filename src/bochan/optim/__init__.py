"""Optimizer wrappers."""

from .standard import optimize_acqf_k_sparse, optimize_acqf_mixed_k_sparse
from .evo import (
    candidate_transform_mixed_factory,
    optimize_acqf_evo,
    optimize_acqf_evo_k_sparse,
    optimize_acqf_evo_mixed,
    optimize_acqf_evo_mixed_k_sparse,
)
from .llm import optimize_acqf_llm_candidate_set
from .nsgaii_adapter import (
    equality_constraints_to_inequality_constraints,
    optimize_acqf_nsgaii,
    validate_discrete_choices,
)
from .nsgaii_legacy_linear_constraint_compat import (
    apply_legacy_nsgaii_linear_constraint_compat,
)
from .thompson_sampling_adapter import (
    optimize_thompson_sampling,
    optimize_thompson_sampling_mixed,
)
from .torch_opt import (
    optimize_acqf_torch,
    optimize_acqf_torch_mixed,
    optimize_acqf_torch_k_sparse,
    optimize_acqf_torch_mixed_k_sparse,
)

apply_legacy_nsgaii_linear_constraint_compat()

__all__ = [
    "optimize_acqf_k_sparse",
    "optimize_acqf_mixed_k_sparse",
    "candidate_transform_mixed_factory",
    "optimize_acqf_evo",
    "optimize_acqf_evo_k_sparse",
    "optimize_acqf_evo_mixed",
    "optimize_acqf_evo_mixed_k_sparse",
    "optimize_acqf_llm_candidate_set",
    "equality_constraints_to_inequality_constraints",
    "optimize_acqf_nsgaii",
    "validate_discrete_choices",
    "optimize_thompson_sampling",
    "optimize_thompson_sampling_mixed",
    "optimize_acqf_torch",
    "optimize_acqf_torch_mixed",
    "optimize_acqf_torch_k_sparse",
    "optimize_acqf_torch_mixed_k_sparse",
]
