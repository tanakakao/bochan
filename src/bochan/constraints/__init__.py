"""BoTorch candidate post-processing utilities.

Typical use:

    post_processing_func = make_grid_k_sparse_post_processing_func(...)
    candidates, acq_value = optimize_acqf(..., post_processing_func=post_processing_func)
"""

from .constraints import (
    LinearConstraint,
    convert_legacy_constraints,
    linear_constraint_violations,
    make_linear_constraint_repair_func,
    project_linear_constraints,
)
from .ksparse import (
    diversify_within_q,
    enforce_sum_on_support,
    expand_categorical_features,
    generate_k_sparse_initial_conditions,
    k_exact_sparse_transform_factory,
    make_k_sparse_linear_constraints_repair,
    make_k_sparse_post_processing_func,
)
from .postprocess import (
    compose_post_processing_funcs,
    make_grid_k_sparse_post_processing_func,
    validate_post_processed_candidates,
)
from .rounding import (
    grid_residual,
    identity_post_processing_func,
    make_grid_rounding_post_processing_func,
    make_grid_rounding_with_linear_repair_func,
    round_numeric,
    round_numeric_preserve_sparse_support,
)

__all__ = [
    "LinearConstraint",
    "convert_legacy_constraints",
    "linear_constraint_violations",
    "make_linear_constraint_repair_func",
    "project_linear_constraints",
    "diversify_within_q",
    "enforce_sum_on_support",
    "expand_categorical_features",
    "generate_k_sparse_initial_conditions",
    "k_exact_sparse_transform_factory",
    "make_k_sparse_linear_constraints_repair",
    "make_k_sparse_post_processing_func",
    "compose_post_processing_funcs",
    "make_grid_k_sparse_post_processing_func",
    "validate_post_processed_candidates",
    "grid_residual",
    "identity_post_processing_func",
    "make_grid_rounding_post_processing_func",
    "make_grid_rounding_with_linear_repair_func",
    "round_numeric",
    "round_numeric_preserve_sparse_support",
]
