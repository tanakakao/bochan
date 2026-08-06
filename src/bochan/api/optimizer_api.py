"""Canonical optimizer configuration and high-level dispatch helpers."""

from __future__ import annotations

from .candidate_uniqueness import ensure_unique_candidates
from .optimizer_config import (
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from .optimizer_dispatch import (
    _BASE_OPTIMIZE_CANDIDATES,
    _common_kwargs,
    _optimize_candidates_once,
    optimize_candidates,
)
from .optimizer_support import (
    EvolutionaryMethod,
    OptimizerName,
    _ALIASES,
    _CANONICAL_OPTIMIZERS,
    _EVOLUTIONARY_METHODS,
    _InternalMixedOptimizerName,
    _MIXED_OPTIMIZERS,
    _configured_thompson_sampling_model,
    _force_sequential_for_kronecker,
    _has_posterior,
    _has_thompson_sampling_context,
    _is_callable_acquisition,
    _optimizer_name,
    _resolve_thompson_sampling_target,
    _uses_kronecker_model,
)

__all__ = [
    "EvolutionaryMethod",
    "OptimizerName",
    "OptimizeConfig",
    "ensure_unique_candidates",
    "optimize_candidates",
    "resolve_optimizer_from_cat_dims",
    "uses_mixed_fixed_features",
]
