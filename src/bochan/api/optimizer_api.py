# ruff: noqa: F401, I001
"""Canonical optimizer configuration and high-level dispatch helpers."""

from __future__ import annotations

from typing import Any

from .candidate_uniqueness import ensure_unique_candidates
from .configs import OptimizeConfig as _BaseOptimizeConfig
from .configs.optimize import (
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from . import optimizer_dispatch as _optimizer_dispatch
from .optimizer_dispatch import (
    _common_kwargs,
    _optimize_candidates_once,
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

# Preserve the existing test/customization seam without mutating a different
# module at runtime.  The value is passed explicitly to the shared dispatcher.
_BASE_OPTIMIZE_CANDIDATES = _optimizer_dispatch._BASE_OPTIMIZE_CANDIDATES


def optimize_candidates(
    acqf: Any,
    bounds: Any,
    config: _BaseOptimizeConfig,
) -> tuple[Any, Any]:
    """Dispatch using the explicitly configured base optimizer dependency."""

    return _optimizer_dispatch.optimize_candidates(
        acqf=acqf,
        bounds=bounds,
        config=config,
        base_optimize_candidates=_BASE_OPTIMIZE_CANDIDATES,
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
