"""Optimizer backend dispatch with shared final-candidate uniqueness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from . import factory as _factory
from .candidate_uniqueness import ensure_unique_candidates
from .configs import OptimizeConfig as _BaseOptimizeConfig
from .optimizer_support import (
    _InternalMixedOptimizerName,
    _force_sequential_for_kronecker,
    _optimizer_name,
    _resolve_thompson_sampling_target,
)

OptimizeBackend = Callable[..., tuple[Any, Any]]
_BASE_OPTIMIZE_CANDIDATES = _factory.optimize_candidates


def _common_kwargs(acqf: Any, bounds: Any, config: _BaseOptimizeConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": config.q,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "return_best_only": config.return_best_only,
        "sequential": config.sequential,
    }
    post_processing_func = _factory._build_post_processing_func(config, bounds)
    if post_processing_func is not None:
        kwargs["post_processing_func"] = post_processing_func
    if config.fixed_features is not None:
        kwargs["fixed_features"] = config.fixed_features
    if config.inequality_constraints is not None:
        kwargs["inequality_constraints"] = config.inequality_constraints
    if config.equality_constraints is not None:
        kwargs["equality_constraints"] = config.equality_constraints
    kwargs.update(config.optimizer_kwargs)
    return kwargs


def _optimize_candidates_once(
    acqf: Any,
    bounds: Any,
    config: _BaseOptimizeConfig,
    *,
    base_optimize_candidates: OptimizeBackend | None = None,
) -> tuple[Any, Any]:
    """Dispatch one optimizer call without final duplicate refill.

    ``base_optimize_candidates`` is an explicit dependency-injection hook used
    by compatibility tests and custom callers. It avoids runtime mutation of
    module or class methods.
    """

    if bounds is None:
        raise ValueError("bounds is required to optimize candidates.")
    config = _force_sequential_for_kronecker(config, acqf)
    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer(acqf, bounds, config)

    name = _optimizer_name(str(optimizer))
    if name in {"evo", "evo_mixed", "optimize_acqf_evo_mixed"}:
        from bochan.optim import optimize_acqf_evo

        return optimize_acqf_evo(**_common_kwargs(acqf, bounds, config))

    if name in {"torch", "torch_mixed", "optimize_acqf_torch_mixed"}:
        from bochan.optim import optimize_acqf_torch

        return optimize_acqf_torch(**_common_kwargs(acqf, bounds, config))

    if name in {"nsgaii", "optimize_acqf_nsgaii"}:
        from bochan.optim import optimize_acqf_nsgaii

        return optimize_acqf_nsgaii(**_common_kwargs(acqf, bounds, config))

    if name in {"thompson_sampling", "thompson_sampling_mixed"}:
        from bochan.optim import optimize_thompson_sampling

        kwargs = _common_kwargs(acqf, bounds, config)
        kwargs.pop("acq_function", None)
        kwargs["model"] = getattr(acqf, "model", None)
        kwargs["target"] = _resolve_thompson_sampling_target(acqf, config)
        return optimize_thompson_sampling(**kwargs)

    if name in {"optimize_acqf_mixed"}:
        from botorch.optim import optimize_acqf_mixed

        kwargs = _common_kwargs(acqf, bounds, config)
        kwargs.pop("fixed_features", None)
        kwargs["fixed_features_list"] = config.fixed_features_list
        return optimize_acqf_mixed(**kwargs)

    if name == "optimize_acqf":
        backend = base_optimize_candidates or _BASE_OPTIMIZE_CANDIDATES
        return backend(acqf, bounds, config)

    raise ValueError(f"Unknown optimizer backend: {optimizer!r}.")


def optimize_candidates(
    acqf: Any,
    bounds: Any,
    config: _BaseOptimizeConfig,
    *,
    base_optimize_candidates: OptimizeBackend | None = None,
) -> tuple[Any, Any]:
    """Optimize candidates and enforce shared final-candidate uniqueness."""

    candidates, values = _optimize_candidates_once(
        acqf,
        bounds,
        config,
        base_optimize_candidates=base_optimize_candidates,
    )
    return ensure_unique_candidates(
        candidates=candidates,
        values=values,
        acqf=acqf,
        bounds=bounds,
        config=config,
        optimize_once=lambda current_config: _optimize_candidates_once(
            acqf,
            bounds,
            current_config,
            base_optimize_candidates=base_optimize_candidates,
        ),
    )
