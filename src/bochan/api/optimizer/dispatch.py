"""Optimizer backend dispatch with shared final-candidate uniqueness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from .. import factory as _factory
from ..candidate.uniqueness import ensure_unique_candidates
from ..configs import OptimizeConfig as _BaseOptimizeConfig
from ..configs.optimizer_names import _InternalMixedOptimizerName, _optimizer_name
from ..support.best_subset import optimize_best_subset_candidates, uses_best_subset
from ..support.multi_group_best_subset import (
    optimize_grouped_best_subset_candidates,
    uses_grouped_best_subset,
)
from .support import (
    _force_sequential_for_kronecker,
    _resolve_thompson_sampling_target,
)

OptimizeBackend = Callable[..., tuple[Any, Any]]
_BASE_OPTIMIZE_CANDIDATES = _factory.optimize_candidates


def _is_multifidelity_acquisition(acqf: Any) -> bool:
    if getattr(acqf, "_bochan_multifidelity_kind", None) is not None:
        return True
    return type(acqf).__name__ in {
        "qMultiFidelityKnowledgeGradient",
        "qMultiFidelityMaxValueEntropy",
        "qMultiFidelityHypervolumeKnowledgeGradient",
    }


def _acquisition_model(acqf: Any) -> Any | None:
    """Resolve the model from direct and one-level wrapped acquisitions."""

    model = getattr(acqf, "model", None)
    if model is not None:
        return model
    base_acqf = getattr(acqf, "base_acqf", None)
    if base_acqf is not None:
        model = getattr(base_acqf, "model", None)
        if model is not None:
            return model
    return getattr(acqf, "_bochan_thompson_model", None)


def _resolve_multifidelity_optimization(acqf: Any, bounds: Any, config: _BaseOptimizeConfig) -> _BaseOptimizeConfig:
    model = _acquisition_model(acqf)
    if model is None:
        return config
    from bochan.models.multifidelity.optimization import (
        enumerate_discrete_fidelities_into_opt_config,
        merge_target_fidelities_into_opt_config,
        prepare_continuous_fidelity_optimization,
    )

    discrete = getattr(config, "fidelity_values", None) is not None or getattr(
        config, "fidelity_assignments", None
    ) is not None
    continuous = bool(getattr(config, "optimize_fidelity", False))
    if _is_multifidelity_acquisition(acqf) and not (discrete or continuous):
        raise ValueError(
            "Multi-fidelity acquisitions require OptimizeConfig.fidelity_values / "
            "fidelity_assignments for discrete fidelity search or optimize_fidelity=True "
            "for continuous joint fidelity optimization."
        )

    if continuous:
        config = prepare_continuous_fidelity_optimization(
            config,
            model=model,
            bounds=bounds,
        )
    else:
        config = enumerate_discrete_fidelities_into_opt_config(
            config,
            model=model,
            bounds=bounds,
        )
    return merge_target_fidelities_into_opt_config(config, model=model)


def _has_inter_point_constraints(config: _BaseOptimizeConfig) -> bool:
    """Return whether optimizer constraints address explicit q-point indices."""

    for constraints in (config.inequality_constraints, config.equality_constraints):
        for item in constraints or ():
            indices = torch.as_tensor(item[0])
            if indices.ndim > 1:
                return True
    return False


def _prepare_mfhvkg_initializer(acqf: Any, config: _BaseOptimizeConfig) -> _BaseOptimizeConfig:
    """Let standard MF-HVKG use BoTorch's specialized one-shot initializer.

    bochan historically injected its generic Best Subset-compatible initializer
    for every one-shot acquisition. That initializer remains necessary for
    inter-point support constraints, but ordinary MF-HVKG should use BoTorch's
    ``gen_one_shot_hvkg_initial_conditions`` so fantasy Pareto sets receive the
    dedicated initialization strategy.
    """

    if getattr(acqf, "_bochan_multifidelity_kind", None) != "mfhvkg":
        return config
    if "ic_generator" in config.optimizer_kwargs or _has_inter_point_constraints(config):
        return config

    from botorch.optim.initializers import gen_one_shot_hvkg_initial_conditions

    optimizer_kwargs = dict(config.optimizer_kwargs)
    optimizer_kwargs["ic_generator"] = gen_one_shot_hvkg_initial_conditions
    return replace(config, optimizer_kwargs=optimizer_kwargs)


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
    if bounds is None:
        raise ValueError("bounds must be provided.")
    backend = base_optimize_candidates or _BASE_OPTIMIZE_CANDIDATES
    config = _force_sequential_for_kronecker(acqf, config)
    config = _prepare_mfhvkg_initializer(acqf, config)
    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return backend(acqf=acqf, bounds=bounds, config=config)

    name = _optimizer_name(str(optimizer))
    is_mixed = config.fixed_features_list is not None
    if name in {"optimize_acqf", "evo", "torch"} and is_mixed:
        mixed_name = {
            "optimize_acqf": "optimize_acqf_mixed",
            "evo": "evo_mixed",
            "torch": "torch_mixed",
        }[name]
        return backend(
            acqf=acqf,
            bounds=bounds,
            config=replace(config, optimizer=_InternalMixedOptimizerName(mixed_name)),
        )

    special = {
        "nsgaii",
        "optimize_acqf_nsgaii",
        "thompson_sampling",
        "optimize_thompson_sampling",
        "thompson_sampling_mixed",
        "optimize_thompson_sampling_mixed",
        "llm_candidate_set",
        "optimize_acqf_llm",
        "optimize_acqf_llm_candidate_set",
    }
    if name not in special:
        return backend(acqf=acqf, bounds=bounds, config=config)

    kwargs = _common_kwargs(acqf, bounds, config)
    if name in {"nsgaii", "optimize_acqf_nsgaii"}:
        from bochan.optim import optimize_acqf_nsgaii

        kwargs = _factory._filter_kwargs_for_callable(optimize_acqf_nsgaii, kwargs)
        return optimize_acqf_nsgaii(**kwargs)
    if name in {"llm_candidate_set", "optimize_acqf_llm", "optimize_acqf_llm_candidate_set"}:
        from bochan.optim import optimize_acqf_llm_candidate_set

        kwargs = _factory._filter_kwargs_for_callable(optimize_acqf_llm_candidate_set, kwargs)
        return optimize_acqf_llm_candidate_set(**kwargs)

    kwargs["acq_function"] = _resolve_thompson_sampling_target(acqf)
    use_mixed = name in {"thompson_sampling_mixed", "optimize_thompson_sampling_mixed"} or (
        name in {"thompson_sampling", "optimize_thompson_sampling"} and is_mixed
    )
    if use_mixed:
        from bochan.optim import optimize_thompson_sampling_mixed

        fixed_features_list = _factory._merge_fixed_features_list(config.fixed_features, config.fixed_features_list)
        if fixed_features_list is None:
            raise ValueError("fixed_features_list is required for mixed Thompson sampling.")
        kwargs.pop("fixed_features", None)
        kwargs["fixed_features_list"] = fixed_features_list
        kwargs = _factory._filter_kwargs_for_callable(optimize_thompson_sampling_mixed, kwargs)
        return optimize_thompson_sampling_mixed(**kwargs)

    from bochan.optim import optimize_thompson_sampling

    kwargs = _factory._filter_kwargs_for_callable(optimize_thompson_sampling, kwargs)
    return optimize_thompson_sampling(**kwargs)


def optimize_candidates(
    acqf: Any,
    bounds: Any,
    config: _BaseOptimizeConfig,
    *,
    base_optimize_candidates: OptimizeBackend | None = None,
) -> tuple[Any, Any]:
    """Dispatch an optimizer and enforce uniqueness on its final q-batch."""

    config = _resolve_multifidelity_optimization(acqf, bounds, config)

    if uses_grouped_best_subset(config):
        return optimize_grouped_best_subset_candidates(
            acqf=acqf,
            bounds=bounds,
            config=config,
            optimize_one=lambda *, acqf, bounds, config: optimize_candidates(
                acqf=acqf,
                bounds=bounds,
                config=config,
                base_optimize_candidates=base_optimize_candidates,
            ),
        )
    if uses_best_subset(config):
        return optimize_best_subset_candidates(
            acqf=acqf,
            bounds=bounds,
            config=config,
            optimize_one=lambda *, acqf, bounds, config: optimize_candidates(
                acqf=acqf,
                bounds=bounds,
                config=config,
                base_optimize_candidates=base_optimize_candidates,
            ),
        )

    def optimize_once(*, acqf: Any, bounds: Any, config: _BaseOptimizeConfig) -> tuple[Any, Any]:
        return _optimize_candidates_once(
            acqf=acqf,
            bounds=bounds,
            config=config,
            base_optimize_candidates=base_optimize_candidates,
        )

    candidates, acq_value = optimize_once(acqf=acqf, bounds=bounds, config=config)
    return ensure_unique_candidates(
        acqf=acqf,
        bounds=bounds,
        config=config,
        candidates=candidates,
        acq_value=acq_value,
        optimize_once=optimize_once,
    )
