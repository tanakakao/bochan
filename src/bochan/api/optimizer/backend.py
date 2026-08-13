"""Low-level candidate optimizer backend dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..configs import OptimizeConfig
from ..configs.optimizer_names import _optimizer_name
from ..support.callables import _filter_kwargs_for_callable


def _build_post_processing_func(config: OptimizeConfig, bounds: Any) -> Callable[..., Any] | None:
    """Resolve explicit or config-driven candidate repair post-processing."""
    if config.post_processing_func is not None:
        return config.post_processing_func
    repair = config.repair_config
    if repair is None:
        return None

    from bochan.constraints.postprocess import make_grid_k_sparse_post_processing_func

    repair_bounds = repair.bounds if repair.bounds is not None else bounds
    if repair_bounds is None:
        raise ValueError("bounds is required when OptimizeConfig.repair_config is specified.")

    equality_constraints = repair.equality_constraints
    if equality_constraints is None:
        equality_constraints = config.equality_constraints

    inequality_constraints = repair.inequality_constraints
    if inequality_constraints is None:
        inequality_constraints = config.inequality_constraints

    fixed_features = repair.fixed_features
    if fixed_features is None:
        fixed_features = config.fixed_features

    return make_grid_k_sparse_post_processing_func(
        bounds=repair_bounds,
        numeric_indices=repair.numeric_indices,
        steps=repair.steps,
        comp_idx=repair.comp_idx,
        k=repair.k,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        inequality_sense=repair.inequality_sense,
        fixed_features=fixed_features,
        final_sum_constraint=repair.final_sum_constraint,
        diversify=repair.diversify,
        diversify_kwargs=repair.diversify_kwargs,
        score=repair.score,
        support_selection=repair.support_selection,
        sample_tau=repair.sample_tau,
        sample_eps=repair.sample_eps,
        generator=repair.generator,
        max_iters=repair.max_iters,
        num_alternations=repair.num_alternations,
        final_priority=repair.final_priority,
        support_eps=repair.support_eps,
    )


def _with_sequential(common_kwargs: dict[str, Any], config: OptimizeConfig) -> dict[str, Any]:
    kwargs = dict(common_kwargs)
    kwargs["sequential"] = config.sequential
    return kwargs


def _merge_fixed_features(base: Mapping[int, float] | None, extra: Mapping[int, float] | None) -> dict[int, float]:
    """Merge fixed-feature dictionaries with ``extra`` taking priority."""
    merged = {int(k): float(v) for k, v in (base or {}).items()}
    for key, value in (extra or {}).items():
        merged[int(key)] = float(value)
    return merged


def _merge_fixed_features_list(
    fixed_features: Mapping[int, float] | None,
    fixed_features_list: Sequence[Mapping[int, float]] | None,
) -> list[dict[int, float]] | None:
    """Apply global fixed features to every mixed fixed-feature assignment."""
    base = _merge_fixed_features(fixed_features, None)
    if fixed_features_list is None:
        return [base] if base else None
    if len(fixed_features_list) == 0:
        raise ValueError("fixed_features_list must not be empty when supplied.")
    return [_merge_fixed_features(base, item) for item in fixed_features_list]


def optimize_candidates(acqf: Any, bounds: Any, config: OptimizeConfig) -> tuple[Any, Any]:
    if bounds is None:
        raise ValueError("bounds must be provided.")

    common_kwargs = {
        "acq_function": acqf,
        "bounds": bounds,
        "q": config.q,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "return_best_only": config.return_best_only,
    }

    post_processing_func = _build_post_processing_func(config, bounds)
    if post_processing_func is not None:
        common_kwargs["post_processing_func"] = post_processing_func

    if config.fixed_features is not None:
        common_kwargs["fixed_features"] = config.fixed_features
    if config.inequality_constraints is not None:
        common_kwargs["inequality_constraints"] = config.inequality_constraints
    if config.equality_constraints is not None:
        common_kwargs["equality_constraints"] = config.equality_constraints
    common_kwargs.update(config.optimizer_kwargs)

    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimizer, kwargs)
        return optimizer(**kwargs)

    optimizer_name = _optimizer_name(str(optimizer))

    if optimizer_name == "optimize_acqf":
        from botorch.optim import optimize_acqf

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf, kwargs)
        return optimize_acqf(**kwargs)

    if optimizer_name == "optimize_acqf_mixed":
        from botorch.optim import optimize_acqf_mixed

        kwargs = _with_sequential(common_kwargs, config)
        merged_fixed_features_list = _merge_fixed_features_list(config.fixed_features, config.fixed_features_list)
        if merged_fixed_features_list is None:
            raise ValueError(
                "OptimizeConfig.fixed_features_list or OptimizeConfig.fixed_features "
                "is required when optimizer='optimize_acqf_mixed'."
            )
        kwargs.pop("fixed_features", None)
        kwargs["fixed_features_list"] = merged_fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_mixed, kwargs)
        return optimize_acqf_mixed(**kwargs)

    if optimizer_name in {"evo", "optimize_acqf_evo"}:
        from bochan.optim import optimize_acqf_evo

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf_evo, kwargs)
        return optimize_acqf_evo(**kwargs)

    if optimizer_name in {"torch", "optimize_acqf_torch"}:
        from bochan.optim import optimize_acqf_torch

        kwargs = _with_sequential(common_kwargs, config)
        kwargs = _filter_kwargs_for_callable(optimize_acqf_torch, kwargs)
        return optimize_acqf_torch(**kwargs)

    if optimizer_name in {"evo_mixed", "optimize_acqf_evo_mixed"}:
        from bochan.optim import optimize_acqf_evo_mixed

        kwargs = _with_sequential(common_kwargs, config)
        if config.fixed_features_list is not None:
            kwargs["fixed_features_list"] = config.fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_evo_mixed, kwargs)
        return optimize_acqf_evo_mixed(**kwargs)

    if optimizer_name in {"torch_mixed", "optimize_acqf_torch_mixed"}:
        from bochan.optim import optimize_acqf_torch_mixed

        kwargs = _with_sequential(common_kwargs, config)
        if config.fixed_features_list is not None:
            kwargs["fixed_features_list"] = config.fixed_features_list
        kwargs = _filter_kwargs_for_callable(optimize_acqf_torch_mixed, kwargs)
        return optimize_acqf_torch_mixed(**kwargs)

    raise ValueError(f"Unknown optimizer: {optimizer}")


__all__ = ["optimize_candidates"]
