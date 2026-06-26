"""Canonical optimizer configuration and high-level dispatch helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

from . import factory as _factory
from .configs import OptimizeConfig as _BaseOptimizeConfig

_BASE_OPTIMIZE_CANDIDATES = _factory.optimize_candidates

EvolutionaryMethod = Literal["ga", "pso", "sa", "cmaes"]
OptimizerName = Literal[
    "optimize_acqf",
    "evo",
    "ga",
    "pso",
    "sa",
    "cmaes",
    "torch",
    "nsgaii",
    "thompson_sampling",
]

_CANONICAL_OPTIMIZERS = {
    "optimize_acqf",
    "evo",
    "torch",
    "nsgaii",
    "thompson_sampling",
}
_EVOLUTIONARY_METHODS = {"ga", "pso", "sa", "cmaes"}
_ALIASES = {
    "optimize_acqf_mixed": "optimize_acqf",
    "optimize_acqf_evo": "evo",
    "evo_mixed": "evo",
    "optimize_acqf_evo_mixed": "evo",
    "optimize_acqf_torch": "torch",
    "torch_mixed": "torch",
    "optimize_acqf_torch_mixed": "torch",
    "optimize_acqf_nsgaii": "nsgaii",
    "optimize_thompson_sampling": "thompson_sampling",
    "optimize_thompson_sampling_mixed": "thompson_sampling",
    "thompson": "thompson_sampling",
}


def _optimizer_name(optimizer: str) -> str:
    return optimizer.replace("-", "_").lower()


@dataclass
class OptimizeConfig(_BaseOptimizeConfig):
    """Candidate optimization configuration using backend-family names.

    Mixed/non-mixed implementations are selected automatically. Evolutionary
    backends may be selected with ``optimizer="evo"`` plus ``evo_method``, or
    directly with ``optimizer="ga"``, ``"pso"``, ``"sa"``, or ``"cmaes"``.

    CMA-ES only optimizes one point at a time. Therefore, when its effective
    method is ``cmaes`` and ``q > 1``, ``sequential`` is enabled automatically.
    """

    optimizer: OptimizerName | str | Callable[..., Any] = "optimize_acqf"
    evo_method: EvolutionaryMethod = "ga"

    def __post_init__(self) -> None:
        if callable(self.optimizer) and not isinstance(self.optimizer, str):
            return

        raw_name = _optimizer_name(str(self.optimizer))
        name = _ALIASES.get(raw_name, raw_name)
        if name in _EVOLUTIONARY_METHODS:
            self.evo_method = name  # type: ignore[assignment]
            name = "evo"
        if name not in _CANONICAL_OPTIMIZERS:
            valid = sorted(_CANONICAL_OPTIMIZERS | _EVOLUTIONARY_METHODS)
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}. Expected one of {valid}.")

        self.optimizer = name
        self.optimizer_kwargs = dict(self.optimizer_kwargs)
        if name == "evo":
            effective_method = _optimizer_name(
                str(self.optimizer_kwargs.setdefault("method", self.evo_method))
            )
            if effective_method not in _EVOLUTIONARY_METHODS:
                raise ValueError(
                    f"Unknown evolutionary method: {effective_method!r}. "
                    f"Expected one of {sorted(_EVOLUTIONARY_METHODS)}."
                )
            self.evo_method = effective_method  # type: ignore[assignment]
            if effective_method == "cmaes" and self.q > 1:
                self.sequential = True


def resolve_optimizer_from_cat_dims(
    *,
    opt_config: _BaseOptimizeConfig,
    cat_dims: Sequence[int] | None,
) -> _BaseOptimizeConfig:
    """Resolve canonical backend names to mixed implementations."""

    if not cat_dims:
        return opt_config
    optimizer = opt_config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return opt_config

    mixed_name = {
        "optimize_acqf": "optimize_acqf_mixed",
        "evo": "evo_mixed",
        "torch": "torch_mixed",
        "thompson_sampling": "thompson_sampling_mixed",
    }.get(_optimizer_name(str(optimizer)))
    return opt_config if mixed_name is None else replace(opt_config, optimizer=mixed_name)


def uses_mixed_fixed_features(optimizer: Any) -> bool:
    """Return whether the backend needs categorical fixed-feature enumeration."""

    if callable(optimizer) and not isinstance(optimizer, str):
        return False
    return _optimizer_name(str(optimizer)) in {
        "optimize_acqf_mixed",
        "evo_mixed",
        "optimize_acqf_evo_mixed",
        "torch_mixed",
        "optimize_acqf_torch_mixed",
        "thompson_sampling_mixed",
        "optimize_thompson_sampling_mixed",
    }


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


def optimize_candidates(acqf: Any, bounds: Any, config: _BaseOptimizeConfig) -> tuple[Any, Any]:
    """Dispatch canonical names, including NSGA-II and Thompson sampling."""

    if bounds is None:
        raise ValueError("bounds must be provided.")
    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return _BASE_OPTIMIZE_CANDIDATES(acqf=acqf, bounds=bounds, config=config)

    name = _optimizer_name(str(optimizer))
    is_mixed = config.fixed_features_list is not None

    if name in {"optimize_acqf", "evo", "torch"} and is_mixed:
        mixed_name = {
            "optimize_acqf": "optimize_acqf_mixed",
            "evo": "evo_mixed",
            "torch": "torch_mixed",
        }[name]
        return _BASE_OPTIMIZE_CANDIDATES(
            acqf=acqf,
            bounds=bounds,
            config=replace(config, optimizer=mixed_name),
        )

    special = {
        "nsgaii",
        "optimize_acqf_nsgaii",
        "thompson_sampling",
        "optimize_thompson_sampling",
        "thompson_sampling_mixed",
        "optimize_thompson_sampling_mixed",
    }
    if name not in special:
        return _BASE_OPTIMIZE_CANDIDATES(acqf=acqf, bounds=bounds, config=config)

    kwargs = _common_kwargs(acqf, bounds, config)
    if name in {"nsgaii", "optimize_acqf_nsgaii"}:
        from bochan.optim import optimize_acqf_nsgaii

        kwargs = _factory._filter_kwargs_for_callable(optimize_acqf_nsgaii, kwargs)
        return optimize_acqf_nsgaii(**kwargs)

    use_mixed = name in {
        "thompson_sampling_mixed",
        "optimize_thompson_sampling_mixed",
    } or (name in {"thompson_sampling", "optimize_thompson_sampling"} and is_mixed)
    if use_mixed:
        from bochan.optim import optimize_thompson_sampling_mixed

        fixed_features_list = _factory._merge_fixed_features_list(
            config.fixed_features,
            config.fixed_features_list,
        )
        if fixed_features_list is None:
            raise ValueError("fixed_features_list is required for mixed Thompson sampling.")
        kwargs.pop("fixed_features", None)
        kwargs["fixed_features_list"] = fixed_features_list
        kwargs = _factory._filter_kwargs_for_callable(optimize_thompson_sampling_mixed, kwargs)
        return optimize_thompson_sampling_mixed(**kwargs)

    from bochan.optim import optimize_thompson_sampling

    kwargs = _factory._filter_kwargs_for_callable(optimize_thompson_sampling, kwargs)
    return optimize_thompson_sampling(**kwargs)


__all__ = [
    "EvolutionaryMethod",
    "OptimizerName",
    "OptimizeConfig",
    "optimize_candidates",
    "resolve_optimizer_from_cat_dims",
    "uses_mixed_fixed_features",
]
