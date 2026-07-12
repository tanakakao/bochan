"""Canonical optimizer configuration and high-level dispatch helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

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
    "llm_candidate_set",
]

_CANONICAL_OPTIMIZERS = {
    "optimize_acqf",
    "evo",
    "torch",
    "nsgaii",
    "thompson_sampling",
    "llm_candidate_set",
}
_EVOLUTIONARY_METHODS = {"ga", "pso", "sa", "cmaes"}
_MIXED_OPTIMIZERS = {
    "optimize_acqf_mixed",
    "evo_mixed",
    "optimize_acqf_evo_mixed",
    "torch_mixed",
    "optimize_acqf_torch_mixed",
    "thompson_sampling_mixed",
    "optimize_thompson_sampling_mixed",
}
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
    "thompson_sampling_mixed": "thompson_sampling",
    "optimize_thompson_sampling_mixed": "thompson_sampling",
    "thompson": "thompson_sampling",
    "llm": "llm_candidate_set",
    "llm_candidate": "llm_candidate_set",
    "optimize_acqf_llm": "llm_candidate_set",
    "optimize_acqf_llm_candidate_set": "llm_candidate_set",
}


class _InternalMixedOptimizerName(str):
    """Mark a mixed optimizer name selected internally from categorical dims.

    Publicly supplied old mixed names are still normalized to canonical family
    names. The marker survives ``dataclasses.replace`` so downstream config copies
    retain the internally selected mixed implementation.
    """


def _optimizer_name(optimizer: str) -> str:
    return optimizer.replace("-", "_").lower()


def _uses_kronecker_model(value: Any, *, _seen: set[int] | None = None) -> bool:
    """Return whether an acquisition/model tree contains a Kronecker model."""

    if value is None:
        return False
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return False
    _seen.add(value_id)

    if "kronecker" in value.__class__.__name__.lower():
        return True

    nested_model = getattr(value, "model", None)
    if (
        nested_model is not None
        and nested_model is not value
        and _uses_kronecker_model(nested_model, _seen=_seen)
    ):
        return True

    nested_models = getattr(value, "models", None)
    if nested_models is not None:
        try:
            return any(_uses_kronecker_model(model, _seen=_seen) for model in nested_models)
        except TypeError:
            pass
    return False


def _force_sequential_for_kronecker(
    acqf: Any,
    config: _BaseOptimizeConfig,
) -> _BaseOptimizeConfig:
    """Stabilize SciPy optimization for Kronecker model acquisitions.

    Native joint ``q > 1`` optimization of Kronecker multi-task posteriors can
    fail during LinearOperator backward because the task dimension ``m`` and the
    flattened event dimension ``q * m`` are mixed. Sequential optimization keeps
    each internal optimization step at ``q=1`` while returning the requested
    number of candidates.

    Kronecker posteriors can also fail during input-gradient evaluation when an
    input perturbation transform expands a single candidate to ``n_w`` replicas.
    In that case, even sequential ``q=1`` optimization differentiates through a
    ``q * n_w`` Kronecker event. BoTorch's SciPy generator supports finite
    differences via ``options={"with_grad": False}``, so disable autograd
    gradients for Kronecker acquisitions unless the caller explicitly set that
    option.

    Args:
        acqf: Acquisition function being optimized.
        config: Candidate optimizer configuration.

    Returns:
        Optimizer configuration with Kronecker-safe defaults applied when the
        standard SciPy ``optimize_acqf`` backend is used.
    """

    optimizer = config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return config

    name = _ALIASES.get(_optimizer_name(str(optimizer)), _optimizer_name(str(optimizer)))
    if name != "optimize_acqf" or not _uses_kronecker_model(acqf):
        return config

    needs_update = False
    kwargs = dict(config.optimizer_kwargs)
    options = dict(kwargs.get("options") or {})
    if "with_grad" not in options:
        options["with_grad"] = False
        kwargs["options"] = options
        needs_update = True

    sequential = config.sequential
    if config.q > 1 and not sequential:
        sequential = True
        needs_update = True

    if not needs_update:
        return config
    return replace(config, sequential=sequential, optimizer_kwargs=kwargs)


def _has_posterior(value: Any) -> bool:
    """Return whether ``value`` can be used as a BoTorch posterior model."""

    return callable(getattr(value, "posterior", None))


def _configured_thompson_sampling_model(acqf: Any) -> Any | None:
    """Return an explicitly stored Thompson-sampling model when available."""

    for name in ("_bochan_thompson_model", "_thompson_sampling_model"):
        model = getattr(acqf, name, None)
        if model is not None and _has_posterior(model):
            return model
    return None


def _is_callable_acquisition(acqf: Any) -> bool:
    """Return whether ``acqf`` is an acquisition rather than a posterior model.

    Active-learning acquisitions such as BALD, entropy, variance, straddle, and
    ICU are callable and own their selection semantics, but commonly have
    ``objective=None``. They must remain available to the Thompson adapter so it
    can rank the finite candidate pool by acquisition value instead of sampling
    raw multiclass probability tensors from the underlying model.
    """

    return (
        callable(acqf)
        and not _has_posterior(acqf)
        and getattr(acqf, "model", None) is not None
    )


def _has_thompson_sampling_context(acqf: Any) -> bool:
    """Return whether stripping an acquisition to its model would lose semantics.

    Finite-pool posterior sampling uses the fitted model for draws, but the
    Thompson adapter also needs acquisition-owned objective, posterior-transform,
    and outcome-constraint state. Context-free callable acquisitions still need
    to be preserved because the adapter optimizes their acquisition values over
    the finite pool rather than treating raw posterior class probabilities as
    scalar Thompson values.
    """

    return _is_callable_acquisition(acqf) or any(
        getattr(acqf, name, None) is not None
        for name in (
            "objective",
            "posterior_transform",
            "constraints",
            "outcome_constraints",
        )
    )


def _resolve_thompson_sampling_target(acqf: Any) -> Any:
    """Return the object consumed by the Thompson sampling adapter.

    Keep the acquisition whenever it owns Thompson-relevant context or is itself
    a callable acquisition. The adapter can resolve and sample from ``acqf.model``
    for explicit posterior objectives, or score the acquisition directly for
    active-learning criteria such as BALD and entropy.

    Non-callable context-free wrappers may still be reduced to a posterior model
    for the lightweight direct-sampling path. If the acquisition exposes only a
    latent internal ``.model`` without a public ``posterior`` method, keep the
    acquisition so the adapter can recover a configured model or use acquisition
    scoring.
    """

    if _has_thompson_sampling_context(acqf):
        return acqf

    configured_model = _configured_thompson_sampling_model(acqf)
    if configured_model is not None:
        return configured_model

    model = getattr(acqf, "model", None)
    if model is not None and _has_posterior(model):
        return model
    if _has_posterior(acqf):
        return acqf
    return acqf


@dataclass
class OptimizeConfig(_BaseOptimizeConfig):
    """Candidate optimizer configuration using backend-family names.

    Mixed/non-mixed implementations are selected automatically. Evolutionary
    backends may be selected with ``optimizer="evo"`` plus ``evo_method``, or
    directly with ``optimizer="ga"``, ``"pso"``, ``"sa"``, or ``"cmaes"``.
    ``optimizer="llm_candidate_set"`` asks an LLM for a candidate set and then
    reranks that set with the existing acquisition function.

    CMA-ES only optimizes one point at a time. Therefore, when its effective
    method is ``cmaes`` and ``q > 1``, ``sequential`` is enabled automatically.
    """

    optimizer: OptimizerName | str | Callable[..., Any] = "optimize_acqf"
    evo_method: EvolutionaryMethod = "ga"

    def __post_init__(self) -> None:
        if callable(self.optimizer) and not isinstance(self.optimizer, str):
            return

        raw_name = _optimizer_name(str(self.optimizer))
        preserve_mixed = isinstance(self.optimizer, _InternalMixedOptimizerName)
        name = raw_name if preserve_mixed else _ALIASES.get(raw_name, raw_name)
        if not preserve_mixed and name in _EVOLUTIONARY_METHODS:
            self.evo_method = name  # type: ignore[assignment]
            name = "evo"

        valid_names = _MIXED_OPTIMIZERS if preserve_mixed else _CANONICAL_OPTIMIZERS
        if name not in valid_names:
            valid = sorted(_CANONICAL_OPTIMIZERS | _EVOLUTIONARY_METHODS)
            raise ValueError(f"Unknown optimizer: {self.optimizer!r}. Expected one of {valid}.")

        self.optimizer = _InternalMixedOptimizerName(name) if preserve_mixed else name
        self.optimizer_kwargs = dict(self.optimizer_kwargs)
        if name in {"evo", "evo_mixed", "optimize_acqf_evo_mixed"}:
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
    return (
        opt_config
        if mixed_name is None
        else replace(opt_config, optimizer=_InternalMixedOptimizerName(mixed_name))
    )


def uses_mixed_fixed_features(optimizer: Any) -> bool:
    """Return whether the backend needs categorical fixed-feature enumeration."""

    if callable(optimizer) and not isinstance(optimizer, str):
        return False
    return _optimizer_name(str(optimizer)) in _MIXED_OPTIMIZERS


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
    """Dispatch canonical names, including NSGA-II, Thompson sampling, and LLM reranking."""

    if bounds is None:
        raise ValueError("bounds must be provided.")
    config = _force_sequential_for_kronecker(acqf, config)
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
            config=replace(
                config,
                optimizer=_InternalMixedOptimizerName(mixed_name),
            ),
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
        return _BASE_OPTIMIZE_CANDIDATES(acqf=acqf, bounds=bounds, config=config)

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
