"""Shared optimizer-name, model, and Thompson-sampling helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from .configs import OptimizeConfig as _BaseOptimizeConfig

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


