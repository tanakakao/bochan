"""Public candidate-optimizer configuration and categorical dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .configs import OptimizeConfig as _BaseOptimizeConfig
from .optimizer_support import (
    EvolutionaryMethod,
    OptimizerName,
    _ALIASES,
    _CANONICAL_OPTIMIZERS,
    _EVOLUTIONARY_METHODS,
    _InternalMixedOptimizerName,
    _MIXED_OPTIMIZERS,
    _optimizer_name,
)

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

    Final q-batches are checked for duplicate post-processed candidates by
    default. Duplicate slots are refilled from additional q=1 restart optima of
    the same acquisition and backend, preserving the initial joint/sequential
    policy and avoiding acquisition wrappers.
    """

    optimizer: OptimizerName | str | Callable[..., Any] = "optimize_acqf"
    evo_method: EvolutionaryMethod = "ga"
    ensure_unique_candidates: bool = True
    duplicate_tolerance: float = 1e-10
    duplicate_refill_attempts: int = 4
    duplicate_pool_restarts: int = 16

    def __post_init__(self) -> None:
        if self.duplicate_tolerance < 0:
            raise ValueError("duplicate_tolerance must be non-negative.")
        if self.duplicate_refill_attempts < 1:
            raise ValueError("duplicate_refill_attempts must be at least 1.")
        if self.duplicate_pool_restarts < 1:
            raise ValueError("duplicate_pool_restarts must be at least 1.")

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


