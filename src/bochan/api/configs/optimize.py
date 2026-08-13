"""Public candidate-optimizer configuration and categorical dispatch."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..optimizer_support import (
    _ALIASES,
    _CANONICAL_OPTIMIZERS,
    _EVOLUTIONARY_METHODS,
    _MIXED_OPTIMIZERS,
    EvolutionaryMethod,
    OptimizerName,
    _InternalMixedOptimizerName,
    _optimizer_name,
)
from .base import OptimizeConfig as _BaseOptimizeConfig


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
    duplicate_tolerances: Sequence[float] | None = None
    final_candidate_postprocess: Callable[[Any], Any] | None = None
    duplicate_refill_attempts: int = 4
    duplicate_pool_restarts: int = 16

    def __post_init__(self) -> None:
        if self.duplicate_tolerance < 0:
            raise ValueError("duplicate_tolerance must be non-negative.")
        if self.duplicate_tolerances is not None:
            tolerances = tuple(float(value) for value in self.duplicate_tolerances)
            if any(not math.isfinite(value) or value < 0 for value in tolerances):
                raise ValueError("duplicate_tolerances must contain finite non-negative values.")
            self.duplicate_tolerances = tolerances
        if self.final_candidate_postprocess is not None and not callable(self.final_candidate_postprocess):
            raise ValueError("final_candidate_postprocess must be callable.")
        if self.duplicate_refill_attempts < 1:
            raise ValueError("duplicate_refill_attempts must be at least 1.")
        if self.duplicate_pool_restarts < 1:
            raise ValueError("duplicate_pool_restarts must be at least 1.")

        # Top-level OptimizeConfig.inequality_constraints follow BoTorch's
        # canonical a^T x >= rhs convention. If candidate repair falls back to
        # those constraints, resolve the repair-local sense here rather than by
        # replacing factory helpers at import time.
        repair = self.repair_config
        if (
            repair is not None
            and repair.inequality_constraints is None
            and self.inequality_constraints is not None
            and str(repair.inequality_sense).lower() != "ge"
        ):
            self.repair_config = replace(repair, inequality_sense="ge")

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
            effective_method = _optimizer_name(str(self.optimizer_kwargs.setdefault("method", self.evo_method)))
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
    return opt_config if mixed_name is None else replace(opt_config, optimizer=_InternalMixedOptimizerName(mixed_name))


def uses_mixed_fixed_features(optimizer: Any) -> bool:
    """Return whether the backend needs categorical fixed-feature enumeration."""

    if callable(optimizer) and not isinstance(optimizer, str):
        return False
    return _optimizer_name(str(optimizer)) in _MIXED_OPTIMIZERS
