"""Public candidate-optimizer configuration and categorical dispatch."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .base import OptimizeConfig as _BaseOptimizeConfig
from .optimizer_names import (
    _ALIASES,
    _CANONICAL_OPTIMIZERS,
    _EVOLUTIONARY_METHODS,
    _MIXED_OPTIMIZERS,
    EvolutionaryMethod,
    OptimizerName,
    _InternalMixedOptimizerName,
    _optimizer_name,
)

FidelityValues = Sequence[float] | Mapping[int, Sequence[float]]
FidelityAssignments = Sequence[Mapping[int, float]]


def _normalize_value_sequence(values: Sequence[float], *, label: str) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ValueError(f"{label} must not be empty when supplied.")
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{label} must contain finite values.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} must not contain duplicates.")
    return resolved


def _normalize_fidelity_values(values: FidelityValues) -> tuple[float, ...] | dict[int, tuple[float, ...]]:
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("fidelity_values mapping must not be empty when supplied.")
        normalized: dict[int, tuple[float, ...]] = {}
        for raw_index, raw_values in values.items():
            index = int(raw_index)
            if index in normalized:
                raise ValueError("fidelity_values contains duplicate feature keys.")
            normalized[index] = _normalize_value_sequence(
                raw_values,
                label=f"fidelity_values[{raw_index}]",
            )
        return normalized
    return _normalize_value_sequence(values, label="fidelity_values")


def _normalize_fidelity_assignments(
    assignments: FidelityAssignments,
) -> tuple[dict[int, float], ...]:
    if not assignments:
        raise ValueError("fidelity_assignments must not be empty when supplied.")
    normalized: list[dict[int, float]] = []
    seen: set[tuple[tuple[int, float], ...]] = set()
    for assignment in assignments:
        if not assignment:
            raise ValueError("Each fidelity_assignments item must contain at least one feature.")
        item: dict[int, float] = {}
        for raw_index, raw_value in assignment.items():
            index = int(raw_index)
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("fidelity_assignments must contain finite values.")
            if index in item:
                raise ValueError("fidelity_assignments contains duplicate feature keys.")
            item[index] = value
        key = tuple(sorted(item.items()))
        if key in seen:
            raise ValueError("fidelity_assignments must not contain duplicate assignments.")
        seen.add(key)
        normalized.append(item)
    return tuple(normalized)


@dataclass
class OptimizeConfig(_BaseOptimizeConfig):
    """Candidate optimizer configuration using backend-family names."""

    optimizer: OptimizerName | str | Callable[..., Any] = "optimize_acqf"
    evo_method: EvolutionaryMethod = "ga"
    fidelity_values: FidelityValues | None = None
    fidelity_assignments: FidelityAssignments | None = None
    optimize_fidelity: bool = False
    ensure_unique_candidates: bool = True
    duplicate_tolerance: float = 1e-10
    duplicate_tolerances: Sequence[float] | None = None
    final_candidate_postprocess: Callable[[Any], Any] | None = None
    duplicate_refill_attempts: int = 4
    duplicate_pool_restarts: int = 16

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["final_candidate_postprocess"] = None
        return state

    def __post_init__(self) -> None:
        active_modes = sum(
            (
                self.fidelity_values is not None,
                self.fidelity_assignments is not None,
                bool(self.optimize_fidelity),
            )
        )
        if active_modes > 1:
            raise ValueError(
                "fidelity_values, fidelity_assignments, and optimize_fidelity=True are mutually "
                "exclusive query-fidelity modes."
            )
        if self.fidelity_values is not None:
            self.fidelity_values = _normalize_fidelity_values(self.fidelity_values)
        if self.fidelity_assignments is not None:
            self.fidelity_assignments = _normalize_fidelity_assignments(self.fidelity_assignments)
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
        repair = self.repair_config
        if repair is not None and repair.inequality_constraints is None and self.inequality_constraints is not None and str(repair.inequality_sense).lower() != "ge":
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
                raise ValueError(f"Unknown evolutionary method: {effective_method!r}. Expected one of {sorted(_EVOLUTIONARY_METHODS)}.")
            self.evo_method = effective_method  # type: ignore[assignment]
            if effective_method == "cmaes" and self.q > 1:
                self.sequential = True


def resolve_optimizer_from_cat_dims(*, opt_config: _BaseOptimizeConfig, cat_dims: Sequence[int] | None) -> _BaseOptimizeConfig:
    if not cat_dims:
        return opt_config
    optimizer = opt_config.optimizer
    if callable(optimizer) and not isinstance(optimizer, str):
        return opt_config
    mixed_name = {"optimize_acqf": "optimize_acqf_mixed", "evo": "evo_mixed", "torch": "torch_mixed", "thompson_sampling": "thompson_sampling_mixed"}.get(_optimizer_name(str(optimizer)))
    return opt_config if mixed_name is None else replace(opt_config, optimizer=_InternalMixedOptimizerName(mixed_name))


def uses_mixed_fixed_features(optimizer: Any) -> bool:
    if callable(optimizer) and not isinstance(optimizer, str):
        return False
    return _optimizer_name(str(optimizer)) in _MIXED_OPTIMIZERS
