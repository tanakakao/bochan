"""One-shot acquisition helpers for acquisition-aware Best Subset search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from ..configs import OptimizeConfig


def is_one_shot_acquisition(acqf: Any) -> bool:
    """Return whether ``acqf`` uses BoTorch's augmented one-shot parameterization."""

    try:
        from botorch.acquisition.acquisition import OneShotAcquisitionFunction
    except ImportError:  # pragma: no cover - BoTorch is a core dependency.
        return False
    return isinstance(acqf, OneShotAcquisitionFunction)


def base_one_shot_acquisition(acqf: Any) -> Any:
    """Return the underlying BoTorch one-shot acquisition for transparent wrappers."""

    return getattr(acqf, "_bochan_one_shot_base", acqf)


def resolve_one_shot_ic_generator(acqf: Any) -> Any | None:
    """Recover BoTorch's specialized one-shot initializer through wrappers."""

    if not is_one_shot_acquisition(acqf):
        return None
    base = base_one_shot_acquisition(acqf)
    try:
        from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
        from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
            qHypervolumeKnowledgeGradient,
        )
        from botorch.optim.initializers import (
            gen_one_shot_hvkg_initial_conditions,
            gen_one_shot_kg_initial_conditions,
        )
    except ImportError:  # pragma: no cover
        return None
    if isinstance(base, qHypervolumeKnowledgeGradient):
        return gen_one_shot_hvkg_initial_conditions
    if isinstance(base, qKnowledgeGradient):
        return gen_one_shot_kg_initial_conditions
    return None


def _merged_fixed_features(config: OptimizeConfig) -> dict[int, float]:
    fixed = {
        int(key): float(value)
        for key, value in (config.fixed_features or {}).items()
    }
    repair = config.repair_config
    if repair is not None:
        for key, value in (repair.fixed_features or {}).items():
            index = int(key)
            numeric = float(value)
            previous = fixed.get(index)
            if previous is not None and abs(previous - numeric) > 1e-12:
                raise ValueError(
                    f"Conflicting fixed values were provided for feature {index}."
                )
            fixed[index] = numeric
    return fixed


def _indices_list(value: Any) -> list[int]:
    tensor = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    return [int(item) for item in tensor.tolist()]


def _coefficients_list(value: Any) -> list[float]:
    tensor = torch.as_tensor(value, dtype=torch.double).reshape(-1)
    return [float(item) for item in tensor.tolist()]


def _constraint_signature(item: Any, *, sense: str = "ge") -> tuple[Any, ...]:
    indices, coefficients, rhs = item
    coeff = _coefficients_list(coefficients)
    rhs_value = float(rhs)
    normalized_sense = str(sense).lower()
    if normalized_sense == "le":
        coeff = [-value for value in coeff]
        rhs_value = -rhs_value
    return (
        tuple(_indices_list(indices)),
        tuple(round(value, 14) for value in coeff),
        round(rhs_value, 14),
    )


def _repair_floor_constraints(
    config: OptimizeConfig,
    sparse_indices: Sequence[int],
) -> tuple[dict[int, float], set[int]]:
    """Extract sparse-dimension positive floors from repair inequalities."""

    repair = config.repair_config
    if repair is None or repair.inequality_constraints is None:
        return {}, set()
    sparse = {int(index) for index in sparse_indices}
    floors: dict[int, float] = {}
    consumed: set[int] = set()
    sense = str(repair.inequality_sense).lower()
    for position, item in enumerate(repair.inequality_constraints):
        indices, coefficients, rhs = item
        resolved_indices = _indices_list(indices)
        resolved_coefficients = _coefficients_list(coefficients)
        if len(resolved_indices) != 1 or resolved_indices[0] not in sparse:
            continue
        coefficient = float(resolved_coefficients[0])
        rhs_value = float(rhs)
        if sense == "le":
            coefficient = -coefficient
            rhs_value = -rhs_value
        if coefficient <= 0.0:
            continue
        floor = rhs_value / coefficient
        if floor <= 0.0:
            continue
        index = resolved_indices[0]
        floors[index] = max(floors.get(index, 0.0), floor)
        consumed.add(position)
    return floors, consumed


def _validate_repair_constraints_are_represented(
    config: OptimizeConfig,
    *,
    consumed_inequalities: set[int],
) -> None:
    """Reject repair-only domain constraints that would disappear without repair."""

    repair = config.repair_config
    if repair is None:
        return
    optimizer_equalities = {
        _constraint_signature(item)
        for item in (config.equality_constraints or ())
    }
    for item in repair.equality_constraints or ():
        if _constraint_signature(item) not in optimizer_equalities:
            raise ValueError(
                "One-shot best_subset requires repair equality constraints to also be "
                "present in OptimizeConfig.equality_constraints."
            )

    optimizer_inequalities = {
        _constraint_signature(item)
        for item in (config.inequality_constraints or ())
    }
    for position, item in enumerate(repair.inequality_constraints or ()):
        if position in consumed_inequalities:
            continue
        if (
            _constraint_signature(item, sense=str(repair.inequality_sense))
            not in optimizer_inequalities
        ):
            raise ValueError(
                "One-shot best_subset requires repair inequality constraints to also be "
                "present in OptimizeConfig.inequality_constraints."
            )


def _constraint_tensor(
    *,
    point_index: int,
    feature_index: int,
    coefficient: float,
    rhs: float,
    bounds: Tensor,
) -> tuple[Tensor, Tensor, float]:
    indices = torch.tensor(
        [[int(point_index), int(feature_index)]],
        dtype=torch.long,
        device=bounds.device,
    )
    coefficients = torch.tensor(
        [float(coefficient)],
        dtype=bounds.dtype,
        device=bounds.device,
    )
    return indices, coefficients, float(rhs)


def _support_constraints(
    *,
    q: int,
    sparse_indices: Sequence[int],
    support: Sequence[int],
    active_floors: Mapping[int, float],
    bounds: Tensor,
) -> tuple[list[Any], list[Any]]:
    """Build actual-candidate-only support constraints for a one-shot tree."""

    support_set = {int(index) for index in support}
    equalities: list[Any] = []
    inequalities: list[Any] = []
    for feature_index in (int(index) for index in sparse_indices):
        if feature_index not in support_set:
            for point_index in range(int(q)):
                equalities.append(
                    _constraint_tensor(
                        point_index=point_index,
                        feature_index=feature_index,
                        coefficient=1.0,
                        rhs=0.0,
                        bounds=bounds,
                    )
                )
            continue
        floor = float(active_floors.get(feature_index, 0.0))
        if floor <= 0.0:
            continue
        for point_index in range(int(q)):
            inequalities.append(
                _constraint_tensor(
                    point_index=point_index,
                    feature_index=feature_index,
                    coefficient=1.0,
                    rhs=floor,
                    bounds=bounds,
                )
            )
    return equalities, inequalities


def validate_one_shot_best_subset(
    acqf: Any,
    config: OptimizeConfig,
) -> None:
    """Validate the currently exact continuous one-shot Best Subset contract."""

    if not is_one_shot_acquisition(acqf):
        return
    if bool(config.sequential):
        raise ValueError("One-shot best_subset requires sequential=False.")
    if not bool(config.return_best_only):
        raise ValueError("One-shot best_subset requires return_best_only=True.")
    if config.fixed_features_list is not None:
        raise ValueError(
            "One-shot best_subset does not yet support mixed fixed_features_list; "
            "use a continuous optimizer domain."
        )
    optimizer_name = str(config.optimizer).replace("-", "_").lower()
    if optimizer_name != "optimize_acqf":
        raise ValueError(
            "One-shot best_subset currently requires optimizer='optimize_acqf'."
        )
    if config.post_processing_func is not None:
        raise ValueError(
            "One-shot best_subset does not support model-space post_processing_func."
        )
    if getattr(config, "final_candidate_postprocess", None) is not None:
        raise ValueError(
            "One-shot best_subset with final_candidate_postprocess requires conditional "
            "re-optimization of auxiliary one-shot variables and is not yet supported."
        )
    repair = config.repair_config
    if repair is None:
        return
    if repair.steps is not None:
        raise ValueError(
            "One-shot best_subset does not yet support repair step grids."
        )
    if repair.numeric_indices not in (None, [], ()):
        raise ValueError(
            "One-shot best_subset does not yet support repair-time numeric rounding."
        )
    if bool(repair.diversify):
        raise ValueError(
            "One-shot best_subset does not yet support repair-time diversification."
        )


def one_shot_support_config(
    config: OptimizeConfig,
    *,
    sparse_indices: Sequence[int],
    support: Sequence[int],
    bounds: Tensor,
    optimizer_kwargs: Mapping[str, Any],
) -> OptimizeConfig:
    """Build an inner config that constrains only the real q experiment points.

    BoTorch one-shot acquisitions optimize an augmented q-batch containing the
    actual experiment candidates followed by auxiliary fantasy/value-function
    solutions. Support-specific zero/floor constraints belong only to the actual
    candidates. Global fixed features and ordinary domain constraints remain
    intra-point constraints and therefore also apply to auxiliary solutions.
    """

    repair = config.repair_config
    if repair is None:
        raise ValueError("one-shot best_subset requires repair_config.")
    floors, consumed = _repair_floor_constraints(config, sparse_indices)
    _validate_repair_constraints_are_represented(
        config,
        consumed_inequalities=consumed,
    )
    extra_equalities, extra_inequalities = _support_constraints(
        q=int(config.q),
        sparse_indices=sparse_indices,
        support=support,
        active_floors=floors,
        bounds=bounds,
    )

    equality_constraints = [
        *(config.equality_constraints or ()),
        *extra_equalities,
    ]
    inequality_constraints = [
        *(config.inequality_constraints or ()),
        *extra_inequalities,
    ]

    if repair.final_sum_constraint is not None:
        indices, rhs = repair.final_sum_constraint
        candidate = (
            torch.as_tensor(indices, dtype=torch.long, device=bounds.device),
            torch.ones(len(indices), dtype=bounds.dtype, device=bounds.device),
            float(rhs),
        )
        signature = _constraint_signature(candidate)
        existing = {_constraint_signature(item) for item in equality_constraints}
        if signature not in existing:
            equality_constraints.append(candidate)

    replacements: dict[str, Any] = {
        "repair_config": None,
        "fixed_features": _merged_fixed_features(config) or None,
        "equality_constraints": equality_constraints or None,
        "inequality_constraints": inequality_constraints or None,
        "optimizer_kwargs": dict(optimizer_kwargs),
    }
    if hasattr(config, "ensure_unique_candidates"):
        # q=1 duplicate refill would invalidate the q-indexed support constraints.
        # Preserve the native joint one-shot batch for this inner support problem.
        replacements["ensure_unique_candidates"] = False
    return replace(config, **replacements)


__all__ = [
    "base_one_shot_acquisition",
    "is_one_shot_acquisition",
    "one_shot_support_config",
    "resolve_one_shot_ic_generator",
    "validate_one_shot_best_subset",
]
