"""Acquisition-aware support search for k-sparse candidate optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from math import comb
from typing import Any

from ..configs import OptimizeConfig

BEST_SUBSET_MAX_COMBINATIONS_KWARG = "best_subset_max_combinations"
DEFAULT_BEST_SUBSET_MAX_COMBINATIONS = 2000

OptimizeOne = Callable[..., tuple[Any, Any]]


def uses_best_subset(config: OptimizeConfig) -> bool:
    """Return whether ``config`` requests acquisition-aware support search."""
    repair = config.repair_config
    return bool(repair is not None and repair.support_selection == "best_subset")


def _merged_fixed_features(config: OptimizeConfig) -> dict[int, float]:
    """Merge optimizer- and repair-level fixed features for subset search."""
    merged = {int(key): float(value) for key, value in (config.fixed_features or {}).items()}
    repair = config.repair_config
    if repair is not None:
        for key, value in (repair.fixed_features or {}).items():
            merged[int(key)] = float(value)
    return merged


def _validate_fixed_features_list(
    fixed_features_list: Sequence[Mapping[int, float]] | None,
    comp_idx: Sequence[int],
) -> None:
    """Reject mixed assignments that also mutate k-sparse support dimensions."""
    if fixed_features_list is None:
        return
    sparse_dims = {int(index) for index in comp_idx}
    conflicting = sorted(
        {
            int(key)
            for item in fixed_features_list
            for key in item
            if int(key) in sparse_dims
        }
    )
    if conflicting:
        raise ValueError(
            "best_subset does not support fixed_features_list entries on k-sparse "
            f"dimensions. Conflicting indices: {conflicting}."
        )


def enumerate_best_subset_supports(config: OptimizeConfig) -> list[tuple[int, ...]]:
    """Enumerate feasible exact-k supports after applying fixed-feature rules.

    Non-zero fixed k-sparse dimensions are required in every support. Zero-fixed
    dimensions are excluded. Remaining dimensions are enumerated exactly so the
    acquisition function, rather than local coefficient magnitude, chooses the
    support.
    """
    repair = config.repair_config
    if repair is None or repair.support_selection != "best_subset":
        raise ValueError("enumerate_best_subset_supports requires support_selection='best_subset'.")

    comp_idx = tuple(int(index) for index in (repair.comp_idx or ()))
    if not comp_idx:
        raise ValueError("best_subset requires a non-empty repair_config.comp_idx.")
    if len(set(comp_idx)) != len(comp_idx):
        raise ValueError("repair_config.comp_idx must not contain duplicate indices.")

    k = int(repair.k)
    if k < 0 or k > len(comp_idx):
        raise ValueError(
            f"best_subset requires 0 <= k <= len(comp_idx). Got k={k}, len={len(comp_idx)}."
        )

    _validate_fixed_features_list(config.fixed_features_list, comp_idx)
    fixed = _merged_fixed_features(config)
    required = {index for index in comp_idx if fixed.get(index, 0.0) != 0.0}
    forbidden = {index for index in comp_idx if index in fixed and fixed[index] == 0.0}

    if len(required) > k:
        raise ValueError(
            "best_subset has more non-zero fixed k-sparse dimensions than k: "
            f"required={len(required)}, k={k}."
        )

    free = [index for index in comp_idx if index not in required and index not in forbidden]
    choose = k - len(required)
    if choose > len(free):
        raise ValueError(
            "best_subset cannot construct an exact-k support after fixed-feature "
            f"constraints: required={len(required)}, free={len(free)}, k={k}."
        )

    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    max_combinations = int(
        optimizer_kwargs.get(
            BEST_SUBSET_MAX_COMBINATIONS_KWARG,
            DEFAULT_BEST_SUBSET_MAX_COMBINATIONS,
        )
    )
    if max_combinations < 1:
        raise ValueError(f"{BEST_SUBSET_MAX_COMBINATIONS_KWARG} must be >= 1.")

    n_combinations = comb(len(free), choose)
    if n_combinations > max_combinations:
        raise ValueError(
            "best_subset exact enumeration would evaluate "
            f"{n_combinations} supports, exceeding the limit {max_combinations}. "
            f"Increase optimizer_kwargs['{BEST_SUBSET_MAX_COMBINATIONS_KWARG}'] "
            "or reduce comp_idx / k."
        )

    supports: list[tuple[int, ...]] = []
    for selected in combinations(free, choose):
        active = required | set(selected)
        supports.append(tuple(index for index in comp_idx if index in active))
    return supports


def _config_for_support(config: OptimizeConfig, support: Sequence[int]) -> OptimizeConfig:
    """Build an inner optimization config with one shared support fixed."""
    repair = config.repair_config
    if repair is None:
        raise ValueError("best_subset requires repair_config.")

    comp_idx = tuple(int(index) for index in (repair.comp_idx or ()))
    support_tuple = tuple(int(index) for index in support)
    support_set = set(support_tuple)
    inactive = [index for index in comp_idx if index not in support_set]

    fixed = _merged_fixed_features(config)
    for index in inactive:
        fixed[index] = 0.0

    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    optimizer_kwargs.pop(BEST_SUBSET_MAX_COMBINATIONS_KWARG, None)

    inner_repair = replace(
        repair,
        comp_idx=support_tuple,
        k=len(support_tuple),
        support_selection="topk",
        fixed_features=fixed,
    )
    return replace(
        config,
        repair_config=inner_repair,
        fixed_features=fixed,
        optimizer_kwargs=optimizer_kwargs,
    )


def _scalar_acquisition_value(value: Any) -> float:
    """Convert one support's optimized joint acquisition value to a scalar."""
    if hasattr(value, "detach"):
        detached = value.detach()
        if int(detached.numel()) != 1:
            raise ValueError(
                "best_subset requires one acquisition value per optimized support. "
                "Use return_best_only=True and a joint q acquisition."
            )
        return float(detached.reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "best_subset requires a scalar acquisition value per optimized support."
        ) from exc


def _evaluate_acquisition(acqf: Any, candidates: Any) -> Any:
    """Evaluate the acquisition on the final repaired candidate set."""
    value = acqf(candidates)
    if hasattr(value, "detach"):
        return value.detach()
    return value


def optimize_best_subset_candidates(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    """Optimize the acquisition function for every feasible shared support.

    For q > 1, one support is shared by the entire q-batch. The inner optimizer
    still optimizes the joint q acquisition normally, so process variables and
    active sparse values remain jointly optimized within each support. Supports
    are compared by re-evaluating the acquisition on each final repaired
    candidate set.
    """
    if not uses_best_subset(config):
        return optimize_one(acqf=acqf, bounds=bounds, config=config)
    if not config.return_best_only:
        raise ValueError("best_subset currently requires OptimizeConfig.return_best_only=True.")

    supports = enumerate_best_subset_supports(config)
    best_candidates: Any | None = None
    best_value: Any | None = None
    best_score: float | None = None

    for support in supports:
        inner_config = _config_for_support(config, support)
        candidates, _ = optimize_one(
            acqf=acqf,
            bounds=bounds,
            config=inner_config,
        )
        acq_value = _evaluate_acquisition(acqf, candidates)
        score = _scalar_acquisition_value(acq_value)
        if best_score is None or score > best_score:
            best_candidates = candidates
            best_value = acq_value
            best_score = score

    if best_candidates is None:
        raise RuntimeError("best_subset did not produce any feasible support.")
    return best_candidates, best_value
