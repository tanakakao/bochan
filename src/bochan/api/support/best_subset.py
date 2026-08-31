"""Acquisition-aware support search for sparse candidate optimization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from math import comb
from typing import Any, Literal

from ..configs import OptimizeConfig

BEST_SUBSET_MAX_COMBINATIONS_KWARG = "best_subset_max_combinations"
BEST_SUBSET_STRATEGY_KWARG = "best_subset_strategy"
BEST_SUBSET_BEAM_WIDTH_KWARG = "best_subset_beam_width"
BEST_SUBSET_BEAM_STEPS_KWARG = "best_subset_beam_steps"
BEST_SUBSET_MAX_EVALUATIONS_KWARG = "best_subset_max_evaluations"
BEST_SUBSET_MIN_K_KWARG = "best_subset_min_k"
BEST_SUBSET_MAX_K_KWARG = "best_subset_max_k"

DEFAULT_BEST_SUBSET_MAX_COMBINATIONS = 2000
DEFAULT_BEST_SUBSET_STRATEGY = "exact"
DEFAULT_BEST_SUBSET_BEAM_WIDTH = 8
DEFAULT_BEST_SUBSET_BEAM_STEPS = 4
DEFAULT_BEST_SUBSET_MAX_EVALUATIONS = 200

BestSubsetStrategy = Literal["exact", "beam", "auto"]
OptimizeOne = Callable[..., tuple[Any, Any]]

_SUPPORT_SEARCH_KWARGS = {
    BEST_SUBSET_MAX_COMBINATIONS_KWARG,
    BEST_SUBSET_STRATEGY_KWARG,
    BEST_SUBSET_BEAM_WIDTH_KWARG,
    BEST_SUBSET_BEAM_STEPS_KWARG,
    BEST_SUBSET_MAX_EVALUATIONS_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    BEST_SUBSET_MAX_K_KWARG,
}


class InfeasibleBestSubsetSupportError(ValueError):
    """Signal that one support is infeasible without invalidating the whole search.

    Support-specific postprocessors may raise this exception when a structurally
    valid support cannot satisfy additional support-dependent constraints. Exact
    and beam Best Subset search skip only this explicit exception; all other
    optimizer and configuration errors remain visible to the caller.
    """


@dataclass(frozen=True)
class _SupportProblem:
    comp_idx: tuple[int, ...]
    min_k: int
    max_k: int
    required: frozenset[int]
    forbidden: frozenset[int]
    free: tuple[int, ...]
    min_choose: int
    max_choose: int

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(range(self.min_k, self.max_k + 1))

    @property
    def choose_counts(self) -> tuple[int, ...]:
        return tuple(range(self.min_choose, self.max_choose + 1))

    @property
    def count(self) -> int:
        return sum(comb(len(self.free), choose) for choose in self.choose_counts)


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
    """Reject mixed assignments that also mutate sparse support dimensions."""
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


def _requested_k_range(config: OptimizeConfig, *, default_k: int) -> tuple[int, int]:
    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    minimum = int(optimizer_kwargs.get(BEST_SUBSET_MIN_K_KWARG, default_k))
    maximum = int(optimizer_kwargs.get(BEST_SUBSET_MAX_K_KWARG, default_k))
    if minimum < 0:
        raise ValueError(f"{BEST_SUBSET_MIN_K_KWARG} must be >= 0.")
    if maximum < minimum:
        raise ValueError(
            f"{BEST_SUBSET_MAX_K_KWARG} must be >= {BEST_SUBSET_MIN_K_KWARG}."
        )
    return minimum, maximum


def _support_problem(config: OptimizeConfig) -> _SupportProblem:
    repair = config.repair_config
    if repair is None or repair.support_selection != "best_subset":
        raise ValueError("best_subset support search requires support_selection='best_subset'.")

    comp_idx = tuple(int(index) for index in (repair.comp_idx or ()))
    if not comp_idx:
        raise ValueError("best_subset requires a non-empty repair_config.comp_idx.")
    if len(set(comp_idx)) != len(comp_idx):
        raise ValueError("repair_config.comp_idx must not contain duplicate indices.")

    default_k = int(repair.k)
    if default_k < 0 or default_k > len(comp_idx):
        raise ValueError(
            "best_subset requires 0 <= repair_config.k <= len(comp_idx). "
            f"Got k={default_k}, len={len(comp_idx)}."
        )
    min_k, max_k = _requested_k_range(config, default_k=default_k)
    if max_k > len(comp_idx):
        raise ValueError(
            "best_subset cardinality range must not exceed len(comp_idx). "
            f"Got max_k={max_k}, len={len(comp_idx)}."
        )

    _validate_fixed_features_list(config.fixed_features_list, comp_idx)
    fixed = _merged_fixed_features(config)
    required = frozenset(index for index in comp_idx if fixed.get(index, 0.0) != 0.0)
    forbidden = frozenset(index for index in comp_idx if index in fixed and fixed[index] == 0.0)

    if len(required) > max_k:
        raise ValueError(
            "best_subset has more non-zero fixed sparse dimensions than max_k: "
            f"required={len(required)}, max_k={max_k}."
        )

    free = tuple(index for index in comp_idx if index not in required and index not in forbidden)
    effective_min_k = max(min_k, len(required))
    effective_max_k = min(max_k, len(required) + len(free))
    if effective_min_k > effective_max_k:
        raise ValueError(
            "best_subset cannot construct any support in the requested cardinality range "
            "after fixed-feature constraints: "
            f"required={len(required)}, free={len(free)}, "
            f"requested=[{min_k}, {max_k}]."
        )

    return _SupportProblem(
        comp_idx=comp_idx,
        min_k=effective_min_k,
        max_k=effective_max_k,
        required=required,
        forbidden=forbidden,
        free=free,
        min_choose=effective_min_k - len(required),
        max_choose=effective_max_k - len(required),
    )


def _max_combinations(config: OptimizeConfig) -> int:
    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    value = int(
        optimizer_kwargs.get(
            BEST_SUBSET_MAX_COMBINATIONS_KWARG,
            DEFAULT_BEST_SUBSET_MAX_COMBINATIONS,
        )
    )
    if value < 1:
        raise ValueError(f"{BEST_SUBSET_MAX_COMBINATIONS_KWARG} must be >= 1.")
    return value


def _resolve_strategy(
    config: OptimizeConfig,
    problem: _SupportProblem | None = None,
) -> BestSubsetStrategy:
    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    strategy = str(
        optimizer_kwargs.get(
            BEST_SUBSET_STRATEGY_KWARG,
            DEFAULT_BEST_SUBSET_STRATEGY,
        )
    ).lower()
    if strategy not in {"exact", "beam", "auto"}:
        raise ValueError(
            f"{BEST_SUBSET_STRATEGY_KWARG} must be one of 'exact', 'beam', or 'auto'. "
            f"Got {strategy!r}."
        )
    if strategy != "auto":
        return strategy  # type: ignore[return-value]

    problem = problem or _support_problem(config)
    return "exact" if problem.count <= _max_combinations(config) else "beam"


def enumerate_best_subset_supports(config: OptimizeConfig) -> list[tuple[int, ...]]:
    """Enumerate feasible supports across the requested cardinality range.

    Without ``best_subset_min_k`` / ``best_subset_max_k`` this preserves the
    historical exact-k behavior and uses ``repair_config.k`` for both bounds.
    Non-zero fixed sparse dimensions are required in every support and zero-fixed
    dimensions are excluded.
    """
    problem = _support_problem(config)
    max_combinations = _max_combinations(config)
    if problem.count > max_combinations:
        raise ValueError(
            "best_subset exact enumeration would evaluate "
            f"{problem.count} supports, exceeding the limit {max_combinations}. "
            f"Increase optimizer_kwargs['{BEST_SUBSET_MAX_COMBINATIONS_KWARG}'], "
            f"set optimizer_kwargs['{BEST_SUBSET_STRATEGY_KWARG}']='beam' (or 'auto'), "
            "or reduce comp_idx / the cardinality range."
        )

    supports: list[tuple[int, ...]] = []
    for choose in problem.choose_counts:
        for selected in combinations(problem.free, choose):
            active = set(problem.required) | set(selected)
            supports.append(
                tuple(index for index in problem.comp_idx if index in active)
            )
    return supports


def _inner_optimizer_kwargs(config: OptimizeConfig) -> dict[str, Any]:
    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    for key in _SUPPORT_SEARCH_KWARGS:
        optimizer_kwargs.pop(key, None)
    return optimizer_kwargs


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
        optimizer_kwargs=_inner_optimizer_kwargs(config),
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


def _evaluate_support(
    *,
    support: Sequence[int],
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any, float]:
    inner_config = _config_for_support(config, support)
    candidates, _ = optimize_one(
        acqf=acqf,
        bounds=bounds,
        config=inner_config,
    )
    acq_value = _evaluate_acquisition(acqf, candidates)
    return candidates, acq_value, _scalar_acquisition_value(acq_value)


def _optimize_exact_best_subset(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    supports = enumerate_best_subset_supports(config)
    best_candidates: Any | None = None
    best_value: Any | None = None
    best_score: float | None = None
    infeasible_count = 0

    for support in supports:
        try:
            candidates, acq_value, score = _evaluate_support(
                support=support,
                acqf=acqf,
                bounds=bounds,
                config=config,
                optimize_one=optimize_one,
            )
        except InfeasibleBestSubsetSupportError:
            infeasible_count += 1
            continue
        if best_score is None or score > best_score:
            best_candidates = candidates
            best_value = acq_value
            best_score = score

    if best_candidates is None:
        raise InfeasibleBestSubsetSupportError(
            "best_subset did not produce any feasible support; "
            f"all {infeasible_count or len(supports)} evaluated supports were infeasible."
        )
    return best_candidates, best_value


def _beam_settings(config: OptimizeConfig) -> tuple[int, int, int]:
    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    width = int(
        optimizer_kwargs.get(
            BEST_SUBSET_BEAM_WIDTH_KWARG,
            DEFAULT_BEST_SUBSET_BEAM_WIDTH,
        )
    )
    steps = int(
        optimizer_kwargs.get(
            BEST_SUBSET_BEAM_STEPS_KWARG,
            DEFAULT_BEST_SUBSET_BEAM_STEPS,
        )
    )
    max_evaluations = int(
        optimizer_kwargs.get(
            BEST_SUBSET_MAX_EVALUATIONS_KWARG,
            DEFAULT_BEST_SUBSET_MAX_EVALUATIONS,
        )
    )
    if width < 1:
        raise ValueError(f"{BEST_SUBSET_BEAM_WIDTH_KWARG} must be >= 1.")
    if steps < 0:
        raise ValueError(f"{BEST_SUBSET_BEAM_STEPS_KWARG} must be >= 0.")
    if max_evaluations < 1:
        raise ValueError(f"{BEST_SUBSET_MAX_EVALUATIONS_KWARG} must be >= 1.")
    return width, steps, max_evaluations


def _canonical_support(
    indices: set[int] | frozenset[int],
    comp_idx: Sequence[int],
) -> tuple[int, ...]:
    return tuple(index for index in comp_idx if index in indices)


def _topk_seed_support(
    *,
    support_k: int,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    problem: _SupportProblem,
    optimize_one: OptimizeOne,
) -> tuple[int, ...]:
    """Build one shared seed support for a requested cardinality."""
    choose = support_k - len(problem.required)
    if choose == 0:
        return _canonical_support(problem.required, problem.comp_idx)

    repair = config.repair_config
    if repair is None:
        raise ValueError("best_subset requires repair_config.")

    fixed = _merged_fixed_features(config)
    seed_repair = replace(
        repair,
        k=support_k,
        support_selection="topk",
        fixed_features=fixed,
    )
    seed_config = replace(
        config,
        repair_config=seed_repair,
        fixed_features=fixed,
        optimizer_kwargs=_inner_optimizer_kwargs(config),
    )
    candidates, _ = optimize_one(
        acqf=acqf,
        bounds=bounds,
        config=seed_config,
    )

    import torch

    X = candidates if torch.is_tensor(candidates) else torch.as_tensor(candidates)
    idx_t = torch.as_tensor(problem.comp_idx, device=X.device, dtype=torch.long)
    group = X.index_select(dim=-1, index=idx_t)
    scores = group.abs() if repair.score == "abs" else group
    if scores.ndim == 1:
        aggregate = scores
    else:
        aggregate = scores.reshape(-1, scores.shape[-1]).mean(dim=0)

    position = {index: pos for pos, index in enumerate(problem.comp_idx)}
    ranked_free = sorted(
        problem.free,
        key=lambda index: (
            -float(aggregate[position[index]].item()),
            position[index],
        ),
    )
    active = set(problem.required) | set(ranked_free[:choose])
    return _canonical_support(active, problem.comp_idx)


def _support_neighbors(
    support: tuple[int, ...],
    problem: _SupportProblem,
) -> list[tuple[int, ...]]:
    """Return swap, add, and drop neighbors inside the cardinality range."""
    active = set(support)
    removable = [index for index in support if index not in problem.required]
    addable = [index for index in problem.free if index not in active]
    neighbors: set[tuple[int, ...]] = set()

    for drop in removable:
        for add in addable:
            candidate = (active - {drop}) | {add}
            neighbors.add(_canonical_support(candidate, problem.comp_idx))

    if len(support) < problem.max_k:
        for add in addable:
            neighbors.add(_canonical_support(active | {add}, problem.comp_idx))

    if len(support) > problem.min_k:
        for drop in removable:
            neighbors.add(_canonical_support(active - {drop}, problem.comp_idx))

    return list(neighbors)


def _support_order(
    support: tuple[int, ...],
    problem: _SupportProblem,
) -> tuple[int, ...]:
    position = {index: pos for pos, index in enumerate(problem.comp_idx)}
    return (len(support), *(position[index] for index in support))


def _optimize_beam_best_subset(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    """Approximate support search with swap/add/drop beam neighborhoods."""
    problem = _support_problem(config)
    width, steps, max_evaluations = _beam_settings(config)
    if max_evaluations < len(problem.cardinalities):
        raise ValueError(
            f"{BEST_SUBSET_MAX_EVALUATIONS_KWARG} must be at least the number of "
            "allowed cardinalities so beam search can seed each k once. "
            f"Got max_evaluations={max_evaluations}, "
            f"cardinalities={len(problem.cardinalities)}."
        )

    seeds = [
        _topk_seed_support(
            support_k=support_k,
            acqf=acqf,
            bounds=bounds,
            config=config,
            problem=problem,
            optimize_one=optimize_one,
        )
        for support_k in problem.cardinalities
    ]
    seeds = list(dict.fromkeys(seeds))

    records: dict[tuple[int, ...], tuple[Any, Any, float]] = {}
    visited: set[tuple[int, ...]] = set()
    for seed in seeds:
        visited.add(seed)
        try:
            records[seed] = _evaluate_support(
                support=seed,
                acqf=acqf,
                bounds=bounds,
                config=config,
                optimize_one=optimize_one,
            )
        except InfeasibleBestSubsetSupportError:
            continue

    if not records:
        raise InfeasibleBestSubsetSupportError(
            "best_subset beam search did not produce any feasible seed support."
        )

    beam = sorted(
        records,
        key=lambda support: (
            -records[support][2],
            _support_order(support, problem),
        ),
    )[:width]

    for _ in range(steps):
        remaining = max_evaluations - len(visited)
        if remaining <= 0:
            break

        neighbor_set: set[tuple[int, ...]] = set()
        for support in beam:
            neighbor_set.update(_support_neighbors(support, problem))
        candidates_to_evaluate = sorted(
            neighbor_set - visited,
            key=lambda support: _support_order(support, problem),
        )
        if not candidates_to_evaluate:
            break

        evaluated: list[tuple[int, ...]] = []
        for support in candidates_to_evaluate[:remaining]:
            visited.add(support)
            try:
                records[support] = _evaluate_support(
                    support=support,
                    acqf=acqf,
                    bounds=bounds,
                    config=config,
                    optimize_one=optimize_one,
                )
            except InfeasibleBestSubsetSupportError:
                continue
            evaluated.append(support)

        pool = set(beam) | set(evaluated)
        if not pool:
            break
        beam = sorted(
            pool,
            key=lambda support: (
                -records[support][2],
                _support_order(support, problem),
            ),
        )[:width]

    best_support = min(
        records,
        key=lambda support: (
            -records[support][2],
            _support_order(support, problem),
        ),
    )
    best_candidates, best_value, _ = records[best_support]
    return best_candidates, best_value


def optimize_best_subset_candidates(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    """Optimize an acquisition function over sparse supports.

    By default the search remains exact-k and uses ``repair_config.k``. Setting
    ``optimizer_kwargs['best_subset_min_k']`` and ``['best_subset_max_k']``
    enables variable-cardinality search. Exact mode enumerates every support
    across the range. Beam mode seeds every allowed cardinality and explores
    swap/add/drop neighbors. Auto compares the summed support count across the
    whole range against ``best_subset_max_combinations``.

    For q > 1, one support is shared by the entire q-batch. Every support is
    compared by re-evaluating the acquisition on its final repaired candidate.
    A support-specific ``InfeasibleBestSubsetSupportError`` is skipped without
    hiding unrelated optimizer or configuration failures.
    """
    if not uses_best_subset(config):
        return optimize_one(acqf=acqf, bounds=bounds, config=config)
    if not config.return_best_only:
        raise ValueError("best_subset currently requires OptimizeConfig.return_best_only=True.")

    problem = _support_problem(config)
    strategy = _resolve_strategy(config, problem)
    if strategy == "exact":
        return _optimize_exact_best_subset(
            acqf=acqf,
            bounds=bounds,
            config=config,
            optimize_one=optimize_one,
        )
    return _optimize_beam_best_subset(
        acqf=acqf,
        bounds=bounds,
        config=config,
        optimize_one=optimize_one,
    )


__all__ = [
    "InfeasibleBestSubsetSupportError",
    "enumerate_best_subset_supports",
    "optimize_best_subset_candidates",
    "uses_best_subset",
]
