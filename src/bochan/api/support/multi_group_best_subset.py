"""Acquisition-aware Best Subset search over multiple independent sparse groups."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, product
from math import comb, prod
from typing import Any

from ..configs import OptimizeConfig
from .best_subset import (
    BEST_SUBSET_BEAM_STEPS_KWARG,
    BEST_SUBSET_BEAM_WIDTH_KWARG,
    BEST_SUBSET_MAX_COMBINATIONS_KWARG,
    BEST_SUBSET_MAX_EVALUATIONS_KWARG,
    BEST_SUBSET_MAX_K_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    BEST_SUBSET_STRATEGY_KWARG,
    DEFAULT_BEST_SUBSET_BEAM_STEPS,
    DEFAULT_BEST_SUBSET_BEAM_WIDTH,
    DEFAULT_BEST_SUBSET_MAX_COMBINATIONS,
    DEFAULT_BEST_SUBSET_MAX_EVALUATIONS,
    DEFAULT_BEST_SUBSET_STRATEGY,
    InfeasibleBestSubsetSupportError,
)

BEST_SUBSET_GROUPS_KWARG = "best_subset_groups"
OptimizeOne = Callable[..., tuple[Any, Any]]

_SUPPORT_SEARCH_KWARGS = {
    BEST_SUBSET_GROUPS_KWARG,
    BEST_SUBSET_MAX_COMBINATIONS_KWARG,
    BEST_SUBSET_STRATEGY_KWARG,
    BEST_SUBSET_BEAM_WIDTH_KWARG,
    BEST_SUBSET_BEAM_STEPS_KWARG,
    BEST_SUBSET_MAX_EVALUATIONS_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    BEST_SUBSET_MAX_K_KWARG,
}


@dataclass(frozen=True)
class _GroupProblem:
    name: str
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


@dataclass(frozen=True)
class _GroupedProblem:
    groups: tuple[_GroupProblem, ...]

    @property
    def sparse_indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group.comp_idx)

    @property
    def count(self) -> int:
        return int(prod(group.count for group in self.groups))

    @property
    def cardinality_combinations(self) -> tuple[tuple[int, ...], ...]:
        return tuple(product(*(group.cardinalities for group in self.groups)))


def uses_grouped_best_subset(config: OptimizeConfig) -> bool:
    """Return whether ``config`` requests multi-group Best Subset search."""

    repair = config.repair_config
    groups = (config.optimizer_kwargs or {}).get(BEST_SUBSET_GROUPS_KWARG)
    return bool(
        repair is not None
        and repair.support_selection == "best_subset"
        and groups
    )


def _merged_fixed_features(config: OptimizeConfig) -> dict[int, float]:
    merged = {
        int(key): float(value)
        for key, value in (config.fixed_features or {}).items()
    }
    repair = config.repair_config
    if repair is not None:
        for key, value in (repair.fixed_features or {}).items():
            merged[int(key)] = float(value)
    return merged


def _group_specs(config: OptimizeConfig) -> list[Mapping[str, Any]]:
    raw = (config.optimizer_kwargs or {}).get(BEST_SUBSET_GROUPS_KWARG)
    if raw is None:
        return []
    if isinstance(raw, Mapping) or isinstance(raw, (str, bytes)):
        raise TypeError(
            f"optimizer_kwargs['{BEST_SUBSET_GROUPS_KWARG}'] must be a sequence of group mappings."
        )
    specs = list(raw)
    if not specs:
        raise ValueError(f"{BEST_SUBSET_GROUPS_KWARG} must contain at least one group.")
    if any(not isinstance(spec, Mapping) for spec in specs):
        raise TypeError(f"Each {BEST_SUBSET_GROUPS_KWARG} entry must be a mapping.")
    return specs


def _parse_k_range(spec: Mapping[str, Any], *, name: str) -> tuple[int, int]:
    exact = spec.get("k")
    minimum_raw = spec.get("min_k", exact)
    maximum_raw = spec.get("max_k", exact)
    if minimum_raw is None or maximum_raw is None:
        raise ValueError(
            f"Best Subset group {name!r} must define k or both min_k and max_k."
        )
    minimum = int(minimum_raw)
    maximum = int(maximum_raw)
    if minimum < 0:
        raise ValueError(f"Best Subset group {name!r} requires min_k >= 0.")
    if maximum < minimum:
        raise ValueError(
            f"Best Subset group {name!r} requires max_k >= min_k."
        )
    return minimum, maximum


def _validate_fixed_features_list(
    values: Sequence[Mapping[int, float]] | None,
    sparse_indices: Sequence[int],
) -> None:
    if values is None:
        return
    sparse = {int(index) for index in sparse_indices}
    conflicting = sorted(
        {
            int(key)
            for item in values
            for key in item
            if int(key) in sparse
        }
    )
    if conflicting:
        raise ValueError(
            "best_subset does not support fixed_features_list entries on grouped "
            f"sparse dimensions. Conflicting indices: {conflicting}."
        )


def _problem(config: OptimizeConfig) -> _GroupedProblem:
    repair = config.repair_config
    if repair is None or repair.support_selection != "best_subset":
        raise ValueError(
            "Grouped best_subset requires repair_config.support_selection='best_subset'."
        )

    optimizer_kwargs = dict(config.optimizer_kwargs or {})
    for key in (BEST_SUBSET_MIN_K_KWARG, BEST_SUBSET_MAX_K_KWARG):
        if key in optimizer_kwargs:
            raise ValueError(
                f"{key} is ambiguous with {BEST_SUBSET_GROUPS_KWARG}; put min_k/max_k "
                "inside each group specification instead."
            )

    fixed = _merged_fixed_features(config)
    groups: list[_GroupProblem] = []
    seen_names: set[str] = set()
    seen_indices: set[int] = set()
    for position, spec in enumerate(_group_specs(config)):
        name = str(spec.get("name", f"group_{position}"))
        if name in seen_names:
            raise ValueError(f"Duplicate Best Subset group name {name!r}.")
        seen_names.add(name)

        raw_indices = spec.get("comp_idx")
        if raw_indices is None:
            raise ValueError(f"Best Subset group {name!r} requires comp_idx.")
        comp_idx = tuple(int(index) for index in raw_indices)
        if not comp_idx:
            raise ValueError(f"Best Subset group {name!r} requires non-empty comp_idx.")
        if len(set(comp_idx)) != len(comp_idx):
            raise ValueError(f"Best Subset group {name!r} comp_idx contains duplicates.")
        overlap = seen_indices & set(comp_idx)
        if overlap:
            raise ValueError(
                "Best Subset groups must be disjoint. Overlapping indices: "
                f"{sorted(overlap)}."
            )
        seen_indices.update(comp_idx)

        minimum, maximum = _parse_k_range(spec, name=name)
        if maximum > len(comp_idx):
            raise ValueError(
                f"Best Subset group {name!r} max_k={maximum} exceeds "
                f"len(comp_idx)={len(comp_idx)}."
            )

        required = frozenset(
            index for index in comp_idx if fixed.get(index, 0.0) != 0.0
        )
        forbidden = frozenset(
            index for index in comp_idx if index in fixed and fixed[index] == 0.0
        )
        free = tuple(
            index
            for index in comp_idx
            if index not in required and index not in forbidden
        )
        effective_minimum = max(minimum, len(required))
        effective_maximum = min(maximum, len(required) + len(free))
        if effective_minimum > effective_maximum:
            raise ValueError(
                f"Best Subset group {name!r} cannot satisfy cardinality range "
                f"[{minimum}, {maximum}] after fixed-feature rules."
            )
        groups.append(
            _GroupProblem(
                name=name,
                comp_idx=comp_idx,
                min_k=effective_minimum,
                max_k=effective_maximum,
                required=required,
                forbidden=forbidden,
                free=free,
                min_choose=effective_minimum - len(required),
                max_choose=effective_maximum - len(required),
            )
        )

    problem = _GroupedProblem(tuple(groups))
    repair_indices = tuple(int(index) for index in (repair.comp_idx or ()))
    if set(repair_indices) != set(problem.sparse_indices):
        raise ValueError(
            "Grouped best_subset requires repair_config.comp_idx to contain exactly "
            "the union of all best_subset_groups comp_idx values."
        )
    _validate_fixed_features_list(config.fixed_features_list, problem.sparse_indices)
    return problem


def _max_combinations(config: OptimizeConfig) -> int:
    value = int(
        (config.optimizer_kwargs or {}).get(
            BEST_SUBSET_MAX_COMBINATIONS_KWARG,
            DEFAULT_BEST_SUBSET_MAX_COMBINATIONS,
        )
    )
    if value < 1:
        raise ValueError(f"{BEST_SUBSET_MAX_COMBINATIONS_KWARG} must be >= 1.")
    return value


def _resolve_strategy(config: OptimizeConfig, problem: _GroupedProblem) -> str:
    strategy = str(
        (config.optimizer_kwargs or {}).get(
            BEST_SUBSET_STRATEGY_KWARG,
            DEFAULT_BEST_SUBSET_STRATEGY,
        )
    ).lower()
    if strategy not in {"exact", "beam", "auto"}:
        raise ValueError(
            f"{BEST_SUBSET_STRATEGY_KWARG} must be one of exact, beam, or auto."
        )
    if strategy == "auto":
        return "exact" if problem.count <= _max_combinations(config) else "beam"
    return strategy


def _group_supports(group: _GroupProblem) -> list[tuple[int, ...]]:
    supports: list[tuple[int, ...]] = []
    for choose in group.choose_counts:
        for selected in combinations(group.free, choose):
            active = set(group.required) | set(selected)
            supports.append(
                tuple(index for index in group.comp_idx if index in active)
            )
    return supports


def _flatten_supports(
    local_supports: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    return tuple(index for support in local_supports for index in support)


def enumerate_grouped_best_subset_supports(
    config: OptimizeConfig,
) -> list[tuple[int, ...]]:
    """Enumerate the Cartesian product of all group-local feasible supports."""

    problem = _problem(config)
    maximum = _max_combinations(config)
    if problem.count > maximum:
        raise ValueError(
            "grouped best_subset exact enumeration would evaluate "
            f"{problem.count} support combinations, exceeding "
            f"{BEST_SUBSET_MAX_COMBINATIONS_KWARG}={maximum}."
        )
    return [
        _flatten_supports(local)
        for local in product(*(_group_supports(group) for group in problem.groups))
    ]


def _inner_optimizer_kwargs(config: OptimizeConfig) -> dict[str, Any]:
    values = dict(config.optimizer_kwargs or {})
    for key in _SUPPORT_SEARCH_KWARGS:
        values.pop(key, None)
    return values


def _config_for_support(
    config: OptimizeConfig,
    problem: _GroupedProblem,
    support: Sequence[int],
) -> OptimizeConfig:
    repair = config.repair_config
    if repair is None:
        raise ValueError("grouped best_subset requires repair_config.")
    support_tuple = tuple(int(index) for index in support)
    support_set = set(support_tuple)
    fixed = _merged_fixed_features(config)
    for index in problem.sparse_indices:
        if index not in support_set:
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
    if hasattr(value, "detach"):
        detached = value.detach()
        if int(detached.numel()) != 1:
            raise ValueError(
                "grouped best_subset requires one scalar acquisition value per support."
            )
        return float(detached.reshape(-1)[0].item())
    return float(value)


def _evaluate_support(
    *,
    support: Sequence[int],
    problem: _GroupedProblem,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any, float]:
    inner_config = _config_for_support(config, problem, support)
    candidates, _ = optimize_one(acqf=acqf, bounds=bounds, config=inner_config)
    value = acqf(candidates)
    if hasattr(value, "detach"):
        value = value.detach()
    return candidates, value, _scalar_acquisition_value(value)


def _optimize_exact(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    problem: _GroupedProblem,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    supports = enumerate_grouped_best_subset_supports(config)
    best_candidates: Any | None = None
    best_value: Any | None = None
    best_score: float | None = None
    infeasible = 0
    for support in supports:
        try:
            candidates, value, score = _evaluate_support(
                support=support,
                problem=problem,
                acqf=acqf,
                bounds=bounds,
                config=config,
                optimize_one=optimize_one,
            )
        except InfeasibleBestSubsetSupportError:
            infeasible += 1
            continue
        if best_score is None or score > best_score:
            best_candidates, best_value, best_score = candidates, value, score
    if best_candidates is None:
        raise InfeasibleBestSubsetSupportError(
            "grouped best_subset did not produce any feasible support; "
            f"all {infeasible or len(supports)} evaluated combinations were infeasible."
        )
    return best_candidates, best_value


def _beam_settings(config: OptimizeConfig) -> tuple[int, int, int]:
    kwargs = dict(config.optimizer_kwargs or {})
    width = int(kwargs.get(BEST_SUBSET_BEAM_WIDTH_KWARG, DEFAULT_BEST_SUBSET_BEAM_WIDTH))
    steps = int(kwargs.get(BEST_SUBSET_BEAM_STEPS_KWARG, DEFAULT_BEST_SUBSET_BEAM_STEPS))
    evaluations = int(
        kwargs.get(BEST_SUBSET_MAX_EVALUATIONS_KWARG, DEFAULT_BEST_SUBSET_MAX_EVALUATIONS)
    )
    if width < 1:
        raise ValueError(f"{BEST_SUBSET_BEAM_WIDTH_KWARG} must be >= 1.")
    if steps < 0:
        raise ValueError(f"{BEST_SUBSET_BEAM_STEPS_KWARG} must be >= 0.")
    if evaluations < 1:
        raise ValueError(f"{BEST_SUBSET_MAX_EVALUATIONS_KWARG} must be >= 1.")
    return width, steps, evaluations


def _canonical_local(indices: set[int], group: _GroupProblem) -> tuple[int, ...]:
    return tuple(index for index in group.comp_idx if index in indices)


def _rank_group_free_indices(
    candidates: Any,
    group: _GroupProblem,
    *,
    score: str,
) -> list[int]:
    import torch

    X = candidates if torch.is_tensor(candidates) else torch.as_tensor(candidates)
    idx = torch.as_tensor(group.comp_idx, device=X.device, dtype=torch.long)
    values = X.index_select(dim=-1, index=idx)
    scores = values.abs() if score == "abs" else values
    aggregate = scores if scores.ndim == 1 else scores.reshape(-1, scores.shape[-1]).mean(dim=0)
    position = {index: pos for pos, index in enumerate(group.comp_idx)}
    return sorted(
        group.free,
        key=lambda index: (-float(aggregate[position[index]].item()), position[index]),
    )


def _beam_seed_supports(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    problem: _GroupedProblem,
    optimize_one: OptimizeOne,
) -> list[tuple[int, ...]]:
    """Build one acquisition-optimized dense ranking and seed every k combination."""

    repair = config.repair_config
    if repair is None:
        raise ValueError("grouped best_subset requires repair_config.")
    fixed = _merged_fixed_features(config)
    dense_repair = replace(
        repair,
        comp_idx=None,
        k=0,
        support_selection="topk",
        fixed_features=fixed,
    )
    replacements: dict[str, Any] = {}
    if hasattr(config, "final_candidate_postprocess"):
        replacements["final_candidate_postprocess"] = None
    seed_config = replace(
        config,
        repair_config=dense_repair,
        fixed_features=fixed,
        optimizer_kwargs=_inner_optimizer_kwargs(config),
        **replacements,
    )
    candidates, _ = optimize_one(acqf=acqf, bounds=bounds, config=seed_config)
    rankings = [
        _rank_group_free_indices(candidates, group, score=str(repair.score))
        for group in problem.groups
    ]

    seeds: list[tuple[int, ...]] = []
    for cardinalities in problem.cardinality_combinations:
        local: list[tuple[int, ...]] = []
        for group, ranking, support_k in zip(
            problem.groups, rankings, cardinalities, strict=True
        ):
            choose = support_k - len(group.required)
            active = set(group.required) | set(ranking[:choose])
            local.append(_canonical_local(active, group))
        seeds.append(_flatten_supports(local))
    return list(dict.fromkeys(seeds))


def _local_support(support: Sequence[int], group: _GroupProblem) -> tuple[int, ...]:
    active = set(int(index) for index in support)
    return tuple(index for index in group.comp_idx if index in active)


def _local_neighbors(
    support: tuple[int, ...],
    group: _GroupProblem,
) -> list[tuple[int, ...]]:
    active = set(support)
    removable = [index for index in support if index not in group.required]
    addable = [index for index in group.free if index not in active]
    neighbors: set[tuple[int, ...]] = set()
    for drop in removable:
        for add in addable:
            neighbors.add(_canonical_local((active - {drop}) | {add}, group))
    if len(support) < group.max_k:
        for add in addable:
            neighbors.add(_canonical_local(active | {add}, group))
    if len(support) > group.min_k:
        for drop in removable:
            neighbors.add(_canonical_local(active - {drop}, group))
    return list(neighbors)


def _support_neighbors(
    support: tuple[int, ...],
    problem: _GroupedProblem,
) -> list[tuple[int, ...]]:
    local = [_local_support(support, group) for group in problem.groups]
    neighbors: set[tuple[int, ...]] = set()
    for group_index, group in enumerate(problem.groups):
        for replacement in _local_neighbors(local[group_index], group):
            candidate = list(local)
            candidate[group_index] = replacement
            neighbors.add(_flatten_supports(candidate))
    return list(neighbors)


def _support_order(
    support: tuple[int, ...],
    problem: _GroupedProblem,
) -> tuple[int, ...]:
    order: list[int] = []
    for group in problem.groups:
        local = _local_support(support, group)
        position = {index: pos for pos, index in enumerate(group.comp_idx)}
        order.append(len(local))
        order.extend(position[index] for index in local)
    return tuple(order)


def _optimize_beam(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    problem: _GroupedProblem,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    width, steps, max_evaluations = _beam_settings(config)
    seeds = _beam_seed_supports(
        acqf=acqf,
        bounds=bounds,
        config=config,
        problem=problem,
        optimize_one=optimize_one,
    )
    if max_evaluations < len(seeds):
        raise ValueError(
            f"{BEST_SUBSET_MAX_EVALUATIONS_KWARG} must be at least the number of "
            "group-cardinality seed combinations. "
            f"Got max_evaluations={max_evaluations}, seeds={len(seeds)}."
        )

    records: dict[tuple[int, ...], tuple[Any, Any, float]] = {}
    visited: set[tuple[int, ...]] = set()
    for seed in seeds:
        visited.add(seed)
        try:
            records[seed] = _evaluate_support(
                support=seed,
                problem=problem,
                acqf=acqf,
                bounds=bounds,
                config=config,
                optimize_one=optimize_one,
            )
        except InfeasibleBestSubsetSupportError:
            continue

    if not records:
        frontier = list(seeds)
        while frontier and len(visited) < max_evaluations and not records:
            neighbors: set[tuple[int, ...]] = set()
            for support in frontier:
                neighbors.update(_support_neighbors(support, problem))
            pending = sorted(
                neighbors - visited,
                key=lambda item: _support_order(item, problem),
            )
            if not pending:
                break
            frontier = []
            for support in pending:
                if len(visited) >= max_evaluations:
                    break
                visited.add(support)
                frontier.append(support)
                try:
                    records[support] = _evaluate_support(
                        support=support,
                        problem=problem,
                        acqf=acqf,
                        bounds=bounds,
                        config=config,
                        optimize_one=optimize_one,
                    )
                except InfeasibleBestSubsetSupportError:
                    continue
        if not records:
            raise InfeasibleBestSubsetSupportError(
                "grouped best_subset beam search did not find a feasible seed within "
                f"the evaluation budget ({max_evaluations})."
            )

    beam = sorted(
        records,
        key=lambda item: (-records[item][2], _support_order(item, problem)),
    )[:width]
    for _ in range(steps):
        remaining = max_evaluations - len(visited)
        if remaining <= 0:
            break
        neighbors: set[tuple[int, ...]] = set()
        for support in beam:
            neighbors.update(_support_neighbors(support, problem))
        pending = sorted(
            neighbors - visited,
            key=lambda item: _support_order(item, problem),
        )
        if not pending:
            break
        evaluated: list[tuple[int, ...]] = []
        for support in pending[:remaining]:
            visited.add(support)
            try:
                records[support] = _evaluate_support(
                    support=support,
                    problem=problem,
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
            key=lambda item: (-records[item][2], _support_order(item, problem)),
        )[:width]

    best = min(
        records,
        key=lambda item: (-records[item][2], _support_order(item, problem)),
    )
    candidates, value, _ = records[best]
    return candidates, value


def optimize_grouped_best_subset_candidates(
    *,
    acqf: Any,
    bounds: Any,
    config: OptimizeConfig,
    optimize_one: OptimizeOne,
) -> tuple[Any, Any]:
    """Optimize a joint acquisition over independent sparse-group supports.

    Exact mode evaluates the Cartesian product of each group's local supports.
    Beam mode seeds every allowed group-cardinality combination and changes one
    group at a time with swap/add/drop moves. One complete grouped support is
    shared by the entire q-batch, and every evaluated support is ranked by the
    acquisition value of its final repaired candidate.
    """

    if not uses_grouped_best_subset(config):
        return optimize_one(acqf=acqf, bounds=bounds, config=config)
    if not config.return_best_only:
        raise ValueError(
            "grouped best_subset currently requires OptimizeConfig.return_best_only=True."
        )
    problem = _problem(config)
    strategy = _resolve_strategy(config, problem)
    if strategy == "exact":
        return _optimize_exact(
            acqf=acqf,
            bounds=bounds,
            config=config,
            problem=problem,
            optimize_one=optimize_one,
        )
    return _optimize_beam(
        acqf=acqf,
        bounds=bounds,
        config=config,
        problem=problem,
        optimize_one=optimize_one,
    )


__all__ = [
    "BEST_SUBSET_GROUPS_KWARG",
    "enumerate_grouped_best_subset_supports",
    "optimize_grouped_best_subset_candidates",
    "uses_grouped_best_subset",
]
