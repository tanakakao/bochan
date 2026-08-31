"""Multiple fixed-total Fraction composition groups for Best Subset search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import prod
from typing import Any

from bochan.api import CandidateRepairConfig, OptimizeConfig
from bochan.api.support.multi_group_best_subset import BEST_SUBSET_GROUPS_KWARG

from .cardinality import (
    BEST_SUBSET_MAX_K_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    require_exact_cardinality_for_steps,
    resolve_composition_cardinality_range,
    support_count,
)
from .support import (
    _BEST_SUBSET_SITE_KWARGS,
    _TOLERANCE,
    _active_floor,
    _as_mapping,
    _component_bounds,
    _composition_fixed_values,
    _grid_config_with_fixed_fractions,
    _grid_postprocess,
    _merge_fixed_features,
    _to_ge_constraints,
    _validate_all_supports_feasible,
    _validate_grid_contract,
    _validate_grid_strategy,
    _validate_grid_supports,
    _validate_site,
    _without_cardinality_kwargs,
)


def _merge_search_controls(
    opt_config: OptimizeConfig,
    site_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge site search controls while keeping explicit optimizer kwargs authoritative."""

    result = dict(opt_config.optimizer_kwargs or {})
    for key in (BEST_SUBSET_MIN_K_KWARG, BEST_SUBSET_MAX_K_KWARG):
        if key in result:
            raise ValueError(
                f"{key} is ambiguous for multiple composition Best Subset groups. "
                "Use each site's min_components/max_components instead."
            )
    result.pop(BEST_SUBSET_GROUPS_KWARG, None)
    explicit = set(result)

    for key in _BEST_SUBSET_SITE_KWARGS:
        if key in explicit:
            continue
        values = [config.get(key) for config in site_configs if config.get(key) is not None]
        if not values:
            continue
        normalized = {
            str(value).lower() if key == "best_subset_strategy" else int(value)
            for value in values
        }
        if len(normalized) > 1:
            raise ValueError(
                f"Multiple composition Best Subset sites configure conflicting {key} "
                f"values: {values!r}. Set OptimizeConfig.optimizer_kwargs[{key!r}] "
                "explicitly to choose one shared search policy."
            )
        result[key] = values[0]
    return result


def _constraint_items(values: Any) -> list[Any]:
    if isinstance(values, (str, int)):
        return [values]
    if hasattr(values, "detach"):
        return values.detach().cpu().reshape(-1).tolist()
    return list(values)


def _constraint_feature_names(
    indices: Any,
    feature_names: Sequence[Any],
) -> set[str]:
    names: set[str] = set()
    for item in _constraint_items(indices):
        if isinstance(item, str):
            names.add(item)
            continue
        position = int(item)
        if position < 0 or position >= len(feature_names):
            raise IndexError(
                f"Constraint feature index {position} is outside dimension {len(feature_names)}."
            )
        names.add(str(feature_names[position]))
    return names


def _validate_no_cross_site_stepped_constraints(
    *,
    opt_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    stepped_fraction_names: Mapping[str, set[str]],
    feature_names: Sequence[Any],
) -> None:
    """Reject constraints whose sequential grid projection would be order-dependent."""

    if len(stepped_fraction_names) < 2:
        return
    constraints = [
        *(opt_config.equality_constraints or ()),
        *(opt_config.inequality_constraints or ()),
        *(repair.equality_constraints or ()),
        *(repair.inequality_constraints or ()),
    ]
    for indices, _coefficients, _rhs in constraints:
        names = _constraint_feature_names(indices, feature_names)
        touched = [
            site_name
            for site_name, site_features in stepped_fraction_names.items()
            if names & site_features
        ]
        if len(touched) > 1:
            raise ValueError(
                "A stepped linear constraint cannot couple two composition Best Subset "
                f"sites in the same projection chain. Touched sites: {touched!r}. "
                "Use a single stepped site in that constraint or remove one step grid."
            )


def resolve_multiple_composition_best_subset(
    opt_config: OptimizeConfig,
    *,
    selected_sites: Sequence[str],
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    feature_names: Sequence[Any],
) -> OptimizeConfig:
    """Wire multiple fixed-total Fraction sites into grouped Best Subset search.

    Each site's optional elements form one independent sparse group. Exact search
    evaluates the Cartesian product of group-local supports; Beam changes one group
    at a time. Required elements remain ordinary free-valued variables and forbidden
    elements are fixed to zero. The entire grouped support is shared by a joint q-batch.
    """

    site_names = tuple(str(name) for name in selected_sites)
    if len(site_names) < 2:
        raise ValueError("Multiple composition resolver requires at least two sites.")
    configs = [composition_sites[name] for name in site_names]
    for site_name, config in zip(site_names, configs, strict=True):
        _validate_site(site_name, config)

    repair = opt_config.repair_config or CandidateRepairConfig()
    if repair.comp_idx not in (None, [], ()):
        raise ValueError(
            "Multiple composition best_subset groups own CandidateRepairConfig.comp_idx. "
            "Remove the generic comp_idx setting."
        )
    if int(repair.k) != 0:
        raise ValueError(
            "Multiple composition best_subset groups derive cardinality from each site's "
            "min_components/max_components. Remove CandidateRepairConfig.k."
        )
    if repair.final_sum_constraint is not None:
        raise ValueError(
            "Multiple composition best_subset groups own their per-site sum constraints. "
            "Remove CandidateRepairConfig.final_sum_constraint."
        )

    positions = {str(name): index for index, name in enumerate(feature_names)}
    search_kwargs = _merge_search_controls(opt_config, configs)
    optimizer_fixed = _as_mapping(opt_config.fixed_features)
    repair_fixed = _as_mapping(repair.fixed_features)
    optimizer_equalities = list(opt_config.equality_constraints or ())
    optimizer_inequalities = list(opt_config.inequality_constraints or ())
    repair_equalities = list(repair.equality_constraints or ())
    repair_inequalities = _to_ge_constraints(
        repair.inequality_constraints,
        str(repair.inequality_sense),
    )
    final_postprocess = getattr(opt_config, "final_candidate_postprocess", None)

    site_metadata: list[dict[str, Any]] = []
    stepped_fraction_names: dict[str, set[str]] = {}
    fraction_feature_owners: dict[str, str] = {}
    for site_name, config in zip(site_names, configs, strict=True):
        transformer = composition_transformers.get(site_name)
        if transformer is None:
            raise RuntimeError(
                f"Composition transformer for site {site_name!r} is not fitted. Call fit() first."
            )
        elements = tuple(transformer.fitted_elements)
        fraction_names = tuple(
            f"{transformer.prefix}__fraction__{element}" for element in elements
        )
        missing = [name for name in fraction_names if name not in positions]
        if missing:
            raise KeyError(
                f"Composition fraction features for site {site_name!r} are missing from "
                f"the optimization dataset: {missing!r}."
            )
        overlapping = sorted(
            name for name in fraction_names if name in fraction_feature_owners
        )
        if overlapping:
            owners = sorted({fraction_feature_owners[name] for name in overlapping})
            raise ValueError(
                "Multiple composition Best Subset sites must use disjoint fraction "
                f"feature blocks. Site {site_name!r} overlaps sites {owners!r}: "
                f"{overlapping!r}."
            )
        fraction_feature_owners.update({name: site_name for name in fraction_names})
        if config.get("steps"):
            stepped_fraction_names[site_name] = set(fraction_names)
        site_metadata.append(
            {
                "site_name": site_name,
                "config": config,
                "transformer": transformer,
                "elements": elements,
                "fraction_names": fraction_names,
            }
        )

    _validate_no_cross_site_stepped_constraints(
        opt_config=opt_config,
        repair=repair,
        stepped_fraction_names=stepped_fraction_names,
        feature_names=feature_names,
    )

    group_specs: list[dict[str, Any]] = []
    optional_names_flat: list[str] = []
    support_counts: list[int] = []
    cardinality_seed_counts: list[int] = []

    for metadata in site_metadata:
        site_name = metadata["site_name"]
        config = metadata["config"]
        elements = metadata["elements"]
        fraction_names = metadata["fraction_names"]

        all_fixed = _merge_fixed_features(optimizer_fixed, repair_fixed)
        composition_fixed = _composition_fixed_values(
            all_fixed,
            composition_feature_names=fraction_names,
            feature_names=feature_names,
        )
        grid_config = _grid_config_with_fixed_fractions(
            config,
            elements=elements,
            fraction_names=fraction_names,
            fixed_values=composition_fixed,
        )

        site_opt = replace(
            opt_config,
            fixed_features=optimizer_fixed or None,
            equality_constraints=optimizer_equalities,
            inequality_constraints=optimizer_inequalities,
            optimizer_kwargs=search_kwargs,
            final_candidate_postprocess=final_postprocess,
        )
        site_repair = replace(
            repair,
            equality_constraints=repair_equalities,
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
        )
        _validate_grid_contract(
            opt_config=site_opt,
            repair=site_repair,
            config=grid_config,
            fraction_names=fraction_names,
            feature_names=feature_names,
            all_fixed=all_fixed,
        )

        explicit_required = set(config.get("required_components") or ())
        explicit_forbidden = set(config.get("forbidden_components") or ())
        required: set[str] = set(explicit_required)
        forbidden: set[str] = set(explicit_forbidden)
        by_element = dict(zip(elements, fraction_names, strict=True))
        for element, feature_name in by_element.items():
            lower, upper = _component_bounds(config, element)
            if lower > _TOLERANCE:
                required.add(element)
            if upper <= _TOLERANCE:
                forbidden.add(element)
            if feature_name in composition_fixed:
                if abs(composition_fixed[feature_name]) <= _TOLERANCE:
                    forbidden.add(element)
                else:
                    required.add(element)

        overlap = required & forbidden
        if overlap:
            raise ValueError(
                f"Composition site {site_name!r} has components both required and "
                f"forbidden: {sorted(overlap)!r}."
            )
        for element in forbidden:
            lower, _ = _component_bounds(config, element)
            if lower > _TOLERANCE:
                raise ValueError(
                    f"Forbidden component {element!r} has a positive lower bound at "
                    f"site {site_name!r}."
                )

        optional = [
            element
            for element in elements
            if element not in required and element not in forbidden
        ]
        ordered_required = sorted(required, key=elements.index)
        cardinality = resolve_composition_cardinality_range(
            config,
            required_count=len(required),
            optional_count=len(optional),
            context=f"Composition site {site_name!r} best_subset",
        )
        require_exact_cardinality_for_steps(
            grid_config,
            cardinality,
            context=f"Composition site {site_name!r} best_subset",
        )
        if grid_config.get("steps"):
            _validate_grid_strategy(
                config=grid_config,
                optimizer_kwargs=search_kwargs,
                optional_count=len(optional),
                optional_k=cardinality.optional_maximum,
            )
        for optional_k in cardinality.optional_cardinalities:
            _validate_all_supports_feasible(
                required=ordered_required,
                optional=optional,
                optional_k=optional_k,
                config=grid_config if grid_config.get("steps") else config,
            )

        projector = _grid_postprocess(
            opt_config=site_opt,
            repair=site_repair,
            config=grid_config,
            elements=elements,
            fraction_names=fraction_names,
            feature_names=feature_names,
            exact_k=cardinality.maximum,
        )
        if grid_config.get("steps"):
            _validate_grid_supports(
                projector=projector,
                config=grid_config,
                optimizer_kwargs=search_kwargs,
                required=ordered_required,
                optional=optional,
                optional_k=cardinality.optional_maximum,
            )
        final_postprocess = projector

        forbidden_fixed = {by_element[element]: 0.0 for element in forbidden}
        optimizer_fixed = _merge_fixed_features(optimizer_fixed, forbidden_fixed)
        repair_fixed = _merge_fixed_features(repair_fixed, forbidden_fixed)

        sum_constraint = (fraction_names, [1.0] * len(fraction_names), 1.0)
        optimizer_equalities.append(sum_constraint)
        repair_equalities.append(sum_constraint)

        for element in elements:
            if element in forbidden:
                continue
            feature_name = by_element[element]
            _, upper = _component_bounds(config, element)
            floor_value = _active_floor(config, element, upper)
            repair_inequalities.append(([feature_name], [1.0], floor_value))
            if element in required and feature_name not in optimizer_fixed:
                optimizer_inequalities.append(([feature_name], [1.0], floor_value))

        optional_names = [by_element[element] for element in optional]
        if cardinality.optional_maximum > 0:
            optional_indices = [positions[name] for name in optional_names]
            group_specs.append(
                {
                    "name": site_name,
                    "comp_idx": optional_indices,
                    "min_k": cardinality.optional_minimum,
                    "max_k": cardinality.optional_maximum,
                }
            )
            optional_names_flat.extend(optional_names)
            support_counts.append(support_count(len(optional), cardinality))
            cardinality_seed_counts.append(len(cardinality.optional_cardinalities))

    if group_specs:
        strategy = str(search_kwargs.get("best_subset_strategy", "exact")).lower()
        combination_count = int(prod(support_counts))
        max_combinations = int(search_kwargs.get("best_subset_max_combinations", 2000))
        resolved_strategy = (
            "exact"
            if strategy == "auto" and combination_count <= max_combinations
            else "beam" if strategy == "auto"
            else strategy
        )
        if resolved_strategy == "exact" and combination_count > max_combinations:
            raise ValueError(
                "Multiple composition best_subset exact enumeration would evaluate "
                f"{combination_count} support combinations, exceeding "
                f"best_subset_max_combinations={max_combinations}."
            )
        if resolved_strategy == "beam":
            required_seeds = int(prod(cardinality_seed_counts))
            max_evaluations = int(search_kwargs.get("best_subset_max_evaluations", 200))
            if max_evaluations < required_seeds:
                raise ValueError(
                    "best_subset_max_evaluations must cover every multi-group cardinality "
                    f"seed combination: required={required_seeds}, got={max_evaluations}."
                )

        optimizer_kwargs = dict(search_kwargs)
        optimizer_kwargs.pop(BEST_SUBSET_MIN_K_KWARG, None)
        optimizer_kwargs.pop(BEST_SUBSET_MAX_K_KWARG, None)
        optimizer_kwargs[BEST_SUBSET_GROUPS_KWARG] = tuple(group_specs)
        final_repair = replace(
            repair,
            comp_idx=optional_names_flat,
            k=sum(int(group["max_k"]) for group in group_specs),
            support_selection="best_subset",
            equality_constraints=repair_equalities,
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
            final_sum_constraint=None,
        )
    else:
        optimizer_kwargs = _without_cardinality_kwargs(opt_config.optimizer_kwargs)
        optimizer_kwargs.pop(BEST_SUBSET_GROUPS_KWARG, None)
        final_repair = replace(
            repair,
            comp_idx=None,
            k=0,
            support_selection="topk",
            equality_constraints=repair_equalities,
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
            final_sum_constraint=None,
        )

    return replace(
        opt_config,
        repair_config=final_repair,
        fixed_features=optimizer_fixed or None,
        equality_constraints=optimizer_equalities,
        inequality_constraints=optimizer_inequalities,
        optimizer_kwargs=optimizer_kwargs,
        final_candidate_postprocess=final_postprocess,
    )


__all__ = ["resolve_multiple_composition_best_subset"]
