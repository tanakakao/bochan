"""Composition-aware wiring for acquisition-based best-subset support search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import combinations
from math import comb
from typing import Any

from bochan.api import CandidateRepairConfig, OptimizeConfig

from .cardinality import (
    BEST_SUBSET_MAX_K_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    apply_optional_cardinality_range,
    require_exact_cardinality_for_steps,
    resolve_composition_cardinality_range,
)
from .grid import (
    CompositionGridFinalPostprocess,
    GridLinearConstraint,
    composition_grid_linear_constraints,
    merge_composition_grid_linear_constraints,
)

_SUPPORTED_REPRESENTATIONS = {"none", "fraction", "fractions"}
_TOLERANCE = 1e-8
_BEST_SUBSET_SITE_KWARGS = (
    "best_subset_strategy",
    "best_subset_max_combinations",
    "best_subset_beam_width",
    "best_subset_beam_steps",
    "best_subset_max_evaluations",
)
_NATIVE_FINAL_POSTPROCESS_BYPASS = {
    "nsgaii",
    "nsga2",
    "optimize_acqf_nsgaii",
    "thompson_sampling",
    "optimize_thompson_sampling",
    "thompson_sampling_mixed",
    "optimize_thompson_sampling_mixed",
    "llm_candidate_set",
    "optimize_acqf_llm",
    "optimize_acqf_llm_candidate_set",
}


def _as_mapping(value: Mapping[Any, Any] | None) -> dict[Any, float]:
    return {key: float(item) for key, item in (value or {}).items()}


def _merge_fixed_features(
    base: Mapping[Any, Any] | None,
    extra: Mapping[Any, Any] | None,
) -> dict[Any, float]:
    merged = _as_mapping(base)
    for key, value in (extra or {}).items():
        value = float(value)
        if key in merged and abs(merged[key] - value) > 1e-12:
            raise ValueError(
                f"Conflicting fixed values for composition feature {key!r}: "
                f"{merged[key]} and {value}."
            )
        merged[key] = value
    return merged


def _to_ge_constraints(
    constraints: Sequence[tuple[Any, Any, Any]] | None,
    sense: str,
) -> list[tuple[Any, Any, Any]]:
    """Normalize repair inequalities to the ``a^T x >= rhs`` convention."""
    if sense == "ge":
        return list(constraints or ())
    if sense != "le":
        raise ValueError("repair inequality_sense must be 'le' or 'ge'.")
    converted: list[tuple[Any, Any, Any]] = []
    for indices, coefficients, rhs in constraints or ():
        converted.append(
            (
                indices,
                [-float(value) for value in coefficients],
                -float(rhs),
            )
        )
    return converted


def _component_bounds(config: Mapping[str, Any], element: str) -> tuple[float, float]:
    total = float(config["total"])
    pair = tuple(config["bounds"].get(element, (0.0, total)))
    if len(pair) != 2:
        raise ValueError(
            f"Bounds for composition component {element!r} must have length 2."
        )
    lower, upper = map(float, pair)
    return lower, upper


def _active_floor(config: Mapping[str, Any], element: str, upper: float) -> float:
    total = float(config["total"])
    lower, _ = _component_bounds(config, element)
    if lower > _TOLERANCE:
        return lower / total
    step = (config.get("steps") or {}).get(element)
    if step is not None:
        return min(float(step), upper) / total
    return min(10.0 * _TOLERANCE, upper) / total


def _validate_all_supports_feasible(
    *,
    required: Sequence[str],
    optional: Sequence[str],
    optional_k: int,
    config: Mapping[str, Any],
) -> None:
    """Reject settings where some enumerated support cannot sum to one."""
    total = float(config["total"])
    required_lower = 0.0
    required_upper = 0.0
    for element in required:
        lower, upper = _component_bounds(config, element)
        required_lower += max(
            lower / total,
            _active_floor(config, element, upper),
        )
        required_upper += upper / total

    optional_floors = []
    optional_uppers = []
    for element in optional:
        _, upper = _component_bounds(config, element)
        optional_floors.append(_active_floor(config, element, upper))
        optional_uppers.append(upper / total)

    largest_floor_sum = sum(
        sorted(optional_floors, reverse=True)[:optional_k]
    )
    smallest_upper_sum = sum(sorted(optional_uppers)[:optional_k])
    if required_lower + largest_floor_sum > 1.0 + _TOLERANCE:
        raise ValueError(
            "Composition best_subset has a support whose active lower bounds cannot "
            "satisfy the unit composition sum."
        )
    if required_upper + smallest_upper_sum < 1.0 - _TOLERANCE:
        raise ValueError(
            "Composition best_subset has a support whose active upper bounds cannot "
            "satisfy the unit composition sum. Relax component upper bounds or reduce "
            "the candidate element set."
        )


def _validate_site(
    site_name: str,
    config: Mapping[str, Any],
) -> tuple[int, int]:
    representation = str(config["representation"]).lower()
    if representation not in _SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Composition site {site_name!r} uses representation={representation!r}. "
            "Acquisition-aware element best_subset currently requires 'fractions' "
            "(or 'none') because CLR/ALR/ILR coordinates do not correspond one-to-one "
            "to element presence."
        )
    if config.get("variable_total"):
        raise ValueError(
            f"Composition site {site_name!r} uses variable_total. Composition "
            "best_subset currently supports fixed-total sites only."
        )
    minimum = int(config["min_components"])
    maximum_raw = config.get("max_components")
    if maximum_raw is None:
        raise ValueError(
            f"Composition site {site_name!r} must set max_components for best_subset."
        )
    maximum = int(maximum_raw)
    if minimum < 1 or maximum < minimum:
        raise ValueError(
            f"Composition site {site_name!r} requires 1 <= min_components <= "
            "max_components for best_subset search."
        )
    return minimum, maximum


def _optimizer_kwargs_for_site(
    opt_config: OptimizeConfig,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge optional site search controls without overriding explicit optimizer kwargs."""
    optimizer_kwargs = dict(opt_config.optimizer_kwargs or {})
    for key in _BEST_SUBSET_SITE_KWARGS:
        value = config.get(key)
        if value is not None:
            optimizer_kwargs.setdefault(key, value)
    return optimizer_kwargs


def _without_cardinality_kwargs(values: Mapping[str, Any] | None) -> dict[str, Any]:
    optimizer_kwargs = dict(values or {})
    optimizer_kwargs.pop(BEST_SUBSET_MIN_K_KWARG, None)
    optimizer_kwargs.pop(BEST_SUBSET_MAX_K_KWARG, None)
    return optimizer_kwargs


def _validate_grid_strategy(
    *,
    config: Mapping[str, Any],
    optimizer_kwargs: Mapping[str, Any],
    optional_count: int,
    optional_k: int,
) -> None:
    """Keep step-grid search on exhaustive exact-cardinality supports for now."""
    if not config.get("steps") or optional_k == 0:
        return
    strategy = str(
        optimizer_kwargs.get("best_subset_strategy", "exact")
    ).lower()
    support_count = comb(optional_count, optional_k)
    maximum = int(
        optimizer_kwargs.get("best_subset_max_combinations", 2000)
    )
    if strategy == "auto":
        strategy = "exact" if support_count <= maximum else "beam"
    if strategy != "exact":
        raise ValueError(
            "Composition best_subset with component steps currently requires exact "
            "support search. Use best_subset_strategy='exact', or 'auto' only when "
            "the support count is within best_subset_max_combinations."
        )
    if support_count > maximum:
        raise ValueError(
            "Composition step-grid best_subset exact enumeration would evaluate "
            f"{support_count} supports, exceeding best_subset_max_combinations={maximum}."
        )


def _grid_linear_constraints(
    *,
    opt_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    config: Mapping[str, Any],
    fraction_names: Sequence[str],
    feature_names: Sequence[Any],
) -> tuple[GridLinearConstraint, ...]:
    """Collect composition-only constraints in the projector's amount units."""

    total = float(config["total"])
    optimizer_constraints = composition_grid_linear_constraints(
        equality_constraints=opt_config.equality_constraints,
        inequality_constraints=opt_config.inequality_constraints,
        feature_names=feature_names,
        composition_feature_names=fraction_names,
        inequality_sense="ge",
        rhs_scale=total,
        context="Composition step-grid best_subset",
    )
    repair_constraints = composition_grid_linear_constraints(
        equality_constraints=repair.equality_constraints,
        inequality_constraints=repair.inequality_constraints,
        feature_names=feature_names,
        composition_feature_names=fraction_names,
        inequality_sense=str(repair.inequality_sense),
        rhs_scale=total,
        context="Composition step-grid best_subset",
    )
    return merge_composition_grid_linear_constraints(
        optimizer_constraints,
        repair_constraints,
    )


def _validate_grid_contract(
    *,
    opt_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    config: Mapping[str, Any],
    fraction_names: Sequence[str],
    feature_names: Sequence[Any],
    all_fixed: Mapping[Any, float],
) -> None:
    if not config.get("steps"):
        return

    optimizer_name = str(opt_config.optimizer).replace("-", "_").lower()
    if optimizer_name in _NATIVE_FINAL_POSTPROCESS_BYPASS:
        raise ValueError(
            "Composition step-grid best_subset requires an optimizer backend that "
            "applies final_candidate_postprocess. Use optimize_acqf, evo, or torch."
        )
    if hasattr(opt_config, "ensure_unique_candidates") and not bool(
        opt_config.ensure_unique_candidates
    ):
        raise ValueError(
            "Composition step-grid best_subset requires ensure_unique_candidates=True "
            "so grid-projected candidates are re-evaluated."
        )

    nonzero_fixed = [
        name
        for name in fraction_names
        if name in all_fixed and abs(float(all_fixed[name])) > _TOLERANCE
    ]
    if nonzero_fixed:
        raise ValueError(
            "Composition step-grid best_subset does not yet support non-zero fixed "
            f"composition values: {nonzero_fixed!r}. Use component bounds/required "
            "elements instead."
        )

    _grid_linear_constraints(
        opt_config=opt_config,
        repair=repair,
        config=config,
        fraction_names=fraction_names,
        feature_names=feature_names,
    )


def _grid_postprocess(
    *,
    opt_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    config: Mapping[str, Any],
    elements: Sequence[str],
    fraction_names: Sequence[str],
    feature_names: Sequence[Any],
    exact_k: int,
) -> CompositionGridFinalPostprocess | Any:
    if not config.get("steps"):
        return getattr(opt_config, "final_candidate_postprocess", None)
    positions = {name: index for index, name in enumerate(feature_names)}
    indices = tuple(int(positions[name]) for name in fraction_names)
    return CompositionGridFinalPostprocess.from_config(
        feature_indices=indices,
        elements=elements,
        config=config,
        exact_k=exact_k,
        linear_constraints=_grid_linear_constraints(
            opt_config=opt_config,
            repair=repair,
            config=config,
            fraction_names=fraction_names,
            feature_names=feature_names,
        ),
        previous=getattr(opt_config, "final_candidate_postprocess", None),
    )


def _validate_grid_supports(
    *,
    projector: CompositionGridFinalPostprocess | Any,
    config: Mapping[str, Any],
    required: Sequence[str],
    optional: Sequence[str],
    optional_k: int,
) -> None:
    if not config.get("steps"):
        return
    for selected in combinations(optional, optional_k):
        projector.validate_support([*required, *selected])


def resolve_composition_best_subset(
    opt_config: OptimizeConfig,
    *,
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    feature_names: Sequence[Any],
) -> OptimizeConfig:
    """Wire one fraction-space composition site into generic best-subset search.

    Element support is defined in raw composition space. For fraction representation,
    each element has one model feature, so optional elements can be passed to the core
    sparse best-subset optimizer without confusing log-ratio coordinates for element
    presence. Required elements remain free-valued variables; forbidden elements are
    fixed to zero.

    ``min_components`` / ``max_components`` count all active elements. Required
    elements are removed from the generic sparse group and the residual optional-k
    range is passed to the core Best Subset engine. With component steps the current
    MILP projector remains exact-cardinality only.
    """
    selected_sites = [
        name
        for name, site in composition_sites.items()
        if str(site.get("support_selection", "repair")).lower() == "best_subset"
    ]
    if not selected_sites:
        return opt_config
    if len(selected_sites) > 1:
        raise ValueError(
            "Composition best_subset currently supports one composition site per "
            "candidate optimization because CandidateRepairConfig has one sparse group."
        )

    site_name = selected_sites[0]
    config = composition_sites[site_name]
    _validate_site(site_name, config)
    transformer = composition_transformers.get(site_name)
    if transformer is None:
        raise RuntimeError(
            f"Composition transformer for site {site_name!r} is not fitted. Call fit() first."
        )

    elements = tuple(transformer.fitted_elements)
    fraction_names = tuple(
        f"{transformer.prefix}__fraction__{element}" for element in elements
    )
    missing = [name for name in fraction_names if name not in feature_names]
    if missing:
        raise KeyError(
            f"Composition fraction features for site {site_name!r} are missing from the "
            f"optimization dataset: {missing!r}."
        )

    repair = opt_config.repair_config or CandidateRepairConfig()
    if repair.comp_idx not in (None, [], ()):
        raise ValueError(
            "Composition best_subset owns CandidateRepairConfig.comp_idx. Remove the "
            "generic comp_idx setting when composition support search is enabled."
        )
    if int(repair.k) != 0:
        raise ValueError(
            "Composition best_subset derives cardinality from min_components/"
            "max_components. Remove the generic CandidateRepairConfig.k setting."
        )
    if repair.final_sum_constraint is not None:
        raise ValueError(
            "Composition best_subset owns final_sum_constraint for the selected site."
        )

    optimizer_fixed = _as_mapping(opt_config.fixed_features)
    repair_fixed = _as_mapping(repair.fixed_features)
    all_fixed = _merge_fixed_features(optimizer_fixed, repair_fixed)
    _validate_grid_contract(
        opt_config=opt_config,
        repair=repair,
        config=config,
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
        if feature_name in all_fixed:
            if abs(all_fixed[feature_name]) <= _TOLERANCE:
                forbidden.add(element)
            else:
                required.add(element)

    overlap = required & forbidden
    if overlap:
        raise ValueError(
            f"Composition site {site_name!r} has components that are both required and "
            f"forbidden: {sorted(overlap)!r}."
        )
    for element in forbidden:
        lower, _ = _component_bounds(config, element)
        if lower > _TOLERANCE:
            raise ValueError(
                f"Forbidden component {element!r} has a positive lower bound at site "
                f"{site_name!r}."
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
        config,
        cardinality,
        context=f"Composition site {site_name!r} best_subset",
    )

    effective_optimizer_kwargs = _optimizer_kwargs_for_site(opt_config, config)
    search_optimizer_kwargs = apply_optional_cardinality_range(
        effective_optimizer_kwargs,
        cardinality,
        context=f"Composition site {site_name!r} best_subset",
    )

    if config.get("steps"):
        _validate_grid_strategy(
            config=config,
            optimizer_kwargs=search_optimizer_kwargs,
            optional_count=len(optional),
            optional_k=cardinality.optional_maximum,
        )
    for optional_k in cardinality.optional_cardinalities:
        _validate_all_supports_feasible(
            required=ordered_required,
            optional=optional,
            optional_k=optional_k,
            config=config,
        )

    final_candidate_postprocess = _grid_postprocess(
        opt_config=opt_config,
        repair=repair,
        config=config,
        elements=elements,
        fraction_names=fraction_names,
        feature_names=feature_names,
        exact_k=cardinality.maximum,
    )
    if config.get("steps"):
        _validate_grid_supports(
            projector=final_candidate_postprocess,
            config=config,
            required=ordered_required,
            optional=optional,
            optional_k=cardinality.optional_maximum,
        )

    forbidden_fixed = {by_element[element]: 0.0 for element in forbidden}
    optimizer_fixed = _merge_fixed_features(optimizer_fixed, forbidden_fixed)
    repair_fixed = _merge_fixed_features(repair_fixed, forbidden_fixed)

    sum_constraint = (fraction_names, [1.0] * len(fraction_names), 1.0)
    equality_constraints = [
        *(opt_config.equality_constraints or ()),
        sum_constraint,
    ]
    repair_equalities = [
        *(repair.equality_constraints or ()),
        sum_constraint,
    ]

    optimizer_inequalities = list(opt_config.inequality_constraints or ())
    repair_inequalities = _to_ge_constraints(
        repair.inequality_constraints,
        str(repair.inequality_sense),
    )
    active_elements = [
        element for element in elements if element not in forbidden
    ]
    for element in active_elements:
        feature_name = by_element[element]
        _, upper = _component_bounds(config, element)
        floor_value = _active_floor(config, element, upper)
        repair_inequalities.append(([feature_name], [1.0], floor_value))
        if element in required and feature_name not in optimizer_fixed:
            optimizer_inequalities.append(([feature_name], [1.0], floor_value))

    optional_names = [by_element[element] for element in optional]
    if cardinality.optional_maximum == 0:
        repair = replace(
            repair,
            comp_idx=None,
            k=0,
            support_selection="topk",
            equality_constraints=repair_equalities,
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
            final_sum_constraint=(fraction_names, 1.0),
        )
        optimizer_kwargs = _without_cardinality_kwargs(opt_config.optimizer_kwargs)
    else:
        repair = replace(
            repair,
            comp_idx=optional_names,
            k=cardinality.optional_maximum,
            support_selection="best_subset",
            equality_constraints=repair_equalities,
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
            final_sum_constraint=(fraction_names, 1.0),
        )
        optimizer_kwargs = search_optimizer_kwargs

    return replace(
        opt_config,
        repair_config=repair,
        fixed_features=optimizer_fixed or None,
        equality_constraints=equality_constraints,
        inequality_constraints=optimizer_inequalities,
        optimizer_kwargs=optimizer_kwargs,
        final_candidate_postprocess=final_candidate_postprocess,
    )


__all__ = ["resolve_composition_best_subset"]
