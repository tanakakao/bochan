"""Grouped raw-decision Best Subset support for multiple composition sites."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Any

from torch import Tensor

from bochan.api import (
    CandidateRepairConfig,
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from bochan.api.support.multi_group_best_subset import BEST_SUBSET_GROUPS_KWARG
from bochan.tabular.data import resolve_optimize_config_columns

from .cardinality import (
    BEST_SUBSET_MAX_K_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    require_exact_cardinality_for_steps,
    resolve_composition_cardinality_range,
    support_count,
)
from .logratio_support import (
    RawDecisionAcquisition,
    _raw_fixed_features_list_from_training,
    _reject_one_shot_acquisition,
    is_logratio_best_subset_site,
)
from .logratio_support import (
    _remap_optimize_config as _remap_logratio_optimize_config,
)
from .multi_support import (
    _merge_search_controls,
    _validate_no_cross_site_stepped_constraints,
)
from .raw_bridge import CompositionRawDecisionBridge
from .support import (
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
from .variable_total_support import (
    CompositionVariableTotalDecisionBridge,
    _composition_fixed_amounts,
    _grid_config_with_fixed_amounts,
    is_variable_total_best_subset_site,
)
from .variable_total_support import (
    _active_floor as _variable_active_floor,
)
from .variable_total_support import (
    _component_bounds as _variable_component_bounds,
)
from .variable_total_support import (
    _grid_postprocess as _variable_grid_postprocess,
)
from .variable_total_support import (
    _remap_optimize_config as _remap_variable_optimize_config,
)
from .variable_total_support import (
    _validate_grid_contract as _validate_variable_grid_contract,
)
from .variable_total_support import (
    _validate_grid_strategy as _validate_variable_grid_strategy,
)
from .variable_total_support import (
    _validate_grid_supports as _validate_variable_grid_supports,
)
from .variable_total_support import (
    _validate_support_feasibility as _validate_variable_support_feasibility,
)


def raw_best_subset_site_names(
    sites: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return Best Subset sites that require a raw decision bridge."""

    return tuple(
        str(name)
        for name, config in sites.items()
        if is_logratio_best_subset_site(config)
        or is_variable_total_best_subset_site(config)
    )


def uses_multi_raw_best_subset(
    sites: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether multiple Best Subset sites require one composite raw bridge."""

    selected = [
        str(name)
        for name, config in sites.items()
        if str(config.get("support_selection", "repair")).lower() == "best_subset"
    ]
    return len(selected) > 1 and bool(raw_best_subset_site_names(sites))


@dataclass(frozen=True)
class CompositionMultiRawDecisionBridge:
    """Compose multiple per-site raw bridges without changing the fitted surrogate."""

    stages: tuple[Any, ...]
    site_bridges: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("CompositionMultiRawDecisionBridge requires at least one stage.")

    @property
    def model_feature_names(self) -> tuple[str, ...]:
        return tuple(self.stages[0].model_feature_names)

    @property
    def decision_feature_names(self) -> tuple[str, ...]:
        return tuple(self.stages[-1].decision_feature_names)

    @property
    def model_dim(self) -> int:
        return int(self.stages[0].model_dim)

    @property
    def decision_dim(self) -> int:
        return int(self.stages[-1].decision_dim)

    @property
    def process_index_map(self) -> dict[int, int]:
        """Map original model process indices through every raw bridge stage."""

        mapping = {index: index for index in range(self.model_dim)}
        for stage in self.stages:
            mapping = {
                original: int(stage.process_index_map[current])
                for original, current in mapping.items()
                if current in stage.process_index_map
            }
        return mapping

    def bridge_for_site(self, site_name: str) -> Any:
        for name, bridge in self.site_bridges:
            if name == site_name:
                return bridge
        raise KeyError(f"No raw decision bridge is registered for site {site_name!r}.")

    def decision_to_model(self, values: Tensor) -> Tensor:
        current = values
        for stage in reversed(self.stages):
            current = stage.decision_to_model(current)
        return current

    def model_to_decision(self, values: Tensor) -> Tensor:
        current = values
        for stage in self.stages:
            current = stage.model_to_decision(current)
        return current


@dataclass(frozen=True)
class _VariableTotalSiteView:
    """Final-layout view required by the variable-total grid helpers."""

    decision_feature_names: tuple[str, ...]
    elements: tuple[str, ...]
    amount_names: tuple[str, ...]
    amount_indices: tuple[int, ...]


def _best_subset_site_names(
    sites: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name, config in sites.items()
        if str(config.get("support_selection", "repair")).lower() == "best_subset"
    )


def _build_composite_bridge(
    opt_config: OptimizeConfig,
    *,
    selected_sites: Sequence[str],
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
) -> tuple[CompositionMultiRawDecisionBridge, OptimizeConfig, Tensor]:
    current_config = opt_config
    current_names = tuple(str(name) for name in model_feature_names)
    current_bounds = model_bounds
    stages: list[Any] = []
    site_bridges: list[tuple[str, Any]] = []

    for site_name in selected_sites:
        site_config = composition_sites[site_name]
        transformer = composition_transformers.get(site_name)
        if transformer is None:
            raise RuntimeError(
                f"Composition transformer for site {site_name!r} is not fitted. Call fit() first."
            )

        if is_variable_total_best_subset_site(site_config):
            bridge = CompositionVariableTotalDecisionBridge.from_transformer(
                transformer,
                current_names,
                total_feature=str(site_config["total_feature"]),
            )
            next_bounds = bridge.decision_bounds(
                current_bounds,
                component_bounds=site_config.get("bounds") or {},
                total_bounds=site_config["total_bounds"],
            )
            current_config = _remap_variable_optimize_config(
                current_config,
                bridge,
                next_bounds,
            )
        elif is_logratio_best_subset_site(site_config):
            bridge = CompositionRawDecisionBridge.from_transformer(
                transformer,
                current_names,
            )
            next_bounds = bridge.decision_bounds(
                current_bounds,
                component_bounds=site_config.get("bounds") or {},
                total=float(site_config.get("total", 1.0)),
            )
            current_config = _remap_logratio_optimize_config(
                current_config,
                bridge,
                next_bounds,
            )
        else:
            continue

        stages.append(bridge)
        site_bridges.append((site_name, bridge))
        current_names = tuple(bridge.decision_feature_names)
        current_bounds = next_bounds

    if not stages:
        raise ValueError(
            "Multi raw composition Best Subset requires at least one CLR/ALR/ILR "
            "or variable-total site."
        )
    return (
        CompositionMultiRawDecisionBridge(tuple(stages), tuple(site_bridges)),
        current_config,
        current_bounds,
    )


def _variable_site_view(
    *,
    site_name: str,
    transformer: Any,
    feature_names: Sequence[Any],
) -> _VariableTotalSiteView:
    elements = tuple(str(value) for value in transformer.fitted_elements)
    amount_names = tuple(
        f"{transformer.prefix}__amount__{element}" for element in elements
    )
    positions = {str(name): index for index, name in enumerate(feature_names)}
    missing = [name for name in amount_names if name not in positions]
    if missing:
        raise KeyError(
            f"Variable-total composition amount features for site {site_name!r} "
            f"are missing from the raw optimization dataset: {missing!r}."
        )
    return _VariableTotalSiteView(
        decision_feature_names=tuple(str(name) for name in feature_names),
        elements=elements,
        amount_names=amount_names,
        amount_indices=tuple(positions[name] for name in amount_names),
    )


def _fraction_site_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if str(config.get("representation", "fractions")).lower() in {"clr", "alr", "ilr"}:
        raw = dict(config)
        raw["representation"] = "fractions"
        return raw
    return config


def _resolve_grouped_raw_named_config(
    raw_config: OptimizeConfig,
    *,
    selected_sites: Sequence[str],
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    feature_names: Sequence[Any],
) -> OptimizeConfig:
    configs = [composition_sites[name] for name in selected_sites]
    repair = raw_config.repair_config or CandidateRepairConfig()
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
            "Multiple composition best_subset groups own their per-site total constraints. "
            "Remove CandidateRepairConfig.final_sum_constraint."
        )

    raw_names = tuple(str(name) for name in feature_names)
    positions = {name: index for index, name in enumerate(raw_names)}
    search_kwargs = _merge_search_controls(raw_config, configs)
    optimizer_fixed = _as_mapping(raw_config.fixed_features)
    repair_fixed = _as_mapping(repair.fixed_features)
    optimizer_equalities = list(raw_config.equality_constraints or ())
    optimizer_inequalities = list(raw_config.inequality_constraints or ())
    repair_equalities = list(repair.equality_constraints or ())
    repair_inequalities = _to_ge_constraints(
        repair.inequality_constraints,
        str(repair.inequality_sense),
    )
    final_postprocess = getattr(raw_config, "final_candidate_postprocess", None)

    metadata: list[dict[str, Any]] = []
    stepped_names: dict[str, set[str]] = {}
    owners: dict[str, str] = {}
    for site_name in selected_sites:
        site_config = composition_sites[site_name]
        transformer = composition_transformers.get(site_name)
        if transformer is None:
            raise RuntimeError(
                f"Composition transformer for site {site_name!r} is not fitted. Call fit() first."
            )
        elements = tuple(str(value) for value in transformer.fitted_elements)
        if site_config.get("variable_total"):
            view = _variable_site_view(
                site_name=site_name,
                transformer=transformer,
                feature_names=raw_names,
            )
            decision_names = view.amount_names
            kind = "variable_total"
            effective_config = site_config
        else:
            effective_config = _fraction_site_config(site_config)
            _validate_site(site_name, effective_config)
            decision_names = tuple(
                f"{transformer.prefix}__fraction__{element}" for element in elements
            )
            missing = [name for name in decision_names if name not in positions]
            if missing:
                raise KeyError(
                    f"Composition fraction features for site {site_name!r} are missing "
                    f"from the raw optimization dataset: {missing!r}."
                )
            view = None
            kind = "fraction"

        overlapping = sorted(name for name in decision_names if name in owners)
        if overlapping:
            previous = sorted({owners[name] for name in overlapping})
            raise ValueError(
                "Multiple composition Best Subset sites must use disjoint raw feature "
                f"blocks. Site {site_name!r} overlaps sites {previous!r}: "
                f"{overlapping!r}."
            )
        owners.update({name: site_name for name in decision_names})
        if effective_config.get("steps"):
            stepped_names[site_name] = set(decision_names)
        metadata.append(
            {
                "site_name": site_name,
                "config": effective_config,
                "elements": elements,
                "decision_names": decision_names,
                "kind": kind,
                "view": view,
            }
        )

    _validate_no_cross_site_stepped_constraints(
        opt_config=raw_config,
        repair=repair,
        stepped_fraction_names=stepped_names,
        feature_names=raw_names,
    )

    group_specs: list[dict[str, Any]] = []
    optional_names_flat: list[str] = []
    support_counts: list[int] = []
    cardinality_seed_counts: list[int] = []

    for item in metadata:
        site_name = item["site_name"]
        config = item["config"]
        elements = item["elements"]
        decision_names = item["decision_names"]
        kind = item["kind"]

        all_fixed = _merge_fixed_features(optimizer_fixed, repair_fixed)
        site_opt = replace(
            raw_config,
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

        explicit_required = set(config.get("required_components") or ())
        explicit_forbidden = set(config.get("forbidden_components") or ())
        required = set(explicit_required)
        forbidden = set(explicit_forbidden)
        by_element = dict(zip(elements, decision_names, strict=True))

        if kind == "fraction":
            composition_fixed = _composition_fixed_values(
                all_fixed,
                composition_feature_names=decision_names,
                feature_names=raw_names,
            )
            grid_config = _grid_config_with_fixed_fractions(
                config,
                elements=elements,
                fraction_names=decision_names,
                fixed_values=composition_fixed,
            )
            _validate_grid_contract(
                opt_config=site_opt,
                repair=site_repair,
                config=grid_config,
                fraction_names=decision_names,
                feature_names=raw_names,
                all_fixed=all_fixed,
            )
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
        else:
            view = item["view"]
            composition_fixed = _composition_fixed_amounts(all_fixed, bridge=view)
            grid_config = _grid_config_with_fixed_amounts(
                config,
                bridge=view,
                fixed_values=composition_fixed,
            )
            _validate_variable_grid_contract(
                raw_config=site_opt,
                repair=site_repair,
                site_config=grid_config,
                bridge=view,
                all_fixed=all_fixed,
            )
            for element, amount_name in by_element.items():
                lower, upper = _variable_component_bounds(config, element)
                if lower > _TOLERANCE:
                    required.add(element)
                if upper <= _TOLERANCE:
                    forbidden.add(element)
                if amount_name in composition_fixed:
                    if abs(composition_fixed[amount_name]) <= _TOLERANCE:
                        forbidden.add(element)
                    else:
                        required.add(element)

        overlap = required & forbidden
        if overlap:
            raise ValueError(
                f"Composition site {site_name!r} has components both required and "
                f"forbidden: {sorted(overlap)!r}."
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

        if kind == "fraction":
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
                fraction_names=decision_names,
                feature_names=raw_names,
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
        else:
            if grid_config.get("steps"):
                _validate_variable_grid_strategy(
                    config=grid_config,
                    optimizer_kwargs=search_kwargs,
                    optional_count=len(optional),
                    optional_k=cardinality.optional_maximum,
                )
            for optional_k in cardinality.optional_cardinalities:
                _validate_variable_support_feasibility(
                    config=grid_config if grid_config.get("steps") else config,
                    required=ordered_required,
                    optional=optional,
                    optional_k=optional_k,
                )
            projector = _variable_grid_postprocess(
                raw_config=site_opt,
                repair=site_repair,
                site_config=grid_config,
                bridge=item["view"],
                exact_k=cardinality.maximum,
            )
            if grid_config.get("steps"):
                _validate_variable_grid_supports(
                    projector=projector,
                    site_config=grid_config,
                    optimizer_kwargs=search_kwargs,
                    required=ordered_required,
                    optional=optional,
                    optional_k=cardinality.optional_maximum,
                )
        final_postprocess = projector

        forbidden_fixed = {by_element[element]: 0.0 for element in forbidden}
        optimizer_fixed = _merge_fixed_features(optimizer_fixed, forbidden_fixed)
        repair_fixed = _merge_fixed_features(repair_fixed, forbidden_fixed)

        if kind == "fraction":
            sum_constraint = (
                decision_names,
                [1.0] * len(decision_names),
                1.0,
            )
            optimizer_equalities.append(sum_constraint)
            repair_equalities.append(sum_constraint)
            for element in elements:
                if element in forbidden:
                    continue
                feature_name = by_element[element]
                _, upper = _component_bounds(config, element)
                floor = _active_floor(config, element, upper)
                repair_inequalities.append(([feature_name], [1.0], floor))
                if element in required and feature_name not in optimizer_fixed:
                    optimizer_inequalities.append(([feature_name], [1.0], floor))
        else:
            total_lower, total_upper = map(float, config["total_bounds"])
            total_constraints = [
                (decision_names, [1.0] * len(decision_names), total_lower),
                (decision_names, [-1.0] * len(decision_names), -total_upper),
            ]
            optimizer_inequalities.extend(total_constraints)
            repair_inequalities.extend(total_constraints)
            for element in elements:
                if element in forbidden:
                    continue
                amount_name = by_element[element]
                floor = _variable_active_floor(config, element)
                repair_inequalities.append(([amount_name], [1.0], floor))
                if element in required and amount_name not in optimizer_fixed:
                    optimizer_inequalities.append(([amount_name], [1.0], floor))

        optional_names = [by_element[element] for element in optional]
        if cardinality.optional_maximum > 0:
            group_specs.append(
                {
                    "name": site_name,
                    "comp_idx": [positions[name] for name in optional_names],
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
        maximum = int(search_kwargs.get("best_subset_max_combinations", 2000))
        resolved_strategy = (
            "exact"
            if strategy == "auto" and combination_count <= maximum
            else "beam"
            if strategy == "auto"
            else strategy
        )
        if resolved_strategy == "exact" and combination_count > maximum:
            raise ValueError(
                "Multiple composition best_subset exact enumeration would evaluate "
                f"{combination_count} support combinations, exceeding "
                f"best_subset_max_combinations={maximum}."
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
        optimizer_kwargs = _without_cardinality_kwargs(raw_config.optimizer_kwargs)
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
        raw_config,
        repair_config=final_repair,
        fixed_features=optimizer_fixed or None,
        equality_constraints=optimizer_equalities,
        inequality_constraints=optimizer_inequalities,
        optimizer_kwargs=optimizer_kwargs,
        final_candidate_postprocess=final_postprocess,
    )


def prepare_multi_raw_best_subset_config(
    opt_config: OptimizeConfig,
    *,
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
) -> tuple[CompositionMultiRawDecisionBridge, OptimizeConfig, Tensor]:
    """Build one grouped raw decision problem for multiple composition sites."""

    selected_sites = _best_subset_site_names(composition_sites)
    if len(selected_sites) < 2:
        raise ValueError(
            "Multi raw composition Best Subset requires at least two best_subset sites."
        )
    if not raw_best_subset_site_names(composition_sites):
        raise ValueError(
            "Multi raw composition Best Subset requires a CLR/ALR/ILR or "
            "variable-total site."
        )

    bridge, raw_named_config, raw_bounds = _build_composite_bridge(
        opt_config,
        selected_sites=selected_sites,
        composition_sites=composition_sites,
        composition_transformers=composition_transformers,
        model_feature_names=model_feature_names,
        model_bounds=model_bounds,
    )
    raw_named_config = _resolve_grouped_raw_named_config(
        raw_named_config,
        selected_sites=selected_sites,
        composition_sites=composition_sites,
        composition_transformers=composition_transformers,
        feature_names=bridge.decision_feature_names,
    )
    raw_config = resolve_optimize_config_columns(
        raw_named_config,
        bridge.decision_feature_names,
        dtype=dtype,
        device=device,
    )

    raw_cat_dims = [
        bridge.process_index_map[int(index)]
        for index in (model_cat_dims or ())
        if int(index) in bridge.process_index_map
    ]
    raw_config = resolve_optimizer_from_cat_dims(
        opt_config=raw_config,
        cat_dims=raw_cat_dims,
    )
    if (
        uses_mixed_fixed_features(raw_config.optimizer)
        and raw_config.fixed_features_list is None
    ):
        raw_train_x = bridge.model_to_decision(train_x) if train_x is not None else None
        inferred = _raw_fixed_features_list_from_training(
            raw_train_x,
            raw_cat_dims,
        )
        if inferred:
            raw_config = replace(raw_config, fixed_features_list=inferred)

    return bridge, raw_config, raw_bounds


@dataclass(frozen=True)
class MultiRawBestSubsetResult:
    """Model-space and composite raw-space result from grouped support search."""

    candidates: Tensor
    raw_candidates: Tensor
    acq_value: Any
    raw_opt_config: OptimizeConfig
    bridge: CompositionMultiRawDecisionBridge


def optimize_multi_raw_best_subset(
    base_acqf: Any,
    opt_config: OptimizeConfig,
    *,
    composition_sites: Mapping[str, Mapping[str, Any]],
    composition_transformers: Mapping[str, Any],
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
    optimize_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> MultiRawBestSubsetResult:
    """Optimize multiple composition support groups in one composite raw space."""

    _reject_one_shot_acquisition(base_acqf)
    bridge, raw_config, raw_bounds = prepare_multi_raw_best_subset_config(
        opt_config,
        composition_sites=composition_sites,
        composition_transformers=composition_transformers,
        model_feature_names=model_feature_names,
        model_bounds=model_bounds,
        dtype=dtype,
        device=device,
        model_cat_dims=model_cat_dims,
        train_x=train_x,
    )
    wrapped = RawDecisionAcquisition(base_acqf, bridge)  # type: ignore[arg-type]
    if optimize_fn is None:
        from bochan.api.optimizer.service import optimize_candidates as optimize_fn

    raw_candidates, _raw_value = optimize_fn(
        acqf=wrapped,
        bounds=raw_bounds,
        config=raw_config,
    )
    model_candidates = bridge.decision_to_model(raw_candidates)
    final_value = base_acqf(model_candidates)
    if hasattr(final_value, "detach"):
        final_value = final_value.detach()
    return MultiRawBestSubsetResult(
        candidates=model_candidates,
        raw_candidates=raw_candidates,
        acq_value=final_value,
        raw_opt_config=raw_config,
        bridge=bridge,
    )


__all__ = [
    "CompositionMultiRawDecisionBridge",
    "MultiRawBestSubsetResult",
    "optimize_multi_raw_best_subset",
    "prepare_multi_raw_best_subset_config",
    "raw_best_subset_site_names",
    "uses_multi_raw_best_subset",
]
