"""Acquisition-aware Best Subset search for variable-total compositions.

Variable-total compositions are optimized in raw absolute component amounts.
This keeps element support, component bounds, and the site total in one decision
space. The fitted surrogate still sees its original composition representation
(fractions / CLR / ALR / ILR) plus the fitted total feature.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from math import comb
from typing import Any

import torch
from torch import Tensor

from bochan.api import (
    CandidateRepairConfig,
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from bochan.tabular.data import resolve_optimize_config_columns

from .cardinality import (
    BEST_SUBSET_MAX_K_KWARG,
    BEST_SUBSET_MIN_K_KWARG,
    apply_optional_cardinality_range,
    require_exact_cardinality_for_steps,
    resolve_composition_cardinality_range,
)
from .grid import (
    CompositionVariableTotalGridFinalPostprocess,
    GridLinearConstraint,
    composition_grid_linear_constraints,
    merge_composition_grid_linear_constraints,
)
from .logratio_support import RawDecisionAcquisition
from .raw_bridge import CompositionRawDecisionBridge

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


def is_variable_total_best_subset_site(config: Mapping[str, Any]) -> bool:
    """Return whether one composition site requests variable-total Best Subset."""

    return bool(
        config.get("variable_total")
        and str(config.get("support_selection", "repair")).lower() == "best_subset"
    )


def resolve_variable_total_best_subset_site(
    sites: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    """Return the single variable-total Best Subset site, if configured."""

    selected = [
        (str(name), config)
        for name, config in sites.items()
        if is_variable_total_best_subset_site(config)
    ]
    if not selected:
        return None
    all_best_subset = [
        str(name)
        for name, config in sites.items()
        if str(config.get("support_selection", "repair")).lower() == "best_subset"
    ]
    if len(all_best_subset) > 1:
        raise ValueError(
            "Composition best_subset currently supports one composition site per "
            "candidate optimization."
        )
    return selected[0]


def _validate_tensor(values: Tensor, expected_dim: int, *, name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not values.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if values.ndim < 1 or int(values.shape[-1]) != int(expected_dim):
        raise ValueError(
            f"{name} must have final dimension {expected_dim}, got {tuple(values.shape)}."
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")


@dataclass(frozen=True)
class CompositionVariableTotalDecisionBridge:
    """Replace model composition coordinates + total with absolute element amounts."""

    base: CompositionRawDecisionBridge
    total_feature_name: str
    total_model_index: int
    total_raw_index: int
    amount_names: tuple[str, ...]

    @classmethod
    def from_transformer(
        cls,
        transformer: Any,
        model_feature_names: Sequence[Any],
        *,
        total_feature: str,
    ) -> CompositionVariableTotalDecisionBridge:
        """Build a variable-total bridge from one fitted composition transformer."""

        base = CompositionRawDecisionBridge.from_transformer(
            transformer,
            model_feature_names,
        )
        total_name = str(total_feature)
        try:
            total_model_index = base.model_feature_names.index(total_name)
        except ValueError as exc:
            raise KeyError(
                f"Variable composition total feature {total_name!r} is missing from feature_names."
            ) from exc
        if base.coordinate_start <= total_model_index < base.coordinate_stop:
            raise ValueError("Composition total feature overlaps the model coordinate block.")
        try:
            total_raw_index = base.process_index_map[total_model_index]
        except KeyError as exc:
            raise ValueError(
                "Composition total feature could not be mapped into raw decision space."
            ) from exc

        amount_names = tuple(
            f"{transformer.prefix}__amount__{element}" for element in base.elements
        )
        return cls(
            base=base,
            total_feature_name=total_name,
            total_model_index=int(total_model_index),
            total_raw_index=int(total_raw_index),
            amount_names=amount_names,
        )

    @property
    def model_feature_names(self) -> tuple[str, ...]:
        return self.base.model_feature_names

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return self.base.coordinate_names

    @property
    def elements(self) -> tuple[str, ...]:
        return self.base.elements

    @property
    def model_dim(self) -> int:
        return self.base.model_dim

    @property
    def decision_dim(self) -> int:
        return self.base.decision_dim - 1

    @property
    def _base_to_decision(self) -> dict[int, int]:
        return {
            base_index: (
                base_index
                if base_index < self.total_raw_index
                else base_index - 1
            )
            for base_index in range(self.base.decision_dim)
            if base_index != self.total_raw_index
        }

    @property
    def amount_indices(self) -> tuple[int, ...]:
        mapping = self._base_to_decision
        return tuple(mapping[index] for index in self.base.fraction_indices)

    @property
    def amount_slice(self) -> slice:
        indices = self.amount_indices
        if indices != tuple(range(indices[0], indices[0] + len(indices))):
            raise RuntimeError("Variable-total amount features are not contiguous.")
        return slice(indices[0], indices[-1] + 1)

    @property
    def process_index_map(self) -> dict[int, int]:
        """Map ordinary model-space process indices to amount decision indices."""

        mapping = self._base_to_decision
        result: dict[int, int] = {}
        for model_index, base_index in self.base.process_index_map.items():
            if model_index == self.total_model_index:
                continue
            result[int(model_index)] = mapping[int(base_index)]
        return result

    @property
    def decision_feature_names(self) -> tuple[str, ...]:
        names = list(self.base.decision_feature_names)
        for base_index, amount_name in zip(
            self.base.fraction_indices,
            self.amount_names,
            strict=True,
        ):
            names[base_index] = amount_name
        del names[self.total_raw_index]
        return tuple(names)

    def _base_raw_from_amounts(self, values: Tensor) -> Tensor:
        _validate_tensor(values, self.decision_dim, name="decision values")
        amounts = values[..., list(self.amount_indices)]
        if torch.any(amounts < 0):
            raise ValueError("Raw composition amounts must be non-negative.")
        totals = amounts.sum(dim=-1, keepdim=True)
        if torch.any(totals <= 0):
            raise ValueError("Each variable-total composition must have positive total.")
        fractions = amounts / totals

        fraction_lookup = {
            base_index: local_index
            for local_index, base_index in enumerate(self.base.fraction_indices)
        }
        base_to_decision = self._base_to_decision
        columns: list[Tensor] = []
        for base_index in range(self.base.decision_dim):
            if base_index == self.total_raw_index:
                columns.append(totals.squeeze(-1))
            elif base_index in fraction_lookup:
                columns.append(fractions[..., fraction_lookup[base_index]])
            else:
                columns.append(values[..., base_to_decision[base_index]])
        return torch.stack(columns, dim=-1)

    def decision_to_model(self, values: Tensor) -> Tensor:
        """Map absolute element amounts to fitted composition coordinates + total."""

        return self.base.decision_to_model(self._base_raw_from_amounts(values))

    def model_to_decision(self, values: Tensor) -> Tensor:
        """Map fitted model values to absolute amount decisions."""

        _validate_tensor(values, self.model_dim, name="model values")
        raw = self.base.model_to_decision(values)
        total = raw[..., self.total_raw_index]
        fractions = raw[..., list(self.base.fraction_indices)]
        amounts = fractions * total.unsqueeze(-1)

        fraction_lookup = {
            base_index: local_index
            for local_index, base_index in enumerate(self.base.fraction_indices)
        }
        columns: list[Tensor] = []
        for base_index in range(self.base.decision_dim):
            if base_index == self.total_raw_index:
                continue
            if base_index in fraction_lookup:
                columns.append(amounts[..., fraction_lookup[base_index]])
            else:
                columns.append(raw[..., base_index])
        return torch.stack(columns, dim=-1)

    def amount_values(self, values: Tensor) -> Tensor:
        """Return the absolute component-amount block."""

        _validate_tensor(values, self.decision_dim, name="decision values")
        return values[..., list(self.amount_indices)]

    def decision_bounds(
        self,
        model_bounds: Tensor,
        *,
        component_bounds: Mapping[str, Sequence[float]] | None,
        total_bounds: Sequence[float],
    ) -> Tensor:
        """Build raw-amount bounds while preserving ordinary process bounds."""

        if not isinstance(model_bounds, Tensor):
            raise TypeError("model_bounds must be a torch.Tensor.")
        if model_bounds.ndim != 2 or tuple(model_bounds.shape) != (2, self.model_dim):
            raise ValueError(
                f"model_bounds must have shape (2, {self.model_dim}), got {tuple(model_bounds.shape)}."
            )
        pair = tuple(total_bounds)
        if len(pair) != 2:
            raise ValueError("total_bounds must contain two values.")
        total_lower, total_upper = map(float, pair)
        if total_lower <= 0 or total_lower >= total_upper:
            raise ValueError("total_bounds must be positive and increasing.")

        configured = dict(component_bounds or {})
        process_by_decision = {
            raw_index: model_index
            for model_index, raw_index in self.process_index_map.items()
        }
        amount_position = {
            index: local for local, index in enumerate(self.amount_indices)
        }
        lower: list[Tensor] = []
        upper: list[Tensor] = []
        for decision_index in range(self.decision_dim):
            if decision_index in amount_position:
                element = self.elements[amount_position[decision_index]]
                bounds = tuple(configured.get(element, (0.0, total_upper)))
                if len(bounds) != 2:
                    raise ValueError(f"Bounds for {element!r} must contain two values.")
                low, high = map(float, bounds)
                high = min(high, total_upper)
                if low < 0 or high < low:
                    raise ValueError(
                        f"Invalid variable-total composition bounds for {element!r}: {(low, high)!r}."
                    )
                lower.append(model_bounds.new_tensor(low))
                upper.append(model_bounds.new_tensor(high))
            else:
                model_index = process_by_decision[decision_index]
                lower.append(model_bounds[0, model_index])
                upper.append(model_bounds[1, model_index])
        return torch.stack((torch.stack(lower), torch.stack(upper)))

    def expand_model_index(self, index: Any) -> tuple[Any, ...]:
        """Expand one model-space term into amount-space term(s)."""

        if isinstance(index, str):
            if index in self.coordinate_names:
                raise ValueError(
                    "Direct constraints on composition model coordinates cannot be combined "
                    "with variable-total best_subset. Constrain raw elements instead."
                )
            if index == self.total_feature_name:
                return self.amount_names
            if index in self.amount_names:
                return (index,)
            return (index,)

        resolved = int(index)
        if self.base.coordinate_start <= resolved < self.base.coordinate_stop:
            raise ValueError(
                "Direct constraints on composition model-coordinate indices cannot be "
                "combined with variable-total best_subset."
            )
        if resolved == self.total_model_index:
            return self.amount_indices
        try:
            return (self.process_index_map[resolved],)
        except KeyError as exc:
            raise ValueError(
                f"Cannot map model feature index {resolved} to raw decision space."
            ) from exc


def _as_items(values: Any) -> list[Any]:
    if isinstance(values, (str, int)):
        return [values]
    if torch.is_tensor(values):
        return values.detach().cpu().reshape(-1).tolist()
    return list(values)


def _as_coefficients(values: Any) -> list[float]:
    if torch.is_tensor(values):
        return [float(value) for value in values.detach().cpu().reshape(-1).tolist()]
    if isinstance(values, (int, float)):
        return [float(values)]
    return [float(value) for value in values]


def _map_constraints(
    constraints: Sequence[tuple[Any, Any, Any]] | None,
    bridge: CompositionVariableTotalDecisionBridge,
) -> list[tuple[list[Any], list[float], Any]] | None:
    if constraints is None:
        return None
    mapped: list[tuple[list[Any], list[float], Any]] = []
    for indices, coefficients, rhs in constraints:
        items = _as_items(indices)
        coeffs = _as_coefficients(coefficients)
        if len(items) != len(coeffs):
            raise ValueError("Constraint indices and coefficients must have matching lengths.")
        mapped_indices: list[Any] = []
        mapped_coefficients: list[float] = []
        for index, coefficient in zip(items, coeffs, strict=True):
            expanded = bridge.expand_model_index(index)
            mapped_indices.extend(expanded)
            mapped_coefficients.extend([float(coefficient)] * len(expanded))
        mapped.append((mapped_indices, mapped_coefficients, rhs))
    return mapped


def _map_fixed_features(
    values: Mapping[Any, Any] | None,
    bridge: CompositionVariableTotalDecisionBridge,
) -> dict[Any, float] | None:
    if not values:
        return None
    mapped: dict[Any, float] = {}
    for key, value in values.items():
        expanded = bridge.expand_model_index(key)
        if len(expanded) != 1:
            raise ValueError(
                "A fixed value on the variable composition total cannot be represented "
                "as one raw decision feature. Use a composition total equality constraint."
            )
        mapped[expanded[0]] = float(value)
    return mapped


def _map_fixed_features_list(
    values: Sequence[Mapping[Any, Any]] | None,
    bridge: CompositionVariableTotalDecisionBridge,
) -> list[dict[Any, float]] | None:
    if values is None:
        return None
    return [_map_fixed_features(item, bridge) or {} for item in values]


def _map_numeric_indices(
    values: Sequence[int] | None,
    bridge: CompositionVariableTotalDecisionBridge,
) -> list[int] | None:
    if values is None:
        return None
    mapped: list[int] = []
    for raw in values:
        index = int(raw)
        if bridge.base.coordinate_start <= index < bridge.base.coordinate_stop:
            continue
        if index == bridge.total_model_index:
            continue
        mapped.append(bridge.process_index_map[index])
    return mapped


def _raw_duplicate_tolerances(
    config: OptimizeConfig,
    bridge: CompositionVariableTotalDecisionBridge,
) -> tuple[float, ...] | None:
    tolerances = getattr(config, "duplicate_tolerances", None)
    if tolerances is None:
        return None
    values = tuple(float(value) for value in tolerances)
    if len(values) != bridge.model_dim:
        raise ValueError(
            "duplicate_tolerances width must match the fitted model feature dimension."
        )
    amount_tolerance = max(
        float(getattr(config, "duplicate_tolerance", 1e-10)),
        1e-12,
    )
    process_inverse = {
        raw_index: model_index
        for model_index, raw_index in bridge.process_index_map.items()
    }
    amount_set = set(bridge.amount_indices)
    result: list[float] = []
    for decision_index in range(bridge.decision_dim):
        if decision_index in amount_set:
            result.append(amount_tolerance)
        else:
            result.append(values[process_inverse[decision_index]])
    return tuple(result)


def _raw_final_candidate_postprocess(
    values: Tensor,
    *,
    callback: Callable[[Tensor], Tensor],
    bridge: CompositionVariableTotalDecisionBridge,
) -> Tensor:
    model_values = bridge.decision_to_model(values)
    processed = callback(model_values)
    if not isinstance(processed, Tensor):
        raise TypeError("final_candidate_postprocess must return a torch.Tensor.")
    if tuple(processed.shape) != tuple(model_values.shape):
        raise ValueError("final_candidate_postprocess must preserve candidate tensor shape.")
    result = values.clone()
    for model_index, decision_index in bridge.process_index_map.items():
        result[..., decision_index] = processed[..., model_index]
    return result


def _remap_optimize_config(
    config: OptimizeConfig,
    bridge: CompositionVariableTotalDecisionBridge,
    raw_bounds: Tensor,
) -> OptimizeConfig:
    if config.post_processing_func is not None:
        raise ValueError(
            "Custom model-space post_processing_func is not supported with variable-total "
            "composition best_subset."
        )

    repair = config.repair_config
    if repair is not None:
        repair = replace(
            repair,
            bounds=raw_bounds,
            numeric_indices=_map_numeric_indices(repair.numeric_indices, bridge),
            comp_idx=None if repair.comp_idx is None else list(repair.comp_idx),
            equality_constraints=_map_constraints(repair.equality_constraints, bridge),
            inequality_constraints=_map_constraints(repair.inequality_constraints, bridge),
            fixed_features=_map_fixed_features(repair.fixed_features, bridge),
        )

    final_candidate_postprocess = getattr(config, "final_candidate_postprocess", None)
    if final_candidate_postprocess is not None:
        callback = final_candidate_postprocess

        def process_raw_candidates(values: Tensor) -> Tensor:
            return _raw_final_candidate_postprocess(
                values,
                callback=callback,
                bridge=bridge,
            )

        final_candidate_postprocess = process_raw_candidates

    replacements: dict[str, Any] = {
        "repair_config": repair,
        "fixed_features": _map_fixed_features(config.fixed_features, bridge),
        "fixed_features_list": _map_fixed_features_list(
            config.fixed_features_list,
            bridge,
        ),
        "equality_constraints": _map_constraints(config.equality_constraints, bridge),
        "inequality_constraints": _map_constraints(config.inequality_constraints, bridge),
        "final_candidate_postprocess": final_candidate_postprocess,
    }
    if hasattr(config, "duplicate_tolerances"):
        replacements["duplicate_tolerances"] = _raw_duplicate_tolerances(config, bridge)
    return replace(config, **replacements)


def _component_bounds(
    config: Mapping[str, Any],
    element: str,
) -> tuple[float, float]:
    total_upper = float(config["total_bounds"][1])
    pair = tuple((config.get("bounds") or {}).get(element, (0.0, total_upper)))
    if len(pair) != 2:
        raise ValueError(f"Bounds for {element!r} must contain two values.")
    lower, upper = map(float, pair)
    return max(0.0, lower), min(total_upper, upper)


def _active_floor(config: Mapping[str, Any], element: str) -> float:
    lower, upper = _component_bounds(config, element)
    if lower > _TOLERANCE:
        return lower
    step = (config.get("steps") or {}).get(element)
    if step is not None:
        return min(float(step), upper)
    total_upper = float(config["total_bounds"][1])
    return min(
        max(total_upper * 10.0 * _TOLERANCE, 10.0 * _TOLERANCE),
        upper,
    )


def _optimizer_kwargs_for_site(
    opt_config: OptimizeConfig,
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    """Keep variable-total step grids on exhaustive support search for now."""

    if not config.get("steps") or optional_k == 0:
        return
    strategy = str(optimizer_kwargs.get("best_subset_strategy", "exact")).lower()
    support_count = comb(optional_count, optional_k)
    maximum = int(optimizer_kwargs.get("best_subset_max_combinations", 2000))
    if strategy == "auto":
        strategy = "exact" if support_count <= maximum else "beam"
    if strategy != "exact":
        raise ValueError(
            "Variable-total composition best_subset with component steps currently "
            "requires exact support search. Use best_subset_strategy='exact', or "
            "'auto' only when the support count is within best_subset_max_combinations."
        )
    if support_count > maximum:
        raise ValueError(
            "Variable-total composition step-grid best_subset exact enumeration would "
            f"evaluate {support_count} supports, exceeding "
            f"best_subset_max_combinations={maximum}."
        )


def _grid_linear_constraints(
    *,
    raw_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    bridge: CompositionVariableTotalDecisionBridge,
) -> tuple[GridLinearConstraint, ...]:
    """Collect raw-amount-only constraints for the variable-total grid MILP."""

    optimizer_constraints = composition_grid_linear_constraints(
        equality_constraints=raw_config.equality_constraints,
        inequality_constraints=raw_config.inequality_constraints,
        feature_names=bridge.decision_feature_names,
        composition_feature_names=bridge.amount_names,
        inequality_sense="ge",
        rhs_scale=1.0,
        context="Variable-total composition step-grid best_subset",
    )
    repair_constraints = composition_grid_linear_constraints(
        equality_constraints=repair.equality_constraints,
        inequality_constraints=repair.inequality_constraints,
        feature_names=bridge.decision_feature_names,
        composition_feature_names=bridge.amount_names,
        inequality_sense=str(repair.inequality_sense),
        rhs_scale=1.0,
        context="Variable-total composition step-grid best_subset",
    )
    return merge_composition_grid_linear_constraints(
        optimizer_constraints,
        repair_constraints,
    )


def _fixed_amount_values(
    values: Mapping[Any, Any],
    *,
    bridge: CompositionVariableTotalDecisionBridge,
) -> list[Any]:
    names = set(bridge.amount_names)
    positions = set(bridge.amount_indices)
    return [
        key
        for key, value in values.items()
        if (
            key in names
            or (isinstance(key, int) and int(key) in positions)
        )
        and abs(float(value)) > _TOLERANCE
    ]


def _validate_grid_contract(
    *,
    raw_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    site_config: Mapping[str, Any],
    bridge: CompositionVariableTotalDecisionBridge,
    all_fixed: Mapping[Any, float],
) -> None:
    if not site_config.get("steps"):
        return

    optimizer_name = str(raw_config.optimizer).replace("-", "_").lower()
    if optimizer_name in _NATIVE_FINAL_POSTPROCESS_BYPASS:
        raise ValueError(
            "Variable-total composition step-grid best_subset requires an optimizer "
            "backend that applies final_candidate_postprocess. Use optimize_acqf, evo, "
            "or torch."
        )
    if hasattr(raw_config, "ensure_unique_candidates") and not bool(
        raw_config.ensure_unique_candidates
    ):
        raise ValueError(
            "Variable-total composition step-grid best_subset requires "
            "ensure_unique_candidates=True so grid-projected candidates are re-evaluated."
        )

    nonzero_fixed = _fixed_amount_values(all_fixed, bridge=bridge)
    if nonzero_fixed:
        raise ValueError(
            "Variable-total composition step-grid best_subset does not yet support "
            f"non-zero fixed composition amounts: {nonzero_fixed!r}. Use component "
            "bounds/required elements instead."
        )

    _grid_linear_constraints(
        raw_config=raw_config,
        repair=repair,
        bridge=bridge,
    )


def _grid_postprocess(
    *,
    raw_config: OptimizeConfig,
    repair: CandidateRepairConfig,
    site_config: Mapping[str, Any],
    bridge: CompositionVariableTotalDecisionBridge,
    exact_k: int,
) -> CompositionVariableTotalGridFinalPostprocess | Any:
    if not site_config.get("steps"):
        return getattr(raw_config, "final_candidate_postprocess", None)
    return CompositionVariableTotalGridFinalPostprocess.from_config(
        feature_indices=bridge.amount_indices,
        elements=bridge.elements,
        config=site_config,
        exact_k=exact_k,
        linear_constraints=_grid_linear_constraints(
            raw_config=raw_config,
            repair=repair,
            bridge=bridge,
        ),
        previous=getattr(raw_config, "final_candidate_postprocess", None),
    )


def _validate_grid_supports(
    *,
    projector: CompositionVariableTotalGridFinalPostprocess | Any,
    site_config: Mapping[str, Any],
    required: Sequence[str],
    optional: Sequence[str],
    optional_k: int,
) -> None:
    if not site_config.get("steps"):
        return
    for selected in combinations(optional, optional_k):
        projector.validate_support([*required, *selected])


def _validate_support_feasibility(
    *,
    config: Mapping[str, Any],
    required: Sequence[str],
    optional: Sequence[str],
    optional_k: int,
) -> None:
    total_lower, total_upper = map(float, config["total_bounds"])
    required_lower = sum(_active_floor(config, element) for element in required)
    required_upper = sum(
        _component_bounds(config, element)[1] for element in required
    )
    optional_lower = [_active_floor(config, element) for element in optional]
    optional_upper = [_component_bounds(config, element)[1] for element in optional]

    largest_lower = sum(sorted(optional_lower, reverse=True)[:optional_k])
    smallest_upper = sum(sorted(optional_upper)[:optional_k])
    if required_lower + largest_lower > total_upper + _TOLERANCE:
        raise ValueError(
            "Variable-total composition best_subset has a support whose active lower "
            "bounds exceed the upper total bound."
        )
    if required_upper + smallest_upper < total_lower - _TOLERANCE:
        raise ValueError(
            "Variable-total composition best_subset has a support whose active upper "
            "bounds cannot reach the lower total bound."
        )


def _raw_fixed_features_list_from_training(
    train_x: Tensor | None,
    cat_dims: Sequence[int],
) -> list[dict[int, float]] | None:
    if train_x is None or not cat_dims:
        return None
    values = train_x[..., list(cat_dims)]
    if values.ndim > 2:
        values = values.reshape(-1, len(cat_dims))
    unique = torch.unique(values, dim=0)
    if unique.numel() == 0:
        return None
    return [
        {
            int(index): float(value)
            for index, value in zip(cat_dims, row, strict=True)
        }
        for row in unique.detach().cpu().tolist()
    ]


def prepare_variable_total_best_subset_config(
    opt_config: OptimizeConfig,
    *,
    site_name: str,
    site_config: Mapping[str, Any],
    transformer: Any,
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
) -> tuple[CompositionVariableTotalDecisionBridge, OptimizeConfig, Tensor]:
    """Build raw absolute-amount decision config for variable-total Best Subset."""

    if not is_variable_total_best_subset_site(site_config):
        raise ValueError(
            f"Composition site {site_name!r} is not variable-total best_subset."
        )

    bridge = CompositionVariableTotalDecisionBridge.from_transformer(
        transformer,
        model_feature_names,
        total_feature=str(site_config["total_feature"]),
    )
    raw_bounds = bridge.decision_bounds(
        model_bounds,
        component_bounds=site_config.get("bounds") or {},
        total_bounds=site_config["total_bounds"],
    )
    raw_config = _remap_optimize_config(opt_config, bridge, raw_bounds)
    repair = raw_config.repair_config or CandidateRepairConfig()
    if repair.comp_idx not in (None, [], ()):
        raise ValueError(
            "Variable-total composition best_subset owns CandidateRepairConfig.comp_idx."
        )
    if int(repair.k) != 0:
        raise ValueError(
            "Variable-total composition best_subset derives cardinality from "
            "min_components/max_components."
        )
    if repair.final_sum_constraint is not None:
        raise ValueError(
            "Variable-total composition best_subset does not use final_sum_constraint; "
            "the total is the sum of raw element amounts."
        )

    elements = tuple(transformer.fitted_elements)
    by_element = dict(zip(elements, bridge.amount_names, strict=True))
    explicit_required = set(site_config.get("required_components") or ())
    explicit_forbidden = set(site_config.get("forbidden_components") or ())
    required = set(explicit_required)
    forbidden = set(explicit_forbidden)

    optimizer_fixed = dict(raw_config.fixed_features or {})
    repair_fixed = dict(repair.fixed_features or {})
    all_fixed = {**optimizer_fixed, **repair_fixed}
    for element, amount_name in by_element.items():
        lower, upper = _component_bounds(site_config, element)
        if lower > _TOLERANCE:
            required.add(element)
        if upper <= _TOLERANCE:
            forbidden.add(element)
        if amount_name in all_fixed:
            if abs(float(all_fixed[amount_name])) <= _TOLERANCE:
                forbidden.add(element)
            else:
                required.add(element)

    overlap = required & forbidden
    if overlap:
        raise ValueError(
            "Variable-total composition has components that are both required and "
            f"forbidden: {sorted(overlap)!r}."
        )
    optional = [
        element
        for element in elements
        if element not in required and element not in forbidden
    ]
    ordered_required = sorted(required, key=elements.index)
    cardinality = resolve_composition_cardinality_range(
        site_config,
        required_count=len(required),
        optional_count=len(optional),
        context=f"Variable-total composition site {site_name!r} best_subset",
    )
    require_exact_cardinality_for_steps(
        site_config,
        cardinality,
        context=f"Variable-total composition site {site_name!r} best_subset",
    )

    effective_optimizer_kwargs = _optimizer_kwargs_for_site(raw_config, site_config)
    search_optimizer_kwargs = apply_optional_cardinality_range(
        effective_optimizer_kwargs,
        cardinality,
        context=f"Variable-total composition site {site_name!r} best_subset",
    )
    if site_config.get("steps"):
        _validate_grid_strategy(
            config=site_config,
            optimizer_kwargs=search_optimizer_kwargs,
            optional_count=len(optional),
            optional_k=cardinality.optional_maximum,
        )
    _validate_grid_contract(
        raw_config=raw_config,
        repair=repair,
        site_config=site_config,
        bridge=bridge,
        all_fixed=all_fixed,
    )
    for optional_k in cardinality.optional_cardinalities:
        _validate_support_feasibility(
            config=site_config,
            required=ordered_required,
            optional=optional,
            optional_k=optional_k,
        )

    final_candidate_postprocess = _grid_postprocess(
        raw_config=raw_config,
        repair=repair,
        site_config=site_config,
        bridge=bridge,
        exact_k=cardinality.maximum,
    )
    if site_config.get("steps"):
        _validate_grid_supports(
            projector=final_candidate_postprocess,
            site_config=site_config,
            required=ordered_required,
            optional=optional,
            optional_k=cardinality.optional_maximum,
        )

    forbidden_fixed = {by_element[element]: 0.0 for element in forbidden}
    optimizer_fixed.update(forbidden_fixed)
    repair_fixed.update(forbidden_fixed)

    amount_names = list(bridge.amount_names)
    total_lower, total_upper = map(float, site_config["total_bounds"])
    optimizer_inequalities = list(raw_config.inequality_constraints or ())
    repair_inequalities = list(repair.inequality_constraints or ())
    optimizer_inequalities.extend(
        [
            (amount_names, [1.0] * len(amount_names), total_lower),
            (amount_names, [-1.0] * len(amount_names), -total_upper),
        ]
    )
    repair_inequalities.extend(
        [
            (amount_names, [1.0] * len(amount_names), total_lower),
            (amount_names, [-1.0] * len(amount_names), -total_upper),
        ]
    )

    for element in elements:
        if element in forbidden:
            continue
        floor = _active_floor(site_config, element)
        amount_name = by_element[element]
        repair_inequalities.append(([amount_name], [1.0], floor))
        if element in required and amount_name not in optimizer_fixed:
            optimizer_inequalities.append(([amount_name], [1.0], floor))

    optional_names = [by_element[element] for element in optional]
    if cardinality.optional_maximum == 0:
        repair = replace(
            repair,
            comp_idx=None,
            k=0,
            support_selection="topk",
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
        )
        optimizer_kwargs = _without_cardinality_kwargs(raw_config.optimizer_kwargs)
    else:
        repair = replace(
            repair,
            comp_idx=optional_names,
            k=cardinality.optional_maximum,
            support_selection="best_subset",
            inequality_constraints=repair_inequalities,
            inequality_sense="ge",
            fixed_features=repair_fixed or None,
        )
        optimizer_kwargs = search_optimizer_kwargs

    raw_config = replace(
        raw_config,
        repair_config=repair,
        fixed_features=optimizer_fixed or None,
        inequality_constraints=optimizer_inequalities,
        optimizer_kwargs=optimizer_kwargs,
        final_candidate_postprocess=final_candidate_postprocess,
    )
    raw_config = resolve_optimize_config_columns(
        raw_config,
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
        inferred = _raw_fixed_features_list_from_training(raw_train_x, raw_cat_dims)
        if inferred:
            raw_config = replace(raw_config, fixed_features_list=inferred)

    return bridge, raw_config, raw_bounds


@dataclass(frozen=True)
class VariableTotalBestSubsetResult:
    """Raw amount-space and fitted model-space candidate result."""

    candidates: Tensor
    raw_candidates: Tensor
    acq_value: Any
    raw_opt_config: OptimizeConfig
    bridge: CompositionVariableTotalDecisionBridge


def _reject_one_shot_acquisition(acqf: Any) -> None:
    try:
        from botorch.acquisition.acquisition import OneShotAcquisitionFunction
    except ImportError:
        return
    if isinstance(acqf, OneShotAcquisitionFunction):
        raise NotImplementedError(
            "Variable-total raw-space composition best_subset does not yet support "
            "one-shot acquisition functions such as KG."
        )


def optimize_variable_total_best_subset(
    base_acqf: Any,
    opt_config: OptimizeConfig,
    *,
    site_name: str,
    site_config: Mapping[str, Any],
    transformer: Any,
    model_feature_names: Sequence[Any],
    model_bounds: Tensor,
    dtype: Any = None,
    device: Any = None,
    model_cat_dims: Sequence[int] | None = None,
    train_x: Tensor | None = None,
    optimize_fn: Callable[..., tuple[Any, Any]] | None = None,
) -> VariableTotalBestSubsetResult:
    """Optimize support cardinality, support, and total jointly in raw amounts."""

    _reject_one_shot_acquisition(base_acqf)
    bridge, raw_config, raw_bounds = prepare_variable_total_best_subset_config(
        opt_config,
        site_name=site_name,
        site_config=site_config,
        transformer=transformer,
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
    return VariableTotalBestSubsetResult(
        candidates=model_candidates,
        raw_candidates=raw_candidates,
        acq_value=final_value,
        raw_opt_config=raw_config,
        bridge=bridge,
    )


__all__ = [
    "CompositionVariableTotalDecisionBridge",
    "VariableTotalBestSubsetResult",
    "is_variable_total_best_subset_site",
    "optimize_variable_total_best_subset",
    "prepare_variable_total_best_subset_config",
    "resolve_variable_total_best_subset_site",
]
