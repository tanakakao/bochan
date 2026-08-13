'''Config builders used by the tabular convenience API.'''

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from bochan.api import (
    AcquisitionConfig,
    CandidateRepairConfig,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
)

UNSET = object()


def drop_unset(values: dict[str, Any]) -> dict[str, Any]:
    '''Drop values that were not explicitly supplied.'''

    return {key: value for key, value in values.items() if value is not UNSET}


def _field_names(cls: type) -> set[str]:
    return {field.name for field in fields(cls)}


def _take_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = _field_names(cls)
    taken: dict[str, Any] = {}
    for key in list(values):
        if key in names:
            taken[key] = values.pop(key)
    return taken


def _take_prefixed_fields(
    values: dict[str, Any],
    field_map: Mapping[str, str],
) -> dict[str, Any]:
    '''Map explicit flattened tabular fields to nested config field names.'''

    taken: dict[str, Any] = {}
    for source, target in field_map.items():
        if source in values:
            taken[target] = values.pop(source)
    return taken


def _normalize_constraint_sense(sense: Any) -> str:
    value = str(sense).strip().lower()
    aliases = {
        "eq": "eq",
        "=": "eq",
        "==": "eq",
        "equal": "eq",
        "equals": "eq",
        "ge": "ge",
        ">=": "ge",
        "=>": "ge",
        "gte": "ge",
        "greater_equal": "ge",
        "greater_than_or_equal": "ge",
        "le": "le",
        "<=": "le",
        "=<": "le",
        "lte": "le",
        "less_equal": "le",
        "less_than_or_equal": "le",
    }
    if value not in aliases:
        raise ValueError(
            "Constraint sense must be one of 'eq', 'ge', 'le', '=', '==', '>=', or '<='. "
            f"Got {sense!r}."
        )
    return aliases[value]


def _negate_numeric_sequence_or_value(value: Any) -> Any:
    try:
        return -value
    except TypeError:
        return [-float(item) for item in value]


def _append_constraints(existing: Any | None, extra: list[Any]) -> Any | None:
    if not extra:
        return existing
    if existing is None:
        return extra
    return list(existing) + list(extra)


def _split_linear_constraints(constraints: Any | None) -> tuple[list[Any], list[Any]]:
    '''Split unified tabular constraints into equality and BoTorch-style inequalities.'''

    if constraints is None:
        return [], []

    equality_constraints: list[Any] = []
    inequality_constraints: list[Any] = []
    for constraint in constraints:
        if len(constraint) != 4:
            raise ValueError(
                "Each constraint must be (indices, coefficients, sense, rhs). "
                f"Got {constraint!r}."
            )
        indices, coefficients, sense, rhs = constraint
        normalized_sense = _normalize_constraint_sense(sense)
        if normalized_sense == "eq":
            equality_constraints.append((indices, coefficients, rhs))
        elif normalized_sense == "ge":
            inequality_constraints.append((indices, coefficients, rhs))
        else:
            inequality_constraints.append(
                (
                    indices,
                    _negate_numeric_sequence_or_value(coefficients),
                    _negate_numeric_sequence_or_value(rhs),
                )
            )
    return equality_constraints, inequality_constraints


def _merge_base_dict(
    base: Any | None,
    values: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    '''Merge a config mapping with explicitly supplied direct values.'''

    if isinstance(base, Mapping):
        merged = dict(base)
        merged.update(values)
        return None, merged
    return base, values


def _replace_or_create(cls: type, base: Any | None, values: dict[str, Any]) -> Any | None:
    if isinstance(base, Mapping):
        payload = dict(base)
        payload.update(values)
        return cls(**payload)
    if base is None:
        return cls(**values) if values else None
    return replace(base, **values) if values else base


def _coerce_input_transform_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return InputTransformConfig(**dict(value))
    return value


def _coerce_fit_config_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return make_fit_config(value)
    return value


def _coerce_multi_output_config(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value

    payload = dict(value)
    output_fit_configs = payload.get("output_fit_configs")
    if isinstance(output_fit_configs, Mapping):
        payload["output_fit_configs"] = make_fit_config(output_fit_configs)
    elif isinstance(output_fit_configs, (list, tuple)):
        payload["output_fit_configs"] = [
            _coerce_fit_config_like(item) for item in output_fit_configs
        ]
    return MultiOutputConfig(**payload)


def _coerce_model_config_values(values: dict[str, Any]) -> dict[str, Any]:
    if "input_transform_config" in values:
        values["input_transform_config"] = _coerce_input_transform_config(
            values["input_transform_config"]
        )
    if "multi_output_config" in values:
        values["multi_output_config"] = _coerce_multi_output_config(
            values["multi_output_config"]
        )
    return values


def make_model_config(
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> ModelConfig:
    '''Create or update ``ModelConfig`` from direct keyword arguments or a mapping.'''

    values = drop_unset(values)
    model_config, values = _merge_base_dict(model_config, values)
    model_values = _take_fields(ModelConfig, values)
    model_values = _coerce_model_config_values(model_values)
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown ModelConfig arguments: {unknown!r}.")
    if model_config is None:
        return ModelConfig(**model_values)
    return replace(model_config, **model_values) if model_values else model_config


def make_fit_config(
    fit_config: FitConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> FitConfig | None:
    '''Create or update ``FitConfig`` from canonical or flattened tabular fields.'''

    values = drop_unset(values)
    fit_config, values = _merge_base_dict(fit_config, values)
    fit_values = _take_prefixed_fields(
        values,
        {
            "fit_method": "method",
            "fit_optimizer_kwargs": "optimizer_kwargs",
        },
    )
    fit_values.update(_take_fields(FitConfig, values))
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown FitConfig arguments: {unknown!r}.")
    return _replace_or_create(FitConfig, fit_config, fit_values)


def make_objective_config(
    objective_config: ObjectiveConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> ObjectiveConfig | None:
    '''Create or update ``ObjectiveConfig`` from canonical or flattened fields.'''

    values = drop_unset(values)
    objective_config, values = _merge_base_dict(objective_config, values)
    objective_values = _take_prefixed_fields(
        values,
        {
            "objective_mode": "mode",
            "objective_output": "output",
            "objective_outputs": "outputs",
            "objective_specs": "specs",
            "objective_directions": "directions",
            "objective_weights": "weights",
            "objective_eq_targets": "eq_targets",
            "objective_direction": "direction",
            "objective_weight": "weight",
            "objective_eq_target": "eq_target",
            "objective_n_w": "n_w",
            "objective_risk_type": "risk_type",
            "objective_alpha": "alpha",
            "objective_maximize": "maximize",
            "objective_aggregate_mean_when_no_risk": "aggregate_mean_when_no_risk",
            "objective_allow_unexpanded": "allow_unexpanded",
            "objective_utility_values": "utility_values",
            "objective_ordinal_likelihood": "ordinal_likelihood",
            "objective_kwargs": "objective_kwargs",
        },
    )
    objective_values.update(_take_fields(ObjectiveConfig, values))
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown ObjectiveConfig arguments: {unknown!r}.")
    return _replace_or_create(ObjectiveConfig, objective_config, objective_values)


def make_acquisition_config(
    acq_config: AcquisitionConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> AcquisitionConfig:
    '''Create or update ``AcquisitionConfig`` from direct fields or a mapping.'''

    values = drop_unset(values)
    acq_config, values = _merge_base_dict(acq_config, values)

    direct_name = values.pop("acq_name", UNSET)
    config_name = values.get("name", UNSET)
    if direct_name is not UNSET and config_name is not UNSET:
        raise ValueError("Specify either acq_name or name, not both.")
    if direct_name is not UNSET:
        values["name"] = direct_name

    objective_config = values.pop("objective_config", None)
    objective_direct: dict[str, Any] = {}
    for key in list(values):
        if key.startswith("objective_"):
            objective_direct[key] = values.pop(key)
    if objective_config is not None or objective_direct:
        objective_config = make_objective_config(
            objective_config,
            **objective_direct,
        )

    acq_values = _take_fields(AcquisitionConfig, values)
    if objective_config is not None:
        acq_values["objective_config"] = objective_config

    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown AcquisitionConfig arguments: {unknown!r}.")

    if acq_config is None:
        if "name" not in acq_values:
            raise ValueError("acq_name or acq_config is required.")
        return AcquisitionConfig(**acq_values)
    return replace(acq_config, **acq_values) if acq_values else acq_config


def make_repair_config(
    repair_config: CandidateRepairConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> CandidateRepairConfig | None:
    '''Create or update ``CandidateRepairConfig`` from canonical or flattened fields.'''

    values = drop_unset(values)
    repair_config, values = _merge_base_dict(repair_config, values)
    repair_values = _take_prefixed_fields(
        values,
        {
            "repair_bounds": "bounds",
            "repair_fixed_features": "fixed_features",
            "repair_equality_constraints": "equality_constraints",
            "repair_inequality_constraints": "inequality_constraints",
            "repair_inequality_sense": "inequality_sense",
        },
    )
    repair_values.update(_take_fields(CandidateRepairConfig, values))
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown CandidateRepairConfig arguments: {unknown!r}.")
    return _replace_or_create(CandidateRepairConfig, repair_config, repair_values)


def _repair_config_has_explicit_inequality_sense(repair_config: Any) -> bool:
    return isinstance(repair_config, Mapping) and "inequality_sense" in repair_config


def _repair_config_has_explicit_inequality_constraints(repair_config: Any) -> bool:
    if isinstance(repair_config, Mapping):
        return "inequality_constraints" in repair_config
    return getattr(repair_config, "inequality_constraints", None) is not None


def make_optimize_config(
    opt_config: OptimizeConfig | Mapping[str, Any] | None = None,
    **values: Any,
) -> OptimizeConfig:
    '''Create or update ``OptimizeConfig`` and its nested repair config.'''

    values = drop_unset(values)
    opt_config, values = _merge_base_dict(opt_config, values)

    unified_constraints = values.pop("constraints", None)
    parsed_equalities, parsed_inequalities = _split_linear_constraints(
        unified_constraints
    )
    if parsed_equalities:
        values["equality_constraints"] = _append_constraints(
            values.get("equality_constraints"),
            parsed_equalities,
        )
    if parsed_inequalities:
        values["inequality_constraints"] = _append_constraints(
            values.get("inequality_constraints"),
            parsed_inequalities,
        )

    repair_config = values.pop("repair_config", None)
    opt_values = _take_fields(OptimizeConfig, values)

    repair_direct: dict[str, Any] = {}
    repair_field_names = _field_names(CandidateRepairConfig) - _field_names(OptimizeConfig)
    repair_flattened_fields = {
        "repair_bounds",
        "repair_fixed_features",
        "repair_equality_constraints",
        "repair_inequality_constraints",
        "repair_inequality_sense",
    }
    for key in list(values):
        if key in repair_field_names or key in repair_flattened_fields:
            repair_direct[key] = values.pop(key)

    if (
        parsed_inequalities
        and (repair_config is not None or repair_direct)
        and not _repair_config_has_explicit_inequality_constraints(repair_config)
        and "repair_inequality_constraints" not in repair_direct
        and "inequality_constraints" not in repair_direct
        and not _repair_config_has_explicit_inequality_sense(repair_config)
        and "repair_inequality_sense" not in repair_direct
        and "inequality_sense" not in repair_direct
    ):
        repair_direct["repair_inequality_sense"] = "ge"

    if repair_config is not None or repair_direct:
        repair_config = make_repair_config(repair_config, **repair_direct)

    if repair_config is not None:
        opt_values["repair_config"] = repair_config
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown OptimizeConfig arguments: {unknown!r}.")

    if opt_config is None:
        return OptimizeConfig(**opt_values)
    return replace(opt_config, **opt_values) if opt_values else opt_config
