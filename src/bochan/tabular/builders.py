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


def _take_aliases(values: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    taken: dict[str, Any] = {}
    for alias, target in aliases.items():
        if alias in values:
            taken[target] = values.pop(alias)
    return taken


def _merge_base_dict(base: Any | None, values: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    '''Merge a user-facing config dict with direct override values.'''

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
    elif isinstance(output_fit_configs, list | tuple):
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


def make_model_config(model_config: ModelConfig | Mapping[str, Any] | None = None, **values: Any) -> ModelConfig:
    '''Create or update ``ModelConfig`` from direct keyword arguments or a dict.'''

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


def make_fit_config(fit_config: FitConfig | Mapping[str, Any] | None = None, **values: Any) -> FitConfig | None:
    '''Create or update ``FitConfig`` from direct keyword arguments or a dict.'''

    values = drop_unset(values)
    fit_config, values = _merge_base_dict(fit_config, values)
    fit_values = _take_aliases(
        values,
        {
            "fit_method": "method",
            "fit_optimizer_kwargs": "optimizer_kwargs",
            "fit_beta": "beta",
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
    '''Create or update ``ObjectiveConfig`` from direct keyword arguments or a dict.'''

    values = drop_unset(values)
    objective_config, values = _merge_base_dict(objective_config, values)
    objective_values = _take_aliases(
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
    '''Create or update ``AcquisitionConfig`` from direct keyword arguments or a dict.'''

    values = drop_unset(values)
    acq_config, values = _merge_base_dict(acq_config, values)
    name = values.pop("acq_name", UNSET)
    if name is UNSET:
        name = values.pop("name", UNSET)

    objective_config = values.pop("objective_config", None)
    objective_direct: dict[str, Any] = {}
    for key in list(values):
        if key.startswith("objective_"):
            objective_direct[key] = values.pop(key)
    if objective_config is not None or objective_direct:
        objective_config = make_objective_config(objective_config, **objective_direct)

    acq_values = _take_fields(AcquisitionConfig, values)
    if name is not UNSET:
        acq_values["name"] = name
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
    '''Create or update ``CandidateRepairConfig`` from direct keyword arguments or a dict.'''

    values = drop_unset(values)
    repair_config, values = _merge_base_dict(repair_config, values)
    repair_values = _take_aliases(
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


def make_optimize_config(opt_config: OptimizeConfig | Mapping[str, Any] | None = None, **values: Any) -> OptimizeConfig:
    '''Create or update ``OptimizeConfig`` and nested repair config from direct kwargs or a dict.'''

    values = drop_unset(values)
    opt_config, values = _merge_base_dict(opt_config, values)

    repair_config = values.pop("repair_config", None)
    repair_direct: dict[str, Any] = {}
    repair_field_names = _field_names(CandidateRepairConfig)
    repair_aliases = {
        "repair_bounds",
        "repair_fixed_features",
        "repair_equality_constraints",
        "repair_inequality_constraints",
        "repair_inequality_sense",
    }
    for key in list(values):
        if key in repair_field_names or key in repair_aliases:
            repair_direct[key] = values.pop(key)
    if repair_config is not None or repair_direct:
        repair_config = make_repair_config(repair_config, **repair_direct)

    opt_values = _take_fields(OptimizeConfig, values)
    if repair_config is not None:
        opt_values["repair_config"] = repair_config
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown OptimizeConfig arguments: {unknown!r}.")

    if opt_config is None:
        return OptimizeConfig(**opt_values)
    return replace(opt_config, **opt_values) if opt_values else opt_config
