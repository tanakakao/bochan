'''Config builders used by the tabular convenience API.'''

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from bochan.api import (
    AcquisitionConfig,
    CandidateRepairConfig,
    FitConfig,
    ModelConfig,
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


def _replace_or_create(cls: type, base: Any | None, values: dict[str, Any]) -> Any | None:
    if base is None:
        return cls(**values) if values else None
    return replace(base, **values) if values else base


def make_model_config(model_config: ModelConfig | None = None, **values: Any) -> ModelConfig:
    '''Create or update ``ModelConfig`` from direct keyword arguments.'''

    values = drop_unset(values)
    model_values = _take_fields(ModelConfig, values)
    if values:
        unknown = sorted(values)
        raise TypeError(f"Unknown ModelConfig arguments: {unknown!r}.")
    if model_config is None:
        return ModelConfig(**model_values)
    return replace(model_config, **model_values) if model_values else model_config


def make_fit_config(fit_config: FitConfig | None = None, **values: Any) -> FitConfig | None:
    '''Create or update ``FitConfig`` from direct keyword arguments.'''

    values = drop_unset(values)
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
    objective_config: ObjectiveConfig | None = None,
    **values: Any,
) -> ObjectiveConfig | None:
    '''Create or update ``ObjectiveConfig`` from direct keyword arguments.'''

    values = drop_unset(values)
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
    acq_config: AcquisitionConfig | None = None,
    **values: Any,
) -> AcquisitionConfig:
    '''Create or update ``AcquisitionConfig`` from direct keyword arguments.'''

    values = drop_unset(values)
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
    repair_config: CandidateRepairConfig | None = None,
    **values: Any,
) -> CandidateRepairConfig | None:
    '''Create or update ``CandidateRepairConfig`` from direct keyword arguments.'''

    values = drop_unset(values)
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


def make_optimize_config(opt_config: OptimizeConfig | None = None, **values: Any) -> OptimizeConfig:
    '''Create or update ``OptimizeConfig`` and nested repair config.'''

    values = drop_unset(values)

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
