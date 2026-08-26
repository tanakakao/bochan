"""Construction-time configuration for the tabular optimizer facade."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from typing import Any

from bochan.api import (
    BayesianOptimizer,
    CrossValidationConfig,
    ExperimentFailureConfig,
    FitConfig,
    ModelConfig,
)

from ..composition import CompositionAdapter
from ..config import UNSET, ColumnKey, TabularDataConfig, make_fit_config, make_model_config
from ..observation import ObservationAdapter
from ..structure import StructureTabularAdapter
from ..targets import extract_output_category_maps, merge_target_category_metadata
from .candidates import CandidateService
from .diagnostics import DiagnosticsService
from .settings import merge_input_transform_config, validate_noise_alpha

MODEL_KEYS = {field.name for field in fields(ModelConfig)}
FIT_KEYS = {field.name for field in fields(FitConfig)} | {"fit_method", "fit_optimizer_kwargs"}
DATA_KEYS = {field.name for field in fields(TabularDataConfig)}


def take(values: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """Pop direct API values whose names belong to a canonical config."""

    return {key: values.pop(key) for key in list(values) if key in keys}


def merge_data_config(
    base: TabularDataConfig | None,
    values: Mapping[str, Any],
) -> TabularDataConfig:
    config = base or TabularDataConfig()
    updates = {key: value for key, value in values.items() if value is not None}
    return replace(config, **updates) if updates else config


def resolve_cv_config(
    value: CrossValidationConfig | Mapping[str, Any] | None,
) -> CrossValidationConfig | None:
    if value is None or isinstance(value, CrossValidationConfig):
        return value
    payload = dict(value)
    importance = payload.get("feature_importance_config")
    if isinstance(importance, Mapping):
        from bochan.inspection import FeatureImportanceConfig

        payload["feature_importance_config"] = FeatureImportanceConfig(**dict(importance))
    return CrossValidationConfig(**payload)


def initialize_optimizer(
    owner: Any,
    model_config: ModelConfig | Mapping[str, Any] | None,
    fit_config: FitConfig | Mapping[str, Any] | None,
    *,
    composition_sites: Mapping[str, Mapping[str, Any]] | None,
    composition_total_constraints: Sequence[Any] | None,
    composition_element_constraints: Sequence[Any] | None,
    composition_constraint_rerank: bool,
    composition_constraint_rerank_factor: int,
    composition_constraint_max_supports: int,
    structure_col: ColumnKey | None,
    structure_catalog: Mapping[Any, Any] | None,
    structure_graph_builder: Any | None,
    data_config: TabularDataConfig | None,
    data: Any | None,
    cross_validation: bool,
    cv_config: CrossValidationConfig | Mapping[str, Any] | None,
    failure_config: ExperimentFailureConfig | None,
    target_missing_strategy: str | None,
    experiment_status_col: ColumnKey | None,
    alpha: float | None,
    beta: Any,
    normalize: Any,
    perturbation: Any,
    n_w: Any,
    std: Any,
    kwargs: dict[str, Any],
) -> None:
    """Build canonical config objects and explicit optimizer components."""

    inferred_maps: dict[Any, dict[Any, int]] = {}
    if isinstance(model_config, Mapping):
        payload = dict(model_config)
        multi_output = payload.get("multi_output_config")
        if multi_output is not None:
            payload["multi_output_config"], maps = extract_output_category_maps(multi_output)
            inferred_maps.update(maps)
        model_config = payload

    direct_multi_output = kwargs.get("multi_output_config")
    if direct_multi_output is not None:
        resolved, maps = extract_output_category_maps(direct_multi_output)
        kwargs["multi_output_config"] = resolved
        for output_name, category_map in maps.items():
            existing = inferred_maps.get(output_name)
            if existing is not None and existing != category_map:
                raise ValueError(
                    f"Conflicting category declarations for output {output_name!r}."
                )
            inferred_maps[output_name] = category_map
    merge_target_category_metadata(kwargs, inferred_maps)

    model_values = take(kwargs, MODEL_KEYS)
    transform_config = merge_input_transform_config(
        model_config=model_config,
        input_transform_config=model_values.get("input_transform_config", UNSET),
        normalize=normalize,
        perturbation=perturbation,
        n_w=n_w,
        std=std,
    )
    if transform_config is not UNSET:
        model_values["input_transform_config"] = transform_config
    owner.model_config = make_model_config(model_config, **model_values)

    fit_values = take(kwargs, FIT_KEYS)
    if beta is not UNSET:
        fit_values["beta"] = beta
    owner.fit_config = make_fit_config(fit_config, **fit_values)

    owner.observation = ObservationAdapter(failure_config)
    source_config = merge_data_config(data_config, take(kwargs, DATA_KEYS))
    source_config = owner.observation.resolve_config(
        source_config,
        target_missing_strategy=target_missing_strategy,
        experiment_status_col=experiment_status_col,
    )
    owner.source_data_config = source_config
    owner.data_config = source_config

    owner.composition = CompositionAdapter(composition_sites)
    owner.structure = StructureTabularAdapter(
        column=structure_col,
        catalog=structure_catalog,
        graph_builder=structure_graph_builder,
    )
    owner.candidates = CandidateService(
        composition=owner.composition,
        structure=owner.structure,
        total_constraints=composition_total_constraints,
        element_constraints=composition_element_constraints,
        rerank=composition_constraint_rerank,
        rerank_factor=composition_constraint_rerank_factor,
        max_supports=composition_constraint_max_supports,
    )
    owner.diagnostics = DiagnosticsService()

    owner.alpha = validate_noise_alpha(alpha)
    owner.data = data
    owner.cross_validation = bool(cross_validation)
    owner.cv_config = resolve_cv_config(cv_config)
    owner.cross_validation_result_ = None
    owner.bo_kwargs = dict(kwargs)
    owner.bo = BayesianOptimizer(
        model_config=owner.model_config,
        fit_config=owner.fit_config,
        **kwargs,
    )
    owner.dataset = None
    owner.feature_importance_result_ = None


__all__ = [
    "DATA_KEYS",
    "FIT_KEYS",
    "MODEL_KEYS",
    "initialize_optimizer",
    "merge_data_config",
    "resolve_cv_config",
    "take",
]
