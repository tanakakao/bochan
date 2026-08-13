"""Fit and dataset-conversion lifecycle for the tabular optimizer facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from bochan.api import CrossValidationConfig

from ..config import UNSET, make_fit_config, make_model_config
from ..data import dataframe_to_tensors, numpy_to_tensors
from .configuration import DATA_KEYS, FIT_KEYS, MODEL_KEYS, merge_data_config, resolve_cv_config, take
from .settings import apply_alpha_to_model_config, merge_input_transform_config, validate_noise_alpha


def default_to_dataset(owner: Any, data: Any, y: Any | None = None, *, data_config: Any = None, feature_names: Any = None, target_names: Any = None) -> Any:
    config = data_config or owner.data_config
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return dataframe_to_tensors(data, config)
    return numpy_to_tensors(data, y, config, feature_names=feature_names, target_names=target_names)


def to_dataset(owner: Any, data: Any, y: Any | None = None, *, data_config: Any = None, feature_names: Any = None, target_names: Any = None) -> Any:
    config = data_config or owner.data_config
    def converter(value: Any, target: Any = None, **converter_kwargs: Any) -> Any:
        return default_to_dataset(owner, value, target, **converter_kwargs)
    return owner.observation.to_dataset(data, y, config=config, feature_names=feature_names, target_names=target_names, default_converter=converter)


def model_config_for_dataset(owner: Any, dataset: Any) -> Any:
    config = owner.model_config
    if config.cat_dims is None and dataset.cat_dims:
        config = replace(config, cat_dims=dataset.cat_dims)
    return apply_alpha_to_model_config(config, train_X=dataset.X, train_Y=dataset.Y, explicit_alpha=owner.alpha)


def sync_visualization_metadata(owner: Any) -> None:
    if owner.dataset is None or owner.bo.bundle is None:
        return
    metadata = dict(getattr(owner.bo.bundle, "metadata", {}) or {})
    metadata["feature_cols"] = list(owner.dataset.feature_names)
    metadata["target_cols"] = list(owner.dataset.target_names)
    if owner.dataset.category_maps:
        labels = dict(metadata.get("labels") or {})
        labels.update(owner.dataset.category_maps)
        metadata["labels"] = labels
    owner.bo.bundle.metadata = metadata


def fit_optimizer(owner: Any, data: Any | None, y: Any | None, *, data_config: Any, alpha: Any, beta: Any, normalize: Any, perturbation: Any, n_w: Any, std: Any, target_missing_strategy: Any, experiment_status_col: Any, failure_config: Any, cross_validation: bool | None, cv_config: Any, kwargs: dict[str, Any]) -> Any:
    data = owner.data if data is None else data
    if data is None:
        raise ValueError("No data was supplied. Pass data to fit(...) or use from_csv(...).")
    if alpha is not UNSET:
        owner.alpha = validate_noise_alpha(alpha)
    model_values = take(kwargs, MODEL_KEYS)
    supplied_model = kwargs.pop("model_config", None)
    transform_config = merge_input_transform_config(model_config=supplied_model or owner.model_config, input_transform_config=model_values.get("input_transform_config", UNSET), normalize=normalize, perturbation=perturbation, n_w=n_w, std=std)
    if transform_config is not UNSET:
        model_values["input_transform_config"] = transform_config
    owner.model_config = make_model_config(supplied_model or owner.model_config, **model_values)
    fit_values = take(kwargs, FIT_KEYS)
    supplied_fit = kwargs.pop("fit_config", None)
    if beta is not UNSET:
        fit_values["beta"] = beta
    owner.fit_config = make_fit_config(supplied_fit or owner.fit_config, **fit_values)
    source_config = merge_data_config(data_config or owner.source_data_config, take(kwargs, DATA_KEYS))
    source_config = owner.observation.resolve_config(source_config, target_missing_strategy=target_missing_strategy, experiment_status_col=experiment_status_col)
    owner.source_data_config = source_config
    fit_data = data
    resolved = source_config
    if owner.composition.enabled:
        fit_data = owner.composition.prepare_frame(data, fit_transformers=True)
        resolved = replace(source_config, input_cols=owner.composition.replace_input_cols(source_config.input_cols), categorical_cols=owner.composition.resolve_categorical_cols(source_config.categorical_cols, default_categorical_cols=source_config.categorical_cols or ()), bounds=owner.composition.expanded_bounds(source_config.bounds, fit_data))
    owner.data_config = resolved
    run_cv = owner.cross_validation if cross_validation is None else bool(cross_validation)
    if owner.observation.uses_observation_conversion(resolved) and run_cv:
        raise ValueError("Cross-validation requires an observation-aware validation protocol.")
    dataset = to_dataset(owner, fit_data, y, data_config=resolved)
    if dataset.Y is None:
        raise ValueError("Target values are required for fit(). Set target_cols or pass y.")
    owner.dataset = dataset
    model_config = model_config_for_dataset(owner, dataset)
    resolved_cv = resolve_cv_config(cv_config) if cv_config is not None else owner.cv_config
    owner.cross_validation_result_ = None
    if run_cv:
        owner.cross_validation_result_ = owner.bo.cross_validate(dataset.X, dataset.Y, model_config=model_config, fit_config=owner.fit_config, cv_config=resolved_cv or CrossValidationConfig())
    owner.bo.fit(dataset.X, dataset.Y, model_config=model_config, fit_config=owner.fit_config)
    if dataset.bounds is not None:
        owner.bo.set_bounds(dataset.bounds)
    owner.observation.attach(owner.bo, dataset, failure_config=owner.observation.resolve_failure_config(failure_config))
    sync_visualization_metadata(owner)
    return owner


__all__ = ["default_to_dataset", "fit_optimizer", "model_config_for_dataset", "sync_visualization_metadata", "to_dataset"]
