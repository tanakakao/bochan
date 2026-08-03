"""Adapters that run the React Web workflow through TabularBayesianOptimizer."""

from __future__ import annotations

from typing import Any

from .feature_importance_outputs import relabel_feature_importance_outputs


def _category_key_from_label(series: Any, label: Any) -> Any:
    """Recover an original category value from the Web encoder's string label."""

    observed = series.dropna().unique().tolist()
    for value in observed:
        if value == label or str(value) == str(label):
            return value

    try:
        import pandas as pd

        if pd.api.types.is_integer_dtype(series.dtype):
            return int(float(label))
        if pd.api.types.is_float_dtype(series.dtype):
            return float(label)
    except (TypeError, ValueError):
        pass
    return label


def feature_category_maps(
    data: Any,
    encoded_features: dict[str, Any],
) -> dict[str, dict[Any, int]]:
    """Convert Web category maps to original-label maps accepted by the tabular API."""

    maps: dict[str, dict[Any, int]] = {}
    for column, encoded_map in dict(encoded_features.get("category_maps") or {}).items():
        series = data[column]
        maps[str(column)] = {
            _category_key_from_label(series, label): int(index)
            for label, index in dict(encoded_map).items()
        }
    return maps


def target_category_maps(
    target_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[Any, int]]:
    """Build explicit target maps so classification and ordinal ranks stay stable."""

    maps: dict[str, dict[Any, int]] = {}
    for target, metadata in target_metadata.items():
        if metadata.get("internal_task") == "regression":
            continue
        classes = list(metadata.get("classes") or [])
        maps[target] = {value: index for index, value in enumerate(classes)}
    return maps


def tabular_bounds(encoded_features: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return column-keyed bounds for TabularDataConfig."""

    feature_columns = list(encoded_features["feature_columns"])
    lower, upper = encoded_features["bounds"]
    return {
        column: (float(lower[index]), float(upper[index]))
        for index, column in enumerate(feature_columns)
    }


def categorical_feature_columns(encoded_features: dict[str, Any]) -> list[str]:
    """Resolve encoded categorical positions back to feature names."""

    columns = list(encoded_features["feature_columns"])
    return [columns[int(index)] for index in encoded_features.get("cat_dims", [])]


def categorical_target_columns(
    target_metadata: dict[str, dict[str, Any]],
) -> list[str]:
    """Return targets that require tabular class/rank encoding."""

    return [
        target
        for target, metadata in target_metadata.items()
        if metadata.get("internal_task") != "regression"
    ]


def _mutable_category_frame(
    data: Any,
    *,
    categorical_columns: list[str],
) -> Any:
    """Cast extension string/category columns before replacing labels with codes."""

    import pandas as pd

    frame = data.copy()
    for column in categorical_columns:
        series = frame.loc[:, column]
        if pd.api.types.is_string_dtype(series.dtype) or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            frame[column] = series.astype(object)
    return frame


def fit_tabular_optimizer(
    *,
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    encoded_features: dict[str, Any],
    target_metadata: dict[str, dict[str, Any]],
    model_config: Any,
    fit_config: Any,
    cross_validation: bool = False,
    cv_config: dict[str, Any] | None = None,
) -> Any:
    """Fit or reuse the public pandas-friendly optimizer for the Web workflow."""

    from .logging import current_request_id
    from .model_reuse import (
        current_model_reuse_state,
        register_fitted_model,
        reuse_fitted_tabular_optimizer,
    )

    run_id = current_request_id()
    reuse_state = current_model_reuse_state()
    source_run_id = str((reuse_state or {}).get("source_run_id") or "")
    if source_run_id:
        if not run_id:
            raise RuntimeError("Model reuse requires an active Web request identifier.")
        return reuse_fitted_tabular_optimizer(
            source_run_id=source_run_id,
            current_run_id=run_id,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
        )

    from bochan.tabular import TabularBayesianOptimizer

    categorical_features = categorical_feature_columns(encoded_features)
    categorical_targets = categorical_target_columns(target_metadata)
    fit_data = _mutable_category_frame(
        data,
        categorical_columns=[*categorical_features, *categorical_targets],
    )
    optimizer = TabularBayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        input_cols=feature_columns,
        target_cols=target_columns,
        categorical_cols=categorical_features,
        target_categorical_cols=categorical_targets,
        bounds=tabular_bounds(encoded_features),
        category_maps=feature_category_maps(data, encoded_features),
        target_category_maps=target_category_maps(target_metadata),
        encode_categories=True,
        return_original_categories=True,
        dropna=False,
        cross_validation=cross_validation,
        cv_config=cv_config,
    )
    optimizer.fit(fit_data)
    if optimizer.dataset is None:
        raise RuntimeError("TabularBayesianOptimizer did not retain its fitted dataset.")

    cross_validation_result = optimizer.cross_validation_result_
    if cross_validation_result is not None:
        feature_importance = getattr(
            cross_validation_result,
            "feature_importance",
            None,
        )
        if feature_importance is not None:
            relabel_feature_importance_outputs(
                feature_importance,
                target_columns,
            )

    from .visualization_sessions import attach_fitted_tabular_optimizer

    if run_id:
        attach_fitted_tabular_optimizer(
            run_id,
            tabular_optimizer=optimizer,
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=str(getattr(model_config, "task_type", "")) == "hybrid",
        )
        register_fitted_model(run_id)
    return optimizer


def encoded_features_from_tabular(
    optimizer: Any,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Expose tabular dataset metadata in the legacy Web response shape."""

    dataset = optimizer.dataset
    if dataset is None:
        raise RuntimeError("TabularBayesianOptimizer is not fitted.")
    metadata = dict(fallback)
    metadata["feature_columns"] = list(dataset.feature_names)
    metadata["cat_dims"] = list(dataset.cat_dims)
    metadata["category_maps"] = dict(dataset.category_maps or {})
    metadata["inverse_category_maps"] = dict(dataset.inverse_category_maps or {})
    if dataset.bounds is not None:
        metadata["bounds"] = dataset.bounds.detach().cpu().tolist()
    return metadata


__all__ = [
    "categorical_feature_columns",
    "categorical_target_columns",
    "encoded_features_from_tabular",
    "feature_category_maps",
    "fit_tabular_optimizer",
    "tabular_bounds",
    "target_category_maps",
]
