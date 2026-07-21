"""Adapters that run the React Web workflow through TabularBayesianOptimizer."""

from __future__ import annotations

from typing import Any


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


def fit_tabular_optimizer(
    *,
    data: Any,
    feature_columns: list[str],
    target_columns: list[str],
    encoded_features: dict[str, Any],
    target_metadata: dict[str, dict[str, Any]],
    model_config: Any,
    fit_config: Any,
) -> Any:
    """Fit the Web workflow using the public pandas-friendly optimizer."""

    from bochan.tabular import TabularBayesianOptimizer

    optimizer = TabularBayesianOptimizer(
        model_config=model_config,
        fit_config=fit_config,
        input_cols=feature_columns,
        target_cols=target_columns,
        categorical_cols=categorical_feature_columns(encoded_features),
        target_categorical_cols=categorical_target_columns(target_metadata),
        bounds=tabular_bounds(encoded_features),
        category_maps=feature_category_maps(data, encoded_features),
        target_category_maps=target_category_maps(target_metadata),
        encode_categories=True,
        return_original_categories=True,
        dropna=False,
    )
    optimizer.fit(data)
    if optimizer.dataset is None:
        raise RuntimeError("TabularBayesianOptimizer did not retain its fitted dataset.")
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
