"""DataFrame / numpy conversion for the tabular API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..config import ColumnKey, TabularDataConfig
from .columns import _as_list, _to_tensor, bounds_to_tensor, resolve_column_indices
from .dataset import TabularDataset


def _torch():
    import torch

    return torch


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for DataFrame / CSV tabular APIs.") from exc
    return pd


def _numpy():
    import numpy as np

    return np


def resolve_dtype(dtype: Any | None) -> Any:
    """Resolve a user dtype to a torch dtype."""

    torch = _torch()
    if dtype is None:
        return torch.double
    if isinstance(dtype, str):
        name = dtype.replace("torch.", "")
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype name: {dtype!r}.")
        return getattr(torch, name)
    return dtype


def _mode_value(series, pd: Any) -> Any:
    mode = series.dropna().mode()
    if len(mode) == 0:
        return "__missing__" if not pd.api.types.is_numeric_dtype(series) else 0.0
    return mode.iloc[0]


def _resolve_missing_strategy(config: TabularDataConfig) -> str:
    if config.missing_strategy is not None:
        return str(config.missing_strategy).lower()
    return "drop" if config.dropna else "none"


def _continuous_columns(
    df,
    columns: Sequence[ColumnKey],
    categorical_cols: Sequence[ColumnKey],
    pd: Any,
) -> list[ColumnKey]:
    categorical = set(_as_list(categorical_cols))
    return [
        column
        for column in columns
        if column not in categorical and pd.api.types.is_numeric_dtype(df.loc[:, column])
    ]


def _categorical_or_object_columns(
    df,
    columns: Sequence[ColumnKey],
    categorical_cols: Sequence[ColumnKey],
    pd: Any,
) -> list[ColumnKey]:
    explicit = set(_as_list(categorical_cols))
    return [
        column
        for column in columns
        if column in explicit or not pd.api.types.is_numeric_dtype(df.loc[:, column])
    ]


def _impute_continuous_mean(df, columns: Sequence[ColumnKey]) -> dict[ColumnKey, Any]:
    impute_values: dict[ColumnKey, Any] = {}
    for column in columns:
        value = df.loc[:, column].mean()
        if value != value:
            value = 0.0
        df.loc[:, column] = df.loc[:, column].fillna(value)
        impute_values[column] = float(value)
    return impute_values


def _impute_continuous_iterative(
    df,
    columns: Sequence[ColumnKey],
    *,
    random_state: int | None,
    max_iter: int,
    sample_posterior: bool,
) -> dict[ColumnKey, Any]:
    if not columns:
        return {}
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except ImportError as exc:
        raise ImportError(
            "continuous_impute_strategy='iterative' requires scikit-learn. "
            "Install scikit-learn or use continuous_impute_strategy='mean'."
        ) from exc

    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=sample_posterior,
    )
    imputed = imputer.fit_transform(df.loc[:, list(columns)])
    df.loc[:, list(columns)] = imputed
    return {column: float(df.loc[:, column].mean()) for column in columns}


def _impute_categorical_mode(
    df,
    columns: Sequence[ColumnKey],
    pd: Any,
) -> dict[ColumnKey, Any]:
    impute_values: dict[ColumnKey, Any] = {}
    for column in columns:
        value = _mode_value(df.loc[:, column], pd)
        df.loc[:, column] = df.loc[:, column].fillna(value)
        impute_values[column] = value
    return impute_values


def _apply_missing_value_strategy(
    *,
    work,
    input_cols: Sequence[ColumnKey],
    target_cols: Sequence[ColumnKey],
    config: TabularDataConfig,
    pd: Any,
) -> tuple[Any, dict[ColumnKey, Any], dict[ColumnKey, Any]]:
    """Handle missing values before categorical encoding and tensor conversion."""

    strategy = _resolve_missing_strategy(config)
    if strategy in {"drop", "dropna"}:
        return work.dropna(axis=0, how="any"), {}, {}
    if strategy in {"none", "ignore", "keep"}:
        return work, {}, {}
    if strategy not in {"impute", "fill"}:
        raise ValueError("missing_strategy must be one of 'drop', 'none', or 'impute'.")

    work = work.copy()
    input_categorical_cols = _categorical_or_object_columns(
        work,
        input_cols,
        config.categorical_cols,
        pd,
    )
    input_continuous_cols = _continuous_columns(
        work,
        input_cols,
        input_categorical_cols,
        pd,
    )

    impute_values: dict[ColumnKey, Any] = {}
    target_impute_values: dict[ColumnKey, Any] = {}
    continuous_strategy = config.continuous_impute_strategy.lower()
    if continuous_strategy in {"mean", "avg", "average"}:
        impute_values.update(_impute_continuous_mean(work, input_continuous_cols))
    elif continuous_strategy in {"iterative", "multiple", "multiple_imputation"}:
        impute_values.update(
            _impute_continuous_iterative(
                work,
                input_continuous_cols,
                random_state=config.impute_random_state,
                max_iter=config.impute_max_iter,
                sample_posterior=config.multiple_impute_sample_posterior,
            )
        )
    else:
        raise ValueError("continuous_impute_strategy must be 'mean' or 'iterative'.")

    impute_values.update(_impute_categorical_mode(work, input_categorical_cols, pd))

    if target_cols:
        if config.impute_targets:
            target_categorical_cols = _categorical_or_object_columns(
                work,
                target_cols,
                config.target_categorical_cols or [],
                pd,
            )
            target_continuous_cols = _continuous_columns(
                work,
                target_cols,
                target_categorical_cols,
                pd,
            )
            if continuous_strategy in {"mean", "avg", "average"}:
                target_impute_values.update(
                    _impute_continuous_mean(work, target_continuous_cols)
                )
            else:
                target_impute_values.update(
                    _impute_continuous_iterative(
                        work,
                        target_continuous_cols,
                        random_state=config.impute_random_state,
                        max_iter=config.impute_max_iter,
                        sample_posterior=config.multiple_impute_sample_posterior,
                    )
                )
            target_impute_values.update(
                _impute_categorical_mode(work, target_categorical_cols, pd)
            )
        else:
            work = work.dropna(axis=0, subset=list(target_cols), how="any")

    return work, impute_values, target_impute_values


def _encode_dataframe_category_columns(
    df,
    *,
    columns: Sequence[ColumnKey],
    supplied_maps: Mapping[ColumnKey, Mapping[Any, int]] | None,
    encode_categories: bool,
    pd: Any,
    require_existing: Sequence[ColumnKey],
) -> tuple[dict[ColumnKey, dict[Any, int]], dict[ColumnKey, dict[int, Any]]]:
    """Label-encode string/object category columns in-place and return maps."""

    category_maps: dict[ColumnKey, dict[Any, int]] = {}
    inverse_maps: dict[ColumnKey, dict[int, Any]] = {}
    supplied = dict(supplied_maps or {})
    if not encode_categories:
        return category_maps, inverse_maps

    for column in _as_list(columns):
        if column not in require_existing:
            raise KeyError(
                f"Categorical column {column!r} is not in columns {list(require_existing)!r}."
            )
        values = df.loc[:, column]
        explicit_map = supplied.get(column) or supplied.get(str(column))
        if explicit_map is not None:
            mapping = dict(explicit_map)
        elif pd.api.types.is_numeric_dtype(values):
            continue
        else:
            mapping = {value: index for index, value in enumerate(pd.unique(values.dropna()))}

        encoded = values.map(mapping)
        if encoded.isna().any():
            missing = sorted(set(values[encoded.isna()].tolist()))
            raise ValueError(f"Unmapped categorical values in {column!r}: {missing!r}.")
        # Replace the whole column so pandas may change extension dtypes such as
        # StringDtype to the numeric dtype required by tensor conversion.
        df[column] = encoded
        category_maps[column] = mapping
        inverse_maps[column] = {int(value): key for key, value in mapping.items()}
    return category_maps, inverse_maps


def _infer_string_target_categorical_cols(
    Y_df,
    target_cols: Sequence[ColumnKey],
    pd: Any,
) -> list[ColumnKey]:
    return [
        column
        for column in target_cols
        if not pd.api.types.is_numeric_dtype(Y_df.loc[:, column])
    ]


def dataframe_to_tensors(data: Any, config: TabularDataConfig) -> TabularDataset:
    """Convert a pandas DataFrame to tensors and tabular metadata."""

    pd = _pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dataframe_to_tensors expects a pandas.DataFrame.")

    target_cols = _as_list(config.target_cols)
    target_variance_cols = _as_list(config.target_variance_cols)
    if target_variance_cols and not target_cols:
        raise ValueError("target_variance_cols requires target_cols.")
    if target_variance_cols and len(target_variance_cols) != len(target_cols):
        raise ValueError(
            "target_variance_cols must contain exactly one variance column per target column."
        )
    overlap = sorted(set(target_cols).intersection(target_variance_cols), key=str)
    if overlap:
        raise ValueError(
            "target_variance_cols must be distinct from target_cols; "
            f"overlap={overlap!r}."
        )
    excluded = set(target_cols) | set(target_variance_cols)
    input_cols = (
        [column for column in data.columns if column not in excluded]
        if config.input_cols is None
        else _as_list(config.input_cols)
    )
    variance_inputs = sorted(set(input_cols).intersection(target_variance_cols), key=str)
    if variance_inputs:
        raise ValueError(
            "target_variance_cols must not be included in input_cols; "
            f"overlap={variance_inputs!r}."
        )
    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected_cols = list(dict.fromkeys(input_cols + target_cols + target_variance_cols))
    work = data.loc[:, selected_cols].copy()
    work, impute_values, target_impute_values = _apply_missing_value_strategy(
        work=work,
        input_cols=input_cols,
        target_cols=target_cols,
        config=config,
        pd=pd,
    )
    X_df = work.loc[:, input_cols].copy()
    Y_df = work.loc[:, target_cols].copy() if target_cols else None
    Yvar_df = (
        work.loc[:, target_variance_cols].copy()
        if target_variance_cols
        else None
    )

    category_maps, inverse_maps = _encode_dataframe_category_columns(
        X_df,
        columns=config.categorical_cols,
        supplied_maps=config.category_maps,
        encode_categories=config.encode_categories,
        pd=pd,
        require_existing=input_cols,
    )

    target_category_maps: dict[ColumnKey, dict[Any, int]] = {}
    inverse_target_category_maps: dict[ColumnKey, dict[int, Any]] = {}
    if Y_df is not None:
        target_categorical_cols = _as_list(config.target_categorical_cols)
        if config.target_categorical_cols is None:
            target_categorical_cols = _infer_string_target_categorical_cols(
                Y_df,
                target_cols,
                pd,
            )
        target_category_maps, inverse_target_category_maps = (
            _encode_dataframe_category_columns(
                Y_df,
                columns=target_categorical_cols,
                supplied_maps=config.target_category_maps,
                encode_categories=config.encode_categories,
                pd=pd,
                require_existing=target_cols,
            )
        )

    dtype = resolve_dtype(config.dtype)
    X = _to_tensor(X_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
    Y = None
    if Y_df is not None:
        Y = _to_tensor(Y_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    Yvar = None
    if Yvar_df is not None:
        non_numeric = [
            column
            for column in target_variance_cols
            if not pd.api.types.is_numeric_dtype(Yvar_df.loc[:, column])
        ]
        if non_numeric:
            raise TypeError(
                "target_variance_cols must be numeric variance columns; "
                f"non_numeric={non_numeric!r}."
            )
        variance_values = Yvar_df.to_numpy(dtype=float)
        np = _numpy()
        if not np.isfinite(variance_values).all():
            raise ValueError("target variance values must be finite.")
        if (variance_values <= 0.0).any():
            raise ValueError("target variance values must be strictly positive.")
        Yvar = _to_tensor(variance_values, dtype=dtype, device=config.device)
        if Yvar.ndim == 1:
            Yvar = Yvar.reshape(-1, 1)
        if Y is None or Yvar.shape != Y.shape:
            raise ValueError(
                "Target variance shape must match target shape after tabular conversion."
            )

    feature_names = list(input_cols)
    return TabularDataset(
        X=X,
        Y=Y,
        Yvar=Yvar,
        feature_names=feature_names,
        target_names=list(target_cols),
        cat_dims=resolve_column_indices(config.categorical_cols, feature_names) or [],
        bounds=bounds_to_tensor(
            config.bounds,
            feature_names,
            dtype=dtype,
            device=config.device,
        ),
        category_maps=category_maps,
        inverse_category_maps=inverse_maps,
        target_category_maps=target_category_maps,
        inverse_target_category_maps=inverse_target_category_maps,
        impute_values=impute_values,
        target_impute_values=target_impute_values,
        source_index=work.index,
    )


def numpy_to_tensors(
    X: Any,
    y: Any | None,
    config: TabularDataConfig | None = None,
    *,
    feature_names: Sequence[ColumnKey] | None = None,
    target_names: Sequence[ColumnKey] | None = None,
) -> TabularDataset:
    """Convert numpy-like arrays to tensors and tabular metadata."""

    np = _numpy()
    config = config or TabularDataConfig()
    X_np = np.asarray(X)
    if X_np.ndim != 2:
        raise ValueError("X must have shape n x d.")

    all_feature_names = (
        list(feature_names)
        if feature_names is not None
        else list(range(X_np.shape[1]))
    )
    input_indices = resolve_column_indices(
        config.input_cols,
        all_feature_names,
        none_means_all=True,
    )
    assert input_indices is not None
    feature_names_out = [all_feature_names[index] for index in input_indices]
    X_np = X_np[:, input_indices]

    dtype = resolve_dtype(config.dtype)
    X_tensor = _to_tensor(X_np.astype(float), dtype=dtype, device=config.device)
    Y_tensor = None
    target_names_out = list(target_names or _as_list(config.target_cols))
    if y is not None:
        y_np = np.asarray(y)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        Y_tensor = _to_tensor(y_np.astype(float), dtype=dtype, device=config.device)

    return TabularDataset(
        X=X_tensor,
        Y=Y_tensor,
        feature_names=feature_names_out,
        target_names=target_names_out,
        cat_dims=resolve_column_indices(config.categorical_cols, feature_names_out) or [],
        bounds=bounds_to_tensor(
            config.bounds,
            feature_names_out,
            dtype=dtype,
            device=config.device,
        ),
        category_maps={},
        inverse_category_maps={},
        target_category_maps={},
        inverse_target_category_maps={},
        impute_values={},
        target_impute_values={},
    )


def tensor_to_dataframe(
    X: Any,
    feature_names: Sequence[ColumnKey],
    *,
    inverse_category_maps: Mapping[ColumnKey, Mapping[int, Any]] | None = None,
    decode_categories: bool = True,
):
    """Convert candidate tensors back to a pandas DataFrame."""

    pd = _pandas()
    torch = _torch()
    array = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    df = pd.DataFrame(array, columns=list(feature_names))
    if decode_categories and inverse_category_maps:
        for column, inverse_map in inverse_category_maps.items():
            if column not in df.columns:
                continue
            codes = df[column].round().astype(int)
            decoded = codes.map(inverse_map)
            df[column] = decoded.where(decoded.notna(), df[column])
    return df


__all__ = [
    "TabularDataset",
    "dataframe_to_tensors",
    "numpy_to_tensors",
    "resolve_dtype",
    "tensor_to_dataframe",
]
