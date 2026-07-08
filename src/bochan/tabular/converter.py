'''Converters between pandas / numpy tabular data and bochan tensor APIs.'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from bochan.api import CandidateRepairConfig, OptimizeConfig

from .config import ColumnKey, TabularDataConfig


@dataclass
class TabularDataset:
    '''Tensor data plus metadata needed to convert candidates back to tables.'''

    X: Any
    Y: Any | None
    feature_names: list[ColumnKey]
    target_names: list[ColumnKey]
    cat_dims: list[int]
    bounds: Any | None = None
    category_maps: dict[ColumnKey, dict[Any, int]] | None = None
    inverse_category_maps: dict[ColumnKey, dict[int, Any]] | None = None
    target_category_maps: dict[ColumnKey, dict[Any, int]] | None = None
    inverse_target_category_maps: dict[ColumnKey, dict[int, Any]] | None = None
    impute_values: dict[ColumnKey, Any] | None = None
    target_impute_values: dict[ColumnKey, Any] | None = None
    source_index: Any | None = None


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
    '''Resolve a user dtype to a torch dtype.'''

    torch = _torch()
    if dtype is None:
        return torch.double
    if isinstance(dtype, str):
        name = dtype.replace("torch.", "")
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype name: {dtype!r}.")
        return getattr(torch, name)
    return dtype


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _column_mapping(feature_names: Sequence[ColumnKey]) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for i, name in enumerate(feature_names):
        mapping[name] = i
        mapping[str(name)] = i
    return mapping


def resolve_column_indices(
    columns: Sequence[ColumnKey] | ColumnKey | None,
    feature_names: Sequence[ColumnKey],
    *,
    none_means_all: bool = False,
) -> list[int] | None:
    '''Resolve column names / integer positions to positional indices.'''

    if columns is None:
        return list(range(len(feature_names))) if none_means_all else None

    mapping = _column_mapping(feature_names)
    resolved: list[int] = []
    for col in _as_list(columns):
        if isinstance(col, int) and 0 <= col < len(feature_names):
            resolved.append(int(col))
        elif col in mapping:
            resolved.append(mapping[col])
        elif str(col) in mapping:
            resolved.append(mapping[str(col)])
        else:
            raise KeyError(f"Unknown column {col!r}. Available columns: {list(feature_names)!r}.")
    return resolved


def _lookup_mapping_value(mapping: Mapping[Any, Any], key: Any, feature_names: Sequence[ColumnKey]) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    if isinstance(key, int) and 0 <= key < len(feature_names):
        name = feature_names[key]
        if name in mapping:
            return mapping[name]
        if str(name) in mapping:
            return mapping[str(name)]
    raise KeyError(f"No value found for column {key!r}.")


def _to_tensor(data: Any, *, dtype: Any, device: Any | None) -> Any:
    torch = _torch()
    return torch.as_tensor(data, dtype=dtype, device=device)


def bounds_to_tensor(
    bounds: Any | Mapping[ColumnKey, Sequence[float]] | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> Any | None:
    '''Convert mapping / array bounds to a BoTorch-style ``2 x d`` tensor.'''

    if bounds is None:
        return None

    torch = _torch()
    if torch.is_tensor(bounds):
        return bounds.to(device=device, dtype=dtype)

    if isinstance(bounds, Mapping):
        lower: list[float] = []
        upper: list[float] = []
        for name in feature_names:
            value = _lookup_mapping_value(bounds, name, feature_names)
            if len(value) != 2:
                raise ValueError(f"Bounds for {name!r} must have length 2.")
            lower.append(float(value[0]))
            upper.append(float(value[1]))
        return _to_tensor([lower, upper], dtype=dtype, device=device)

    return _to_tensor(bounds, dtype=dtype, device=device)


def _mode_value(series, pd: Any) -> Any:
    mode = series.dropna().mode()
    if len(mode) == 0:
        return "__missing__" if not pd.api.types.is_numeric_dtype(series) else 0.0
    return mode.iloc[0]


def _resolve_missing_strategy(config: TabularDataConfig) -> str:
    if config.missing_strategy is not None:
        return str(config.missing_strategy).lower()
    return "drop" if config.dropna else "none"


def _continuous_columns(df, columns: Sequence[ColumnKey], categorical_cols: Sequence[ColumnKey], pd: Any) -> list[ColumnKey]:
    categorical = set(_as_list(categorical_cols))
    return [col for col in columns if col not in categorical and pd.api.types.is_numeric_dtype(df.loc[:, col])]


def _categorical_or_object_columns(df, columns: Sequence[ColumnKey], categorical_cols: Sequence[ColumnKey], pd: Any) -> list[ColumnKey]:
    explicit = set(_as_list(categorical_cols))
    return [col for col in columns if col in explicit or not pd.api.types.is_numeric_dtype(df.loc[:, col])]


def _impute_continuous_mean(df, columns: Sequence[ColumnKey]) -> dict[ColumnKey, Any]:
    impute_values: dict[ColumnKey, Any] = {}
    for col in columns:
        value = df.loc[:, col].mean()
        if value != value:  # NaN
            value = 0.0
        df.loc[:, col] = df.loc[:, col].fillna(value)
        impute_values[col] = float(value)
    return impute_values


def _impute_continuous_iterative(
    df,
    columns: Sequence[ColumnKey],
    *,
    random_state: int | None,
    max_iter: int,
    sample_posterior: bool,
) -> dict[ColumnKey, Any]:
    '''Impute numeric columns using sklearn IterativeImputer.'''

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

    # IterativeImputer cannot use non-numeric columns, so only numeric columns are passed.
    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=sample_posterior,
    )
    imputed = imputer.fit_transform(df.loc[:, list(columns)])
    df.loc[:, list(columns)] = imputed
    # Store column means after imputation for traceability / simple future fallback.
    return {col: float(df.loc[:, col].mean()) for col in columns}


def _impute_categorical_mode(df, columns: Sequence[ColumnKey], pd: Any) -> dict[ColumnKey, Any]:
    impute_values: dict[ColumnKey, Any] = {}
    for col in columns:
        value = _mode_value(df.loc[:, col], pd)
        df.loc[:, col] = df.loc[:, col].fillna(value)
        impute_values[col] = value
    return impute_values


def _apply_missing_value_strategy(
    *,
    work,
    input_cols: Sequence[ColumnKey],
    target_cols: Sequence[ColumnKey],
    config: TabularDataConfig,
    pd: Any,
) -> tuple[Any, dict[ColumnKey, Any], dict[ColumnKey, Any]]:
    '''Handle missing values before categorical encoding and tensor conversion.'''

    strategy = _resolve_missing_strategy(config)
    if strategy in {"drop", "dropna"}:
        return work.dropna(axis=0, how="any"), {}, {}
    if strategy in {"none", "ignore", "keep"}:
        return work, {}, {}
    if strategy not in {"impute", "fill"}:
        raise ValueError("missing_strategy must be one of 'drop', 'none', or 'impute'.")

    work = work.copy()
    input_categorical_cols = _categorical_or_object_columns(work, input_cols, config.categorical_cols, pd)
    input_continuous_cols = _continuous_columns(work, input_cols, input_categorical_cols, pd)

    impute_values: dict[ColumnKey, Any] = {}
    target_impute_values: dict[ColumnKey, Any] = {}

    if config.continuous_impute_strategy.lower() in {"mean", "avg", "average"}:
        impute_values.update(_impute_continuous_mean(work, input_continuous_cols))
    elif config.continuous_impute_strategy.lower() in {"iterative", "multiple", "multiple_imputation"}:
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
            target_continuous_cols = _continuous_columns(work, target_cols, target_categorical_cols, pd)
            if config.continuous_impute_strategy.lower() in {"mean", "avg", "average"}:
                target_impute_values.update(_impute_continuous_mean(work, target_continuous_cols))
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
            target_impute_values.update(_impute_categorical_mode(work, target_categorical_cols, pd))
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
    '''Label-encode string/object category columns in-place and return maps.'''

    category_maps: dict[ColumnKey, dict[Any, int]] = {}
    inverse_maps: dict[ColumnKey, dict[int, Any]] = {}
    supplied = dict(supplied_maps or {})

    if not encode_categories:
        return category_maps, inverse_maps

    for col in _as_list(columns):
        if col not in require_existing:
            raise KeyError(f"Categorical column {col!r} is not in columns {list(require_existing)!r}.")

        values = df.loc[:, col]
        explicit_map = supplied.get(col) or supplied.get(str(col))
        if explicit_map is not None:
            mapping = dict(explicit_map)
        elif pd.api.types.is_numeric_dtype(values):
            continue
        else:
            uniques = list(pd.unique(values.dropna()))
            mapping = {value: i for i, value in enumerate(uniques)}

        df.loc[:, col] = values.map(mapping)
        if df.loc[:, col].isna().any():
            missing = sorted(set(values[df.loc[:, col].isna()].tolist()))
            raise ValueError(f"Unmapped categorical values in {col!r}: {missing!r}.")

        category_maps[col] = mapping
        inverse_maps[col] = {int(v): k for k, v in mapping.items()}

    return category_maps, inverse_maps


def _infer_string_target_categorical_cols(Y_df, target_cols: Sequence[ColumnKey], pd: Any) -> list[ColumnKey]:
    '''Infer target columns that need label encoding because they are non-numeric.'''

    inferred: list[ColumnKey] = []
    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(Y_df.loc[:, col]):
            inferred.append(col)
    return inferred


def dataframe_to_tensors(data: Any, config: TabularDataConfig) -> TabularDataset:
    '''Convert a pandas DataFrame to tensors and tabular metadata.'''

    pd = _pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dataframe_to_tensors expects a pandas.DataFrame.")

    target_cols = _as_list(config.target_cols)
    if config.input_cols is None:
        input_cols = [col for col in data.columns if col not in target_cols]
    else:
        input_cols = _as_list(config.input_cols)

    if not input_cols:
        raise ValueError("input_cols could not be inferred. Pass TabularDataConfig.input_cols.")

    selected_cols = list(dict.fromkeys(input_cols + target_cols))
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
            target_categorical_cols = _infer_string_target_categorical_cols(Y_df, target_cols, pd)
        target_category_maps, inverse_target_category_maps = _encode_dataframe_category_columns(
            Y_df,
            columns=target_categorical_cols,
            supplied_maps=config.target_category_maps,
            encode_categories=config.encode_categories,
            pd=pd,
            require_existing=target_cols,
        )

    dtype = resolve_dtype(config.dtype)
    X = _to_tensor(X_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
    Y = None
    if Y_df is not None:
        Y = _to_tensor(Y_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

    feature_names = list(input_cols)
    cat_dims = resolve_column_indices(config.categorical_cols, feature_names) or []
    bounds = bounds_to_tensor(config.bounds, feature_names, dtype=dtype, device=config.device)

    return TabularDataset(
        X=X,
        Y=Y,
        feature_names=feature_names,
        target_names=list(target_cols),
        cat_dims=cat_dims,
        bounds=bounds,
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
    '''Convert numpy-like arrays to tensors and tabular metadata.'''

    np = _numpy()
    config = config or TabularDataConfig()
    X_np = np.asarray(X)
    if X_np.ndim != 2:
        raise ValueError("X must have shape n x d.")

    if feature_names is not None:
        all_feature_names = list(feature_names)
    else:
        all_feature_names = list(range(X_np.shape[1]))

    input_indices = resolve_column_indices(config.input_cols, all_feature_names, none_means_all=True)
    assert input_indices is not None
    feature_names_out = [all_feature_names[i] for i in input_indices]

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

    cat_dims = resolve_column_indices(config.categorical_cols, feature_names_out) or []
    bounds = bounds_to_tensor(config.bounds, feature_names_out, dtype=dtype, device=config.device)

    return TabularDataset(
        X=X_tensor,
        Y=Y_tensor,
        feature_names=feature_names_out,
        target_names=target_names_out,
        cat_dims=cat_dims,
        bounds=bounds,
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
    '''Convert candidate tensors back to a pandas DataFrame.'''

    pd = _pandas()
    torch = _torch()

    if torch.is_tensor(X):
        array = X.detach().cpu().numpy()
    else:
        array = X

    df = pd.DataFrame(array, columns=list(feature_names))

    if decode_categories and inverse_category_maps:
        for col, inv_map in inverse_category_maps.items():
            if col not in df.columns:
                continue
            codes = df[col].round().astype(int)
            decoded = codes.map(inv_map)
            df[col] = decoded.where(decoded.notna(), df[col])
    return df


def _resolve_steps(
    steps: Any,
    numeric_columns: Sequence[ColumnKey] | None,
    feature_names: Sequence[ColumnKey],
) -> Any:
    if steps is None:
        return None
    if not isinstance(steps, Mapping):
        return steps

    cols = list(numeric_columns) if numeric_columns is not None else list(feature_names)
    return [_lookup_mapping_value(steps, col, feature_names) for col in cols]


def _resolve_feature_mapping(
    mapping: Mapping[Any, Any] | None,
    feature_names: Sequence[ColumnKey],
) -> dict[int, float] | None:
    if mapping is None:
        return None
    resolved: dict[int, float] = {}
    for key, value in mapping.items():
        idx = resolve_column_indices([key], feature_names)
        assert idx is not None
        resolved[int(idx[0])] = float(value)
    return resolved


def _resolve_fixed_features_list(
    values: Sequence[Mapping[Any, Any]] | None,
    feature_names: Sequence[ColumnKey],
) -> list[dict[int, float]] | None:
    if values is None:
        return None
    return [_resolve_feature_mapping(item, feature_names) or {} for item in values]


def _is_tensor(value: Any) -> bool:
    torch = _torch()
    return bool(torch.is_tensor(value))


def _resolve_constraint_indices(
    indices: Any,
    feature_names: Sequence[ColumnKey],
    *,
    device: Any | None,
) -> Any:
    torch = _torch()
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long).reshape(-1)

    resolved = resolve_column_indices(indices, feature_names)
    return torch.as_tensor(resolved, dtype=torch.long, device=device).reshape(-1)


def _resolve_constraint_coefficients(
    coefficients: Any,
    *,
    dtype: Any,
    device: Any | None,
) -> Any:
    torch = _torch()
    if torch.is_tensor(coefficients):
        return coefficients.to(device=device, dtype=dtype).reshape(-1)
    return torch.as_tensor(coefficients, dtype=dtype, device=device).reshape(-1)


def _resolve_linear_constraints(
    constraints: Any | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> Any | None:
    if constraints is None:
        return None

    resolved = []
    for indices, coefficients, rhs in constraints:
        resolved.append(
            (
                _resolve_constraint_indices(indices, feature_names, device=device),
                _resolve_constraint_coefficients(coefficients, dtype=dtype, device=device),
                rhs,
            )
        )
    return resolved


def _resolve_sum_constraint_indices(
    indices: Any,
    feature_names: Sequence[ColumnKey],
) -> list[int]:
    if _is_tensor(indices):
        return [int(item) for item in indices.detach().cpu().reshape(-1).tolist()]
    resolved = resolve_column_indices(indices, feature_names)
    return [] if resolved is None else [int(item) for item in resolved]


def _resolve_final_sum_constraint(
    value: tuple[Sequence[Any], float] | None,
    feature_names: Sequence[ColumnKey],
) -> tuple[list[int], float] | None:
    if value is None:
        return None

    indices, rhs = value
    return (_resolve_sum_constraint_indices(indices, feature_names), rhs)


def resolve_repair_config_columns(
    repair: CandidateRepairConfig | None,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> CandidateRepairConfig | None:
    '''Resolve column names inside ``CandidateRepairConfig`` to indices.'''

    if repair is None:
        return None

    numeric_cols = _as_list(repair.numeric_indices) if repair.numeric_indices is not None else None
    if numeric_cols is None and isinstance(repair.steps, Mapping):
        numeric_cols = list(repair.steps.keys())
    comp_cols = _as_list(repair.comp_idx) if repair.comp_idx is not None else None

    numeric_indices = resolve_column_indices(numeric_cols, feature_names) if numeric_cols is not None else None
    comp_idx = resolve_column_indices(comp_cols, feature_names) if comp_cols is not None else None
    steps = _resolve_steps(repair.steps, numeric_cols, feature_names)
    bounds = bounds_to_tensor(repair.bounds, feature_names, dtype=dtype, device=device)

    return replace(
        repair,
        bounds=bounds,
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        equality_constraints=_resolve_linear_constraints(
            repair.equality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        inequality_constraints=_resolve_linear_constraints(
            repair.inequality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        fixed_features=_resolve_feature_mapping(repair.fixed_features, feature_names),
        final_sum_constraint=_resolve_final_sum_constraint(repair.final_sum_constraint, feature_names),
    )


def resolve_optimize_config_columns(
    config: OptimizeConfig,
    feature_names: Sequence[ColumnKey],
    *,
    dtype: Any,
    device: Any | None,
) -> OptimizeConfig:
    '''Resolve column names inside ``OptimizeConfig`` to tensor API indices.'''

    repair_config = resolve_repair_config_columns(
        config.repair_config,
        feature_names,
        dtype=dtype,
        device=device,
    )

    return replace(
        config,
        fixed_features=_resolve_feature_mapping(config.fixed_features, feature_names),
        fixed_features_list=_resolve_fixed_features_list(config.fixed_features_list, feature_names),
        equality_constraints=_resolve_linear_constraints(
            config.equality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        inequality_constraints=_resolve_linear_constraints(
            config.inequality_constraints,
            feature_names,
            dtype=dtype,
            device=device,
        ),
        repair_config=repair_config,
    )
