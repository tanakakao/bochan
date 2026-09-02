"""Observation-aware tabular conversion without target-value imputation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from bochan.api import ObservationData

from ..config import ColumnKey, TabularDataConfig
from ..data import prepare_dataframe_missing_values
from ..data.columns import _as_list, _to_tensor, bounds_to_tensor, resolve_column_indices
from ..data.conversion import (
    _encode_dataframe_category_columns,
    _infer_string_target_categorical_cols,
    _pandas,
    numpy_to_tensors,
    resolve_dtype,
)
from ..data.dataset import TabularDataset


@dataclass
class ObservationTabularDataset(TabularDataset):
    """Tabular dataset plus explicit experiment observation state."""

    observed_mask: Any | None = None
    failed_mask: Any | None = None
    pending_mask: Any | None = None
    experiment_status_name: ColumnKey | None = None

    def observation_data(self) -> ObservationData:
        if self.Y is None:
            raise ValueError("Target values are required to create ObservationData.")
        return ObservationData(
            X=self.X,
            Y=self.Y,
            Yvar=self.Yvar,
            observed_mask=self.observed_mask,
            failed_mask=self.failed_mask,
            pending_mask=self.pending_mask,
        )


def _target_missing_strategy(config: TabularDataConfig) -> str:
    strategy = str(config.target_missing_strategy).strip().lower()
    if strategy not in {"drop", "keep"}:
        raise ValueError("target_missing_strategy must be 'drop' or 'keep'.")
    if config.impute_targets and strategy == "keep":
        raise ValueError(
            "target_missing_strategy='keep' cannot be combined with impute_targets=True. "
            "Unobserved objectives must remain distinct from imputed values."
        )
    return strategy


def _apply_feature_missing_policy(work, input_cols, config, pd):
    strategy = str(
        config.missing_strategy or ("drop" if config.dropna else "none")
    ).lower()
    if strategy in {"drop", "dropna"}:
        missing = work.loc[:, list(input_cols)].isna().any(axis=1)
        return work.loc[~missing].copy(), {}, {}
    if strategy in {"none", "ignore", "keep"}:
        return work.copy(), {}, {}
    if strategy not in {"impute", "fill"}:
        raise ValueError("missing_strategy must be one of 'drop', 'none', or 'impute'.")

    feature_frame = work.loc[:, list(input_cols)].copy()
    prepared, impute_values, _ = prepare_dataframe_missing_values(
        feature_frame,
        config,
        input_cols=input_cols,
        target_cols=[],
    )
    updated = work.loc[prepared.index].copy()
    updated.loc[:, list(input_cols)] = prepared.loc[:, list(input_cols)]
    return updated, impute_values, {}


def _parse_status(work, status_col: ColumnKey | None):
    import numpy as np

    if status_col is None:
        n_rows = len(work)
        return np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=bool)
    if status_col not in work.columns:
        raise KeyError(f"Unknown experiment_status_col={status_col!r}.")
    if work.loc[:, status_col].isna().any():
        raise ValueError("Experiment status must be present for every retained row.")

    status = work.loc[:, status_col].astype(str).str.strip().str.lower()
    valid = {"success", "failed", "pending"}
    invalid = sorted(set(status.unique()) - valid)
    if invalid:
        raise ValueError(
            "Experiment status values must be 'success', 'failed', or 'pending'. "
            f"Invalid values: {invalid}."
        )
    return (status == "failed").to_numpy(), (status == "pending").to_numpy()


def _apply_target_missing_policy(work, target_cols, status_col, config):
    strategy = _target_missing_strategy(config)
    if not target_cols or strategy == "keep":
        return work

    if status_col is None:
        keep = ~work.loc[:, list(target_cols)].isna().any(axis=1)
        return work.loc[keep].copy()

    status = work.loc[:, status_col].astype(str).str.strip().str.lower()
    successful = status == "success"
    incomplete = work.loc[:, list(target_cols)].isna().any(axis=1)
    suppress_objectives = successful & incomplete
    if bool(suppress_objectives.any()):
        work = work.copy()
        work.loc[suppress_objectives, list(target_cols)] = float("nan")
    return work


def _encode_target_categories_allow_missing(
    Y_df,
    *,
    target_cols: Sequence[ColumnKey],
    config: TabularDataConfig,
    pd: Any,
) -> tuple[dict[ColumnKey, dict[Any, int]], dict[ColumnKey, dict[int, Any]]]:
    categorical = _as_list(config.target_categorical_cols)
    if config.target_categorical_cols is None:
        categorical = _infer_string_target_categorical_cols(Y_df, target_cols, pd)

    maps: dict[ColumnKey, dict[Any, int]] = {}
    inverse: dict[ColumnKey, dict[int, Any]] = {}
    supplied = dict(config.target_category_maps or {})
    if not config.encode_categories:
        return maps, inverse

    for column in categorical:
        values = Y_df.loc[:, column]
        explicit = supplied.get(column)
        if explicit is None:
            explicit = supplied.get(str(column))
        if explicit is not None:
            mapping = dict(explicit)
        elif pd.api.types.is_numeric_dtype(values):
            continue
        else:
            unique = list(pd.unique(values.dropna()))
            mapping = {value: index for index, value in enumerate(unique)}

        observed = values.notna()
        encoded = values.copy()
        encoded.loc[observed] = values.loc[observed].map(mapping)
        unmapped = observed & encoded.isna()
        if bool(unmapped.any()):
            missing = sorted(set(values.loc[unmapped].tolist()))
            raise ValueError(
                f"Unmapped categorical values in {column!r}: {missing!r}."
            )
        Y_df.loc[:, column] = encoded
        maps[column] = mapping
        inverse[column] = {int(value): key for key, value in mapping.items()}
    return maps, inverse


def dataframe_to_observation_tensors(
    data: Any,
    config: TabularDataConfig,
) -> ObservationTabularDataset:
    """Convert a DataFrame while retaining target and experiment observation state."""

    pd = _pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("dataframe_to_observation_tensors expects a pandas.DataFrame.")

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
    status_col = config.experiment_status_col
    excluded = set(target_cols) | set(target_variance_cols)
    if status_col is not None:
        excluded.add(status_col)
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

    selected = list(
        dict.fromkeys(
            input_cols
            + target_cols
            + target_variance_cols
            + ([status_col] if status_col is not None else [])
        )
    )
    missing_columns = [column for column in selected if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Unknown tabular columns: {missing_columns!r}.")
    work = data.loc[:, selected].copy()
    work, impute_values, target_impute_values = _apply_feature_missing_policy(
        work,
        input_cols,
        config,
        pd,
    )
    work = _apply_target_missing_policy(work, target_cols, status_col, config)
    failed_mask, pending_mask = _parse_status(work, status_col)

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
        target_category_maps, inverse_target_category_maps = (
            _encode_target_categories_allow_missing(
                Y_df,
                target_cols=target_cols,
                config=config,
                pd=pd,
            )
        )

    dtype = resolve_dtype(config.dtype)
    X = _to_tensor(X_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
    Y = None
    Yvar = None
    observed_mask = None
    import torch

    failed_tensor = torch.as_tensor(failed_mask, dtype=torch.bool, device=X.device)
    pending_tensor = torch.as_tensor(pending_mask, dtype=torch.bool, device=X.device)
    if Y_df is not None:
        Y = _to_tensor(Y_df.to_numpy(dtype=float), dtype=dtype, device=config.device)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        observed_mask = torch.isfinite(Y)
        unavailable = failed_tensor | pending_tensor
        if bool(unavailable.any()):
            observed_mask = observed_mask & ~unavailable.unsqueeze(-1)
            Y = torch.where(
                observed_mask,
                Y,
                torch.full_like(Y, float("nan")),
            )

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
        Yvar = _to_tensor(
            Yvar_df.to_numpy(dtype=float),
            dtype=dtype,
            device=config.device,
        )
        if Yvar.ndim == 1:
            Yvar = Yvar.reshape(-1, 1)
        if Y is None or tuple(Yvar.shape) != tuple(Y.shape):
            raise ValueError("Target variance shape must match target shape.")
        invalid_observed = observed_mask & (~torch.isfinite(Yvar) | (Yvar <= 0.0))
        if bool(invalid_observed.any()):
            raise ValueError(
                "Every observed target cell requires a finite, strictly positive target variance."
            )
        Yvar = torch.where(
            observed_mask,
            Yvar,
            torch.full_like(Yvar, float("nan")),
        )

    feature_names = list(input_cols)
    return ObservationTabularDataset(
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
        observed_mask=observed_mask,
        failed_mask=failed_tensor,
        pending_mask=pending_tensor,
        experiment_status_name=status_col,
    )


def numpy_to_observation_tensors(
    X: Any,
    y: Any | None,
    config: TabularDataConfig,
    *,
    feature_names: Sequence[ColumnKey] | None = None,
    target_names: Sequence[ColumnKey] | None = None,
) -> ObservationTabularDataset:
    if config.experiment_status_col is not None:
        raise ValueError("experiment_status_col requires DataFrame input.")
    if config.target_variance_cols is not None:
        raise ValueError(
            "target_variance_cols requires DataFrame input in observation-aware mode."
        )
    dataset = numpy_to_tensors(
        X,
        y,
        config,
        feature_names=feature_names,
        target_names=target_names,
    )
    import torch

    observed = None if dataset.Y is None else torch.isfinite(dataset.Y)
    n_rows = int(dataset.X.shape[0])
    return ObservationTabularDataset(
        **vars(dataset),
        observed_mask=observed,
        failed_mask=torch.zeros(n_rows, dtype=torch.bool, device=dataset.X.device),
        pending_mask=torch.zeros(n_rows, dtype=torch.bool, device=dataset.X.device),
    )


__all__ = [
    "ObservationTabularDataset",
    "dataframe_to_observation_tensors",
    "numpy_to_observation_tensors",
]
