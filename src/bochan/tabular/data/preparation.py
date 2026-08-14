"""Public DataFrame preparation helpers for the tabular API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import ColumnKey, TabularDataConfig
from .columns import _as_list
from .conversion import _apply_missing_value_strategy, _pandas


def prepare_dataframe_missing_values(
    data: Any,
    config: TabularDataConfig,
    *,
    input_cols: Sequence[ColumnKey] | None = None,
    target_cols: Sequence[ColumnKey] | None = None,
) -> tuple[Any, dict[ColumnKey, Any], dict[ColumnKey, Any]]:
    """Apply configured missing-value handling while preserving extra columns.

    This is the public DataFrame preparation boundary for callers that need the
    same missing-value semantics as tensor conversion without categorical
    encoding or tensor construction.
    """

    pd = _pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "prepare_dataframe_missing_values expects a pandas.DataFrame."
        )

    resolved_targets = (
        _as_list(config.target_cols)
        if target_cols is None
        else list(target_cols)
    )
    if input_cols is None:
        resolved_inputs = (
            [column for column in data.columns if column not in resolved_targets]
            if config.input_cols is None
            else _as_list(config.input_cols)
        )
    else:
        resolved_inputs = list(input_cols)
    if not resolved_inputs:
        raise ValueError(
            "input_cols could not be inferred. Pass input_cols or "
            "TabularDataConfig.input_cols."
        )

    selected = list(dict.fromkeys(resolved_inputs + resolved_targets))
    missing_columns = [column for column in selected if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Unknown tabular columns: {missing_columns!r}.")

    prepared, impute_values, target_impute_values = _apply_missing_value_strategy(
        work=data.loc[:, selected].copy(),
        input_cols=resolved_inputs,
        target_cols=resolved_targets,
        config=config,
        pd=pd,
    )
    result = data.loc[prepared.index].copy()
    result.loc[:, selected] = prepared.loc[:, selected]
    return result, impute_values, target_impute_values


__all__ = ["prepare_dataframe_missing_values"]
