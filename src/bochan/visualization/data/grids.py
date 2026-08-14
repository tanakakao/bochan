"""One- and two-dimensional visualization grid builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..utils import (
    axis_values,
    candidate_result_from,
    decode_values,
    ensure_2d,
    evaluate_acqf_on_points,
    fixed_row_from,
    get_bounds,
    get_train_X,
    infer_feature_cols,
    labels_from,
    to_tensor_like,
)
from .frames import prediction_dataframe

ShowType = Literal["acqf", "pred"]


def get_const_array(
    candidates: np.ndarray,
    value_dict: Mapping[str, Any] | None,
    feature_cols: Sequence[str],
) -> np.ndarray:
    """指定値で固定した基準行を返す。"""

    columns = [column for column in feature_cols if column != "task"]
    array = ensure_2d(candidates)
    if len(columns) < array.shape[1]:
        array = array[:, : len(columns)]
    const_array = array[:1].astype(float, copy=True)
    for key, value in dict(value_dict or {}).items():
        if key not in columns:
            raise ValueError(f"{key!r} is not in feature_cols.")
        const_array[0, columns.index(key)] = value
    return const_array


def create_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    row: np.ndarray,
    feature_cols: Sequence[str],
    select_idx: Sequence[int],
) -> np.ndarray:
    """2D meshgrid と固定行から評価点を作る。"""

    columns = [column for column in feature_cols if column != "task"]
    base = ensure_2d(row)[:, : len(columns)]
    grid = np.repeat(base, repeats=xx.size, axis=0)
    grid[:, list(select_idx)] = np.column_stack([xx.ravel(), yy.ravel()])
    return grid


def grid_1d_plot(
    obj: Any,
    select_col: str,
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """1D 予測曲線用の予測値と軸値を作る。"""

    train_X = get_train_X(obj)
    X_array = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_array.shape[1])
    if select_col not in columns:
        raise ValueError(f"select_col must be one of {columns}.")

    index = columns.index(select_col)
    x = axis_values(
        obj,
        col=select_col,
        col_index=index,
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=get_bounds(obj, train_X),
    )
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = np.repeat(row, repeats=len(x), axis=0)
    grid[:, index] = x
    mean_frame, std_frame = prediction_dataframe(
        obj,
        to_tensor_like(grid, obj),
        target_cols=target_cols,
    )

    mapping = labels_from(obj, select_col)
    if mapping is not None:
        x = np.asarray(decode_values(x.tolist(), mapping), dtype=object)
    return mean_frame, std_frame, x


def grid_2d(
    obj: Any,
    select_cols: Sequence[str],
    target_col: str | None = None,
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    candidate_result: Any | None = None,
    acqf: Any | None = None,
    n: int = 25,
    show_type: ShowType = "acqf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D グリッド上で獲得関数または予測平均を評価する。"""

    if len(select_cols) != 2:
        raise ValueError("select_cols must contain exactly two columns.")
    if show_type == "pred" and target_col is None:
        raise ValueError("target_col is required when show_type='pred'.")

    train_X = get_train_X(obj)
    X_array = ensure_2d(train_X)
    columns = infer_feature_cols(obj, feature_cols, X_array.shape[1])
    indices = [columns.index(column) for column in select_cols]
    bounds = get_bounds(obj, train_X)
    x1 = axis_values(
        obj,
        col=select_cols[0],
        col_index=indices[0],
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=bounds,
    )
    x2 = axis_values(
        obj,
        col=select_cols[1],
        col_index=indices[1],
        feature_cols=columns,
        n=n,
        train_X=train_X,
        bounds=bounds,
    )
    xx, yy = np.meshgrid(x1, x2)
    row = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    grid = create_grid(xx, yy, row, columns, indices)
    grid_tensor = to_tensor_like(grid, obj)

    if show_type == "acqf":
        result = candidate_result or candidate_result_from(obj)
        acquisition = acqf or getattr(result, "acqf", None)
        if acquisition is None:
            raise ValueError(
                "acqf を指定するか、candidate(..., return_result=True) の結果を渡してください。"
            )
        values = evaluate_acqf_on_points(acquisition, grid_tensor)
        surface = np.ravel(values).reshape(len(x2), len(x1))
    else:
        mean_frame, _ = prediction_dataframe(
            obj,
            grid_tensor,
            target_cols=target_cols,
        )
        if target_col not in mean_frame.columns:
            raise ValueError(f"target_col must be one of {list(mean_frame.columns)}.")
        surface = mean_frame[target_col].to_numpy().reshape(len(x2), len(x1))

    for axis_name, values in zip(select_cols, (x1, x2), strict=False):
        mapping = labels_from(obj, axis_name)
        if mapping is None:
            continue
        decoded = np.asarray(decode_values(list(values), mapping), dtype=object)
        if axis_name == select_cols[0]:
            x1 = decoded
        else:
            x2 = decoded
    return surface.reshape(1, len(x2), len(x1)), x1, x2


__all__ = [
    "ShowType",
    "create_grid",
    "get_const_array",
    "grid_1d_plot",
    "grid_2d",
]
