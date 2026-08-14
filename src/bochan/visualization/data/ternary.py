"""Ternary visualization grid builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..utils import (
    candidate_result_from,
    ensure_2d,
    evaluate_acqf_on_points,
    fixed_row_from,
    get_train_X,
    get_train_Y,
    infer_feature_cols,
    infer_target_cols,
    to_numpy,
    to_tensor_like,
)
from .frames import prediction_dataframe
from .grids import ShowType


def tri_grid(
    obj: Any,
    select_cols: Sequence[str],
    target_col: str | None = None,
    value_dict: Mapping[str, Any] | None = None,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    candidate_result: Any | None = None,
    acqf: Any | None = None,
    sum_value: float | None = None,
    n: int = 50,
    show_type: ShowType = "acqf",
) -> tuple[np.ndarray, np.ndarray]:
    """3成分制約を満たす三角グリッド上で獲得関数または予測値を評価する。"""

    if len(select_cols) != 3:
        raise ValueError("select_cols must contain exactly three columns.")
    if show_type == "pred" and target_col is None:
        raise ValueError("予測値ヒートマップを表示するにはtarget_colを指定してください。")

    train_X = get_train_X(obj)
    X_array = ensure_2d(train_X)
    columns = [
        column
        for column in infer_feature_cols(obj, feature_cols, X_array.shape[1])
        if column != "task"
    ]
    selected_indices = [columns.index(column) for column in select_cols]

    model_targets = infer_target_cols(
        obj,
        target_cols,
        ensure_2d(get_train_Y(obj)).shape[1],
    )
    if target_col is not None and target_col not in model_targets:
        raise ValueError(f"target_col must be one of {model_targets}.")

    const_array = fixed_row_from(obj, feature_cols=columns, value_dict=value_dict)
    resolved_sum = _resolve_sum_value(
        obj,
        selected_indices=selected_indices,
        sum_value=sum_value,
    )
    grid_values = _simplex_grid(resolved_sum, n=n)

    grid_array = np.ones((grid_values.shape[0], len(columns))) * const_array
    grid_array[:, selected_indices] = grid_values
    grid_tensor = to_tensor_like(grid_array, obj)

    if show_type == "acqf":
        result = candidate_result or candidate_result_from(obj)
        acquisition = acqf or (
            getattr(result, "acqf", None) if result is not None else None
        )
        if acquisition is None:
            raise ValueError(
                "acqf を指定するか、candidate(..., return_result=True) の結果を渡してください。"
            )
        values = np.ravel(evaluate_acqf_on_points(acquisition, grid_tensor))
    else:
        mean_frame, _ = prediction_dataframe(
            obj,
            grid_tensor,
            target_cols=target_cols,
        )
        if target_col not in mean_frame.columns:
            raise ValueError(f"target_col must be one of {list(mean_frame.columns)}.")
        values = mean_frame[target_col].to_numpy()

    if values.size != len(grid_values):
        raise ValueError(
            "評価値数がグリッド点数と一致しません: "
            f"values={values.size}, grid={len(grid_values)}"
        )

    normalized = grid_values.T
    denominator = normalized.sum(axis=0)
    denominator[denominator == 0] = 1.0
    return values, normalized / denominator


def _resolve_sum_value(
    obj: Any,
    *,
    selected_indices: Sequence[int],
    sum_value: float | None,
) -> float:
    if sum_value is not None:
        return float(sum_value)

    constraint_indices = getattr(obj, "constraint_idx", None)
    constraint_values = getattr(obj, "constraint_values", None)
    if constraint_indices is None or constraint_values is None:
        raise ValueError(
            "三角グリッドの合計値を決める constraint_idx / constraint_values が見つかりません。"
        )

    constraints = list(constraint_indices)
    matches = np.where(
        [all(index in constraint for index in selected_indices) for constraint in constraints]
    )[0]
    if len(matches) == 0:
        raise ValueError("select_cols の3列を含む制約が constraint_idx に見つかりません。")
    values = np.ravel(to_numpy(constraint_values)).astype(float)
    return float(values[int(matches[0])])


def _simplex_grid(sum_value: float, *, n: int) -> np.ndarray:
    axis = np.linspace(0.0, sum_value, int(n))
    xx, yy = np.meshgrid(axis, axis)
    valid = xx.ravel() + yy.ravel() <= sum_value
    first = xx.ravel()[valid]
    second = yy.ravel()[valid]
    third = np.maximum(sum_value - first - second, 0.0)
    return np.column_stack([first, second, third])


__all__ = ["tri_grid"]
