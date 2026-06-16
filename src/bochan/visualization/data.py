"""Data builders for bochan visualization.

Plotly に依存しない、可視化前処理用の関数群です。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .utils import (
    axis_values,
    candidate_result_from,
    decode_values,
    ensure_2d,
    evaluate_acqf_on_points,
    fixed_row_from,
    get_bounds,
    get_train_X,
    get_train_Y,
    infer_feature_cols,
    infer_target_cols,
    labels_from,
    prediction_mean_std,
    to_numpy,
    to_tensor_like,
)

ShowType = Literal["acqf", "pred"]


def prediction_dataframe(obj: Any, X: Any, *, target_cols: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bochan モデルの予測平均・標準偏差を DataFrame で返す。"""

    mean, std = prediction_mean_std(obj, X)
    cols = infer_target_cols(obj, target_cols, mean.shape[1])
    return pd.DataFrame(mean, columns=cols), pd.DataFrame(std, columns=cols)


def training_dataframe(
    obj: Any,
    *,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    X: Any | None = None,
    y: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習 X/Y を DataFrame として返す。"""

    X_arr = ensure_2d(get_train_X(obj) if X is None else X)
    y_arr = ensure_2d(get_train_Y(obj) if y is None else y)
    x_cols = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    y_cols = infer_target_cols(obj, target_cols, y_arr.shape[1])
    return pd.DataFrame(X_arr, columns=x_cols), pd.DataFrame(y_arr, columns=y_cols)


def candidates_dataframe(
    obj: Any,
    *,
    candidate_result: Any | None = None,
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    include_prediction: bool = True,
) -> pd.DataFrame | None:
    """候補点を DataFrame 化し、予測平均・標準偏差・獲得関数値を付ける。"""

    result = candidate_result or candidate_result_from(obj)
    if result is None or getattr(result, "candidates", None) is None:
        return None
    X_arr = ensure_2d(getattr(result, "candidates"))
    x_cols = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    df = pd.DataFrame(X_arr, columns=x_cols)
    if include_prediction:
        mean_df, std_df = prediction_dataframe(obj, getattr(result, "candidates"), target_cols=target_cols)
        for col in mean_df.columns:
            df[f"{col}_mean"] = mean_df[col].to_numpy()
            df[f"{col}_std"] = std_df[col].to_numpy()
    acq_value = getattr(result, "acq_value", None)
    if acq_value is not None:
        v = np.ravel(to_numpy(acq_value))
        if len(v) == len(df):
            df["acq_value"] = v
        elif len(v) == 1:
            df["acq_value"] = float(v[0])
    return df


def get_yyplot_data(obj: Any, *, target_cols: Sequence[str] | None = None, candidate_result: Any | None = None):
    """YY plot 用の ``(cv_result, (pred_mean, pred_std), df_cand)`` を返す。

    旧実装の戻り値に合わせて、CV が無い場合は ``cv_result=None`` とします。
    """

    train_X = get_train_X(obj)
    pred = prediction_dataframe(obj, train_X, target_cols=target_cols)
    df_cand = candidates_dataframe(obj, candidate_result=candidate_result, target_cols=target_cols)
    cv_result = getattr(obj, "cv_results", None)
    return cv_result, pred, df_cand


def get_const_array(candidates: np.ndarray, value_dict: Mapping[str, Any] | None, feature_cols: Sequence[str]) -> np.ndarray:
    """指定された値で固定行を作る。値指定がなければ candidates の先頭行を返す。"""

    cols = [c for c in feature_cols if c != "task"]
    arr = ensure_2d(candidates)
    if len(cols) < arr.shape[1]:
        arr = arr[:, : len(cols)]
    const_array = arr[:1].astype(float, copy=True)
    for key, value in dict(value_dict or {}).items():
        if key not in cols:
            raise ValueError(f"{key!r} is not in feature_cols.")
        const_array[0, cols.index(key)] = value
    return const_array


def create_grid(xx: np.ndarray, yy: np.ndarray, row: np.ndarray, feature_cols: Sequence[str], select_idx: Sequence[int]) -> np.ndarray:
    """2D meshgrid と固定行から n_grid x d の評価点を作る。"""

    cols = [c for c in feature_cols if c != "task"]
    base = ensure_2d(row)[:, : len(cols)]
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
    """1次元の予測曲線用データ ``(pred_mean, pred_std, x)`` を作る。"""

    train_X = get_train_X(obj)
    X_arr = ensure_2d(train_X)
    cols = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    if select_col not in cols:
        raise ValueError(f"select_col must be one of {cols}.")
    idx = cols.index(select_col)
    x = axis_values(obj, col=select_col, col_index=idx, feature_cols=cols, n=n, train_X=train_X, bounds=get_bounds(obj, train_X))
    row = fixed_row_from(obj, feature_cols=cols, value_dict=value_dict)
    grid = np.repeat(row, repeats=len(x), axis=0)
    grid[:, idx] = x
    pred = prediction_dataframe(obj, to_tensor_like(grid, obj), target_cols=target_cols)
    mapping = labels_from(obj, select_col)
    if mapping is not None:
        x = np.asarray(decode_values(x.tolist(), mapping), dtype=object)
    return pred[0], pred[1], x


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
    X_arr = ensure_2d(train_X)
    cols = infer_feature_cols(obj, feature_cols, X_arr.shape[1])
    idx = [cols.index(c) for c in select_cols]
    bounds = get_bounds(obj, train_X)
    x1 = axis_values(obj, col=select_cols[0], col_index=idx[0], feature_cols=cols, n=n, train_X=train_X, bounds=bounds)
    x2 = axis_values(obj, col=select_cols[1], col_index=idx[1], feature_cols=cols, n=n, train_X=train_X, bounds=bounds)
    xx, yy = np.meshgrid(x1, x2)
    row = fixed_row_from(obj, feature_cols=cols, value_dict=value_dict)
    grid = create_grid(xx, yy, row, cols, idx)
    grid_t = to_tensor_like(grid, obj)

    if show_type == "acqf":
        result = candidate_result or candidate_result_from(obj)
        acqf = acqf or getattr(result, "acqf", None)
        if acqf is None:
            raise ValueError("acqf を指定するか、candidate(..., return_result=True) の結果を渡してください。")
        values = evaluate_acqf_on_points(acqf, grid_t)
        Z = np.ravel(values).reshape(len(x2), len(x1))
    else:
        mean_df, _ = prediction_dataframe(obj, grid_t, target_cols=target_cols)
        if target_col not in mean_df.columns:
            raise ValueError(f"target_col must be one of {list(mean_df.columns)}.")
        Z = mean_df[target_col].to_numpy().reshape(len(x2), len(x1))

    for axis_name, values in zip(select_cols, (x1, x2), strict=False):
        mapping = labels_from(obj, axis_name)
        if mapping is not None:
            decoded = np.asarray(decode_values(list(values), mapping), dtype=object)
            if axis_name == select_cols[0]:
                x1 = decoded
            else:
                x2 = decoded
    return Z.reshape(1, len(x2), len(x1)), x1, x2


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
    n: int = 30,
    show_type: ShowType = "acqf",
) -> tuple[np.ndarray, np.ndarray]:
    """3成分制約 x1+x2+x3=max_value の三角グリッド上で評価する。"""

    if len(select_cols) != 3:
        raise ValueError("select_cols must contain exactly three columns.")
    if show_type == "pred" and target_col is None:
        raise ValueError("予測値ヒートマップを表示するにはtarget_colを指定してください。")

    model_feature_cols = list(getattr(obj, "feature_cols", feature_cols or []))
    if not model_feature_cols:
        train_X = get_train_X(obj)
        model_feature_cols = infer_feature_cols(obj, feature_cols, ensure_2d(train_X).shape[1])
    cols = [f for f in model_feature_cols if f != "task"]

    select_idx = [cols.index(col) for col in select_cols]

    model_target_cols = list(getattr(obj, "target_cols", target_cols or []))
    if target_col is not None and model_target_cols and target_col not in model_target_cols:
        raise ValueError(f"target_col must be one of {model_target_cols}.")
    target_idx = model_target_cols.index(target_col) if target_col in model_target_cols else None

    candidates_raw = getattr(obj, "candidates_raw", None)
    if candidates_raw is None:
        result = candidate_result or candidate_result_from(obj)
        candidates_raw = getattr(result, "candidates", None) if result is not None else None
    if candidates_raw is None:
        candidates_raw = get_train_X(obj)
    const_array = get_const_array(candidates_raw, value_dict, cols)

    if sum_value is None:
        eq_cons_idx = getattr(obj, "constraint_idx", None)
        eq_cons_vals = getattr(obj, "constraint_values", None)
        if eq_cons_idx is None or eq_cons_vals is None:
            raise ValueError("三角グリッドの合計値を決める constraint_idx / constraint_values が見つかりません。")

        eq_cons_idx_list = list(eq_cons_idx)
        hit = np.where([all([c in t for c in select_idx]) for t in eq_cons_idx_list])[0]
        if len(hit) == 0:
            raise ValueError("select_cols の3列を含む制約が constraint_idx に見つかりません。")
        cons_pos = int(hit[0])
        _const_idx = eq_cons_idx_list[cons_pos]
        eq_cons_vals_arr = np.ravel(to_numpy(eq_cons_vals)).astype(float)
        max_value = float(eq_cons_vals_arr[cons_pos])
    else:
        max_value = float(sum_value)

    x = np.linspace(0.0, max_value, int(n))
    xx, yy = np.meshgrid(x, x)
    valid_idx = xx.ravel() + yy.ravel() <= max_value
    xx = xx.ravel()[valid_idx].reshape(-1, 1)
    yy = yy.ravel()[valid_idx].reshape(-1, 1)

    zz = max_value - xx - yy
    zz[zz < 0] = 0.0
    grid_values = np.concatenate([xx, yy, zz], axis=1)

    grid_array = np.ones((len(xx), len(cols))) * const_array
    grid_array[:, select_idx] = grid_values

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bochan.visualization requires torch for ternary grid evaluation.") from exc

    train_X = get_train_X(obj)
    dtype = getattr(obj, "dtype", getattr(train_X, "dtype", torch.double))
    device = getattr(train_X, "device", None)
    grid_tensor = torch.tensor(grid_array, dtype=dtype, device=device)
    n_grid = len(grid_values)

    if show_type == "acqf":
        acq_callable = acqf
        if acq_callable is None:
            result = candidate_result or candidate_result_from(obj)
            acq_callable = getattr(result, "acqf", None) if result is not None else None
        if acq_callable is None:
            acq_callable = getattr(obj, "acquisition_function", None)
        if acq_callable is None:
            raise ValueError("acquisition_function が見つかりません。")
        values = np.ravel(evaluate_acqf_on_points(acq_callable, grid_tensor))
        if values.size != n_grid:
            raise ValueError(
                f"acqf の評価値数がグリッド点数と一致しません: values={values.size}, grid={n_grid}"
            )
    else:
        if not hasattr(obj, "predict"):
            mean_df, _ = prediction_dataframe(obj, grid_tensor, target_cols=target_cols)
            if target_col not in mean_df.columns:
                raise ValueError(f"target_col must be one of {list(mean_df.columns)}.")
            values = mean_df[target_col].to_numpy()
        else:
            pred = obj.predict(grid_tensor)[0]
            try:
                values = np.ravel(pred[target_col].to_numpy())
            except Exception:
                pred_arr = ensure_2d(pred)
                if target_idx is None:
                    raise ValueError("target_col の列番号を特定できません。target_cols を指定してください。")
                values = np.ravel(pred_arr[:, target_idx])
        if values.size != n_grid:
            mean_df, _ = prediction_dataframe(obj, grid_tensor, target_cols=target_cols)
            if target_col not in mean_df.columns:
                raise ValueError(f"target_col must be one of {list(mean_df.columns)}.")
            values = mean_df[target_col].to_numpy()

    grid_values = grid_values.T
    denom = grid_values.sum(axis=0)
    denom[denom == 0] = 1.0
    grid_values = grid_values / denom
    return values, grid_values


def study_target_dataframe(study: Any, *, target: str, target_cols: Sequence[str] | None = None, cycle_col: str = "cycle") -> pd.DataFrame:
    """BochanStudy の completed trial から cycle-target DataFrame を作る。"""

    trials = study.completed_trials()
    if not trials:
        return pd.DataFrame(columns=[cycle_col, target])
    Y = ensure_2d([trial.y for trial in trials])
    cols = infer_target_cols(study, target_cols, Y.shape[1])
    if target not in cols:
        raise ValueError(f"target must be one of {cols}.")
    idx = cols.index(target)
    cycles = [trial.metadata.get(cycle_col, i) for i, trial in enumerate(trials)]
    return pd.DataFrame({cycle_col: cycles, target: Y[:, idx]})
