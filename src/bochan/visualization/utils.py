"""Utilities for bochan visualization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

CYCLE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def to_numpy(x: Any) -> np.ndarray:
    """Tensor / ndarray / DataFrame を numpy 配列へ変換する。"""

    if x is None:
        return np.asarray([])
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, pd.DataFrame | pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def ensure_2d(x: Any) -> np.ndarray:
    """入力を n x d の numpy 配列へ整形する。"""

    arr = to_numpy(x)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def to_tensor_like(x: Any, obj: Any) -> Any:
    """obj の dtype / device に合わせて x を torch.Tensor 化する。"""

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("bochan.visualization requires torch for model evaluation.") from exc

    if torch.is_tensor(x):
        return x
    ref = get_train_X(obj)
    dtype = getattr(ref, "dtype", torch.double)
    device = getattr(ref, "device", None)
    return torch.as_tensor(x, dtype=dtype, device=device)


def get_model(obj: Any) -> Any:
    """BayesianOptimizer / ModelBundle / model から model 本体を取り出す。

    ``posterior`` を持つ model wrapper 自身を先に返す。binary classification
    wrapper の ``model`` 属性は latent GP なので、先に unwrap すると
    probability posterior ではなく latent model を参照してしまう。
    """

    if callable(getattr(obj, "probability_posterior", None)) or callable(
        getattr(obj, "posterior", None)
    ):
        return obj
    if hasattr(obj, "bundle") and getattr(obj, "bundle") is not None:
        bundle_model = getattr(obj.bundle, "model", None)
        if bundle_model is not None:
            return bundle_model
    if hasattr(obj, "model") and getattr(obj, "model") is not None:
        return getattr(obj, "model")
    return obj


def get_train_X(obj: Any) -> Any:
    """obj から train_X を推定する。"""

    for candidate in (obj, getattr(obj, "bundle", None), get_model(obj)):
        if candidate is None:
            continue
        if getattr(candidate, "train_X", None) is not None:
            return getattr(candidate, "train_X")
        if hasattr(candidate, "train_inputs"):
            train_inputs = getattr(candidate, "train_inputs")
            return train_inputs[0] if isinstance(train_inputs, tuple) else train_inputs
    raise AttributeError("train_X が見つかりません。optimizer.fit(...) 済みのオブジェクトを渡してください。")


def get_train_Y(obj: Any) -> Any:
    """obj から train_Y を推定する。"""

    for candidate in (obj, getattr(obj, "bundle", None), get_model(obj)):
        if candidate is None:
            continue
        if getattr(candidate, "train_Y", None) is not None:
            return getattr(candidate, "train_Y")
        if getattr(candidate, "train_targets", None) is not None:
            return getattr(candidate, "train_targets")
    raise AttributeError("train_Y が見つかりません。optimizer.fit(...) 済みのオブジェクトを渡してください。")


def get_bounds(obj: Any, train_X: Any | None = None) -> Any:
    """obj / DataContext / train_X から 2 x d bounds を推定する。"""

    for candidate in (obj, getattr(obj, "data_context", None)):
        if candidate is not None and getattr(candidate, "bounds", None) is not None:
            return getattr(candidate, "bounds")
    train_X = get_train_X(obj) if train_X is None else train_X
    arr = ensure_2d(train_X)
    return np.stack([np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)], axis=0)


def infer_feature_cols(obj: Any, feature_cols: Sequence[str] | None = None, n_cols: int | None = None) -> list[str]:
    """説明変数名を推定する。"""

    if feature_cols is not None:
        return list(feature_cols)
    for candidate in (obj, getattr(obj, "bundle", None), getattr(obj, "data_context", None)):
        meta = getattr(candidate, "metadata", None)
        if isinstance(meta, Mapping):
            for key in ("feature_cols", "features", "x_cols"):
                if key in meta:
                    return list(meta[key])
    if n_cols is None:
        n_cols = ensure_2d(get_train_X(obj)).shape[1]
    return [f"x{i}" for i in range(int(n_cols))]


def infer_target_cols(obj: Any, target_cols: Sequence[str] | None = None, n_cols: int | None = None) -> list[str]:
    """目的変数名を推定する。"""

    if target_cols is not None:
        return list(target_cols)
    for candidate in (obj, getattr(obj, "bundle", None), getattr(obj, "data_context", None)):
        meta = getattr(candidate, "metadata", None)
        if isinstance(meta, Mapping):
            for key in ("target_cols", "targets", "y_cols", "output_names"):
                if key in meta:
                    return list(meta[key])
    if n_cols is None:
        n_cols = ensure_2d(get_train_Y(obj)).shape[1]
    return ["y"] if int(n_cols) == 1 else [f"y{i}" for i in range(int(n_cols))]


def categorical_dims_from(obj: Any) -> list[int]:
    """bochan の cat_dims を取り出す。"""

    for candidate in (obj, getattr(obj, "bundle", None)):
        if candidate is not None and getattr(candidate, "cat_dims", None) is not None:
            return list(getattr(candidate, "cat_dims"))
    config = getattr(obj, "model_config", None)
    if config is not None and getattr(config, "cat_dims", None) is not None:
        return list(getattr(config, "cat_dims"))
    return []


def labels_from(obj: Any, col: str) -> Mapping[Any, Any] | None:
    """カテゴリ label mapping を metadata などから取り出す。"""

    for candidate in (obj, getattr(obj, "bundle", None), get_model(obj)):
        if candidate is None:
            continue
        labels = getattr(candidate, "labels", None)
        if isinstance(labels, Mapping) and col in labels:
            return labels[col]
        meta = getattr(candidate, "metadata", None)
        if isinstance(meta, Mapping):
            labels = meta.get("labels")
            if isinstance(labels, Mapping) and col in labels:
                return labels[col]
    return None


def decode_values(values: Sequence[Any], mapping: Mapping[Any, Any] | None) -> list[Any]:
    """整数カテゴリを元ラベルへ戻す。"""

    if mapping is None:
        return list(values)
    inv = {v: k for k, v in mapping.items()}
    return [inv.get(v, v) for v in values]


def candidate_result_from(obj: Any) -> Any | None:
    """BayesianOptimizer / BochanStudy から直近の CandidateResult を取り出す。"""

    if hasattr(obj, "last_candidate_batch") and getattr(obj, "last_candidate_batch") is not None:
        return getattr(obj.last_candidate_batch, "result", None)
    history = getattr(obj, "history", None)
    if history:
        return history[-1]
    return None


def prediction_mean_std(obj: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """予測平均と標準偏差を配列で返す内部 helper。

    binary classification model が ``probability_posterior`` を提供する場合は
    それを最優先する。これにより visualization は latent ``f(x)`` ではなく、
    model likelihood と整合した ``p(y=1 | x)`` を表示する。
    """

    X_t = to_tensor_like(X, obj)
    model = get_model(obj)
    probability_posterior = getattr(model, "probability_posterior", None)

    if callable(probability_posterior):
        posterior = probability_posterior(X_t)
        mean, var = posterior.mean, posterior.variance
    elif hasattr(obj, "predict"):
        try:
            mean, var = obj.predict(X_t, return_type="mean_variance")
        except TypeError:
            posterior = model.posterior(X_t)
            mean, var = posterior.mean, posterior.variance
    else:
        posterior = model.posterior(X_t)
        mean, var = posterior.mean, posterior.variance

    mean_arr = ensure_2d(mean)
    std_arr = np.sqrt(np.clip(ensure_2d(var), 0.0, None))
    return mean_arr, std_arr


def evaluate_acqf_on_points(acqf: Any, X: Any) -> np.ndarray:
    """n x d の点集合で q=1 の獲得関数値を評価する。"""

    try:
        import torch

        X_eval = X.unsqueeze(-2) if torch.is_tensor(X) and X.ndim == 2 else X
        with torch.no_grad():
            return to_numpy(acqf(X_eval))
    except Exception:
        X_arr = ensure_2d(X)
        return np.asarray([float(acqf(row.reshape(1, 1, -1))) for row in X_arr])


def fixed_row_from(
    obj: Any,
    *,
    feature_cols: Sequence[str],
    value_dict: Mapping[str, Any] | None = None,
    reference_x: Any | None = None,
) -> np.ndarray:
    """グリッド評価で固定する基準行を作る。"""

    if reference_x is not None:
        row = ensure_2d(reference_x)[:1].astype(float, copy=True)
    else:
        row = ensure_2d(get_train_X(obj))[:1].astype(float, copy=True)
    for key, value in dict(value_dict or {}).items():
        if key not in feature_cols:
            raise ValueError(f"value_dict のキー {key!r} は feature_cols に存在しません。")
        row[0, list(feature_cols).index(key)] = value
    return row


def axis_values(
    obj: Any,
    *,
    col: str,
    col_index: int,
    feature_cols: Sequence[str],
    n: int,
    train_X: Any | None = None,
    bounds: Any | None = None,
) -> np.ndarray:
    """連続・カテゴリ列の軸値を作る。"""

    cat_dims = categorical_dims_from(obj)
    train_arr = ensure_2d(get_train_X(obj) if train_X is None else train_X)
    if col in feature_cols and col_index in cat_dims:
        return np.unique(train_arr[:, col_index])
    if bounds is not None:
        b = ensure_2d(bounds)
        lo, hi = float(b[0, col_index]), float(b[1, col_index])
    else:
        lo, hi = float(np.nanmin(train_arr[:, col_index])), float(np.nanmax(train_arr[:, col_index]))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, int(n))


def cycle_series(cycle: str | Sequence[Any] | pd.Series | None, *, X: pd.DataFrame | None = None, y: pd.DataFrame | None = None, length: int | None = None) -> pd.Series | None:
    """cycle 指定を Series に正規化する。"""

    if cycle is None:
        return None
    if isinstance(cycle, str):
        if y is not None and cycle in y.columns:
            raw = y[cycle]
        elif X is not None and cycle in X.columns:
            raw = X[cycle]
        else:
            raise ValueError(f"cycle 列 {cycle!r} が X/y に存在しません。")
    else:
        raw = cycle
    s = pd.Series(raw, name="cycle").reset_index(drop=True)
    if length is not None and len(s) != length:
        raise ValueError(f"cycle の長さが一致しません。expected={length}, got={len(s)}")
    return s


def cycle_color_map(cycle: pd.Series | None) -> dict[Any, str]:
    """cycle 値から表示色への辞書を作る。"""

    if cycle is None:
        return {}
    vals = sorted(pd.unique(cycle.dropna()))
    return {v: CYCLE_COLORS[i % len(CYCLE_COLORS)] for i, v in enumerate(vals)}
