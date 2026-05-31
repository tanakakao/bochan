"""
moon_toy_objectives.py

BoTorch / GPyTorch モデル確認用の toy objective 関数群。

特徴:
    - make_moons 風の単目的関数
    - moon + ellipse の多目的関数
    - continuous 入力と mixed 入力を同じ API で切り替え可能
    - regression / binary classification / ordinal classification 用ターゲットを生成可能

典型例:
    # 単目的・連続入力: X.shape = [n, 2]
    train_Y = evaluate_moon_target(train_X, task="regression", mixed=False, as_column=True)

    # 単目的・mixed 入力: X.shape = [n, 3], 3列目がカテゴリ変数 0/1
    train_Y = evaluate_moon_target(train_X, task="classification", mixed=True, categorical_dim=2, as_column=True)

    # 多目的・連続入力: X.shape = [n, 2], train_Y.shape = [n, 2]
    train_Y = evaluate_multi_objective_target(train_X, task="regression", mixed=False)

    # 多目的・mixed 入力: X.shape = [n, 3], train_Y.shape = [n, 2]
    train_Y = evaluate_multi_objective_target(train_X, task="ordinal", mixed=True, categorical_dim=2)
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import torch
from torch import Tensor


CombineMode = Literal["max", "sum"]
TaskType = Literal["regression", "classification", "binary", "ordinal"]


__all__ = [
    "CombineMode",
    "TaskType",
    "_make_moon_curves",
    "moon_center_high_function",
    "ellipse_center_high_function",
    "multi_objective_moon_ellipse_function",
    "binary_object_func",
    "classification_object_func",
    "object_func",
    "ordinal_object_func",
    "evaluate_moon_target",
    "evaluate_multi_objective_target",
    "evaluate_moon_ellipse_target",
    "make_moon_grid",
]


def _as_tensor(
    x: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """入力を Tensor に変換する。Tensor の場合は dtype/device の指定があるときだけ変換する。"""
    if isinstance(x, Tensor):
        if dtype is not None or device is not None:
            return x.to(
                dtype=dtype if dtype is not None else x.dtype,
                device=device if device is not None else x.device,
            )
        return x

    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.as_tensor(x, dtype=dtype, device=device)


def _validate_x_and_dims(
    X: Tensor,
    *,
    continuous_dims: tuple[int, int],
    mixed: bool,
    categorical_dim: int | None,
) -> None:
    """X の最終次元と continuous / categorical dims の整合性を確認する。"""
    if X.ndim == 0:
        raise ValueError("X must have at least one dimension.")
    if len(continuous_dims) != 2:
        raise ValueError("continuous_dims must contain exactly two indices.")
    if min(continuous_dims) < 0:
        raise ValueError("continuous_dims must be non-negative indices.")
    if X.shape[-1] <= max(continuous_dims):
        raise ValueError(
            f"X.shape[-1] must be larger than max(continuous_dims)={max(continuous_dims)}. "
            f"Got X.shape[-1]={X.shape[-1]}."
        )
    if mixed and categorical_dim is None:
        raise ValueError("categorical_dim must be specified when mixed=True.")
    if mixed and categorical_dim is not None:
        if categorical_dim < 0:
            raise ValueError("categorical_dim must be a non-negative index.")
        if X.shape[-1] <= categorical_dim:
            raise ValueError(
                f"X.shape[-1] must be larger than categorical_dim={categorical_dim} when mixed=True. "
                f"Got X.shape[-1]={X.shape[-1]}."
            )


def _flatten_x(X: Tensor) -> tuple[Tensor, torch.Size]:
    """任意 batch shape の X を [N, d] に変形し、元の output shape = X.shape[:-1] を返す。"""
    output_shape = X.shape[:-1]
    X_flat = X.reshape(-1, X.shape[-1])
    return X_flat, output_shape


def _as_float_tuple(
    value: float | Sequence[float],
    *,
    n: int,
    name: str,
) -> tuple[float, ...]:
    """float または長さ n の Sequence[float] を tuple[float, ...] に正規化する。"""
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(n))

    values = tuple(float(v) for v in value)
    if len(values) != n:
        raise ValueError(f"{name} must be a scalar or a sequence of length {n}.")
    return values


def _is_nested_thresholds(value: object) -> bool:
    """thresholds が目的ごとの threshold list かどうかを簡易判定する。"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) == 0:  # type: ignore[arg-type]
        return False
    first = value[0]  # type: ignore[index]
    return isinstance(first, Sequence) and not isinstance(first, (str, bytes))


def _thresholds_per_objective(
    thresholds: float | Sequence[float] | Sequence[Sequence[float]],
    *,
    n_objectives: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    """
    classification / ordinal 用の閾値指定を目的ごとの tuple に正規化する。

    classification の例:
        0.5 -> ((0.5,), (0.5,))
        [0.5, 0.3] -> ((0.5,), (0.3,))

    ordinal の例:
        [-0.2, 0.2] -> ((-0.2, 0.2), (-0.2, 0.2))
        [[-0.2, 0.2], [0.3, 0.6]] -> ((-0.2, 0.2), (0.3, 0.6))
    """
    if isinstance(thresholds, (int, float)):
        return tuple((float(thresholds),) for _ in range(n_objectives))

    if _is_nested_thresholds(thresholds):
        per_obj = tuple(tuple(float(t) for t in ts) for ts in thresholds)  # type: ignore[arg-type]
        if len(per_obj) != n_objectives:
            raise ValueError(f"{name} must contain {n_objectives} threshold sequences.")
    else:
        values = tuple(float(t) for t in thresholds)  # type: ignore[arg-type]
        if len(values) == n_objectives:
            # classification_thresholds=[0.5, 0.3] のような目的ごとの scalar threshold として扱う。
            per_obj = tuple((v,) for v in values)
        else:
            # ordinal_thresholds=[-0.2, 0.2] のような全目的共通 threshold list として扱う。
            per_obj = tuple(values for _ in range(n_objectives))

    for obj_idx, values in enumerate(per_obj):
        if len(values) == 0:
            raise ValueError(f"{name}[{obj_idx}] must contain at least one threshold.")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError(f"{name}[{obj_idx}] must be strictly increasing.")
    return per_obj


def _apply_mixed_effect(
    y: Tensor,
    X_flat: Tensor,
    *,
    mixed: bool,
    categorical_dim: int | None,
    categorical_weights: float | Sequence[float],
    mixed_biases: float | Sequence[float],
) -> Tensor:
    """mixed=True のとき、カテゴリ列による加算効果を y に加える。"""
    if not mixed:
        return y
    if categorical_dim is None:
        raise ValueError("categorical_dim must be specified when mixed=True.")

    if y.ndim == 1:
        cat = X_flat[:, categorical_dim].to(dtype=y.dtype)
        weight = float(categorical_weights) if isinstance(categorical_weights, (int, float)) else float(categorical_weights[0])
        bias = float(mixed_biases) if isinstance(mixed_biases, (int, float)) else float(mixed_biases[0])
        return y + bias + weight * cat

    n_objectives = y.shape[-1]
    weights = _as_float_tuple(categorical_weights, n=n_objectives, name="categorical_weights")
    biases = _as_float_tuple(mixed_biases, n=n_objectives, name="mixed_biases")

    cat = X_flat[:, categorical_dim].to(dtype=y.dtype).unsqueeze(-1)
    w = y.new_tensor(weights).view(1, n_objectives)
    b = y.new_tensor(biases).view(1, n_objectives)
    return y + b + w * cat


def _make_moon_curves(
    n_points: int = 400,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    moon_y_shift: float = 0.0,
    lower_y_shift: float = -0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    make_moons に近い2本の曲線と、その弧パラメータ t を返す。

    Args:
        n_points: 各 moon 曲線を近似する点数。
        dtype: 返す Tensor の dtype。None の場合は torch.get_default_dtype()。
        device: 返す Tensor の device。
        moon_y_shift: 上側 moon の y 方向シフト。
        lower_y_shift: 下側 moon の y 方向シフト。

    Returns:
        upper: 上側 moon 曲線。shape = [n_points, 2]
        lower: 下側 moon 曲線。shape = [n_points, 2]
        t: 弧パラメータ。shape = [n_points]
    """
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1.")

    if dtype is None:
        dtype = torch.get_default_dtype()

    t = torch.linspace(0.0, math.pi, n_points, dtype=dtype, device=device)

    # 上側の月
    upper = torch.column_stack([torch.cos(t), torch.sin(t) + moon_y_shift])

    # 下側の月
    lower = torch.column_stack([1.0 - torch.cos(t), 1.0 - torch.sin(t) + lower_y_shift])

    return upper, lower, t


def moon_center_high_function(
    X: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    sigma_dist: float = 0.12,
    sigma_arc: float = 0.55,
    n_curve_points: int = 400,
    combine: CombineMode = "max",
    mixed: bool = False,
    continuous_dims: tuple[int, int] = (0, 1),
    categorical_dim: int | None = 2,
    categorical_weight: float = 0.15,
    mixed_bias: float = -0.15,
    moon_y_shift: float = 0.0,
    lower_y_shift: float = -0.5,
) -> Tensor:
    """
    make_moons 風の2本の曲線に対して、曲線近傍かつ弧中央付近で高くなる連続値を返す。

    mixed=False の場合:
        X[..., continuous_dims] の2次元だけを使う。

    mixed=True の場合:
        X[..., continuous_dims] の2次元に基づく moon score に対して、
        mixed_bias + categorical_weight * X[..., categorical_dim] を加える。
        典型的には X[..., 2] が 0/1 のカテゴリ変数。

    Args:
        X: 入力。shape = [..., d]。
        sigma_dist: 曲線からの距離方向の広がり。
        sigma_arc: 弧方向の中央強調の広がり。
        n_curve_points: 曲線近似点数。
        combine: upper/lower moon の寄与のまとめ方。"max" または "sum"。
        mixed: True の場合、カテゴリ列の効果を加える。
        continuous_dims: moon 座標として使う2列の index。
        categorical_dim: mixed=True の場合に使うカテゴリ列の index。
        categorical_weight: カテゴリ列にかける係数。
        mixed_bias: mixed=True の場合に加える切片。
        moon_y_shift: 上側 moon の y 方向シフト。
        lower_y_shift: 下側 moon の y 方向シフト。

    Returns:
        y: 連続値。shape = X.shape[:-1]。
    """
    if combine not in {"max", "sum"}:
        raise ValueError("combine must be 'max' or 'sum'.")
    if sigma_dist <= 0:
        raise ValueError("sigma_dist must be positive.")
    if sigma_arc <= 0:
        raise ValueError("sigma_arc must be positive.")

    X_tensor = _as_tensor(X)
    _validate_x_and_dims(
        X_tensor,
        continuous_dims=continuous_dims,
        mixed=mixed,
        categorical_dim=categorical_dim,
    )

    X_flat, output_shape = _flatten_x(X_tensor)
    X_cont = X_flat[:, list(continuous_dims)]

    upper, lower, t = _make_moon_curves(
        n_points=n_curve_points,
        dtype=X_tensor.dtype,
        device=X_tensor.device,
        moon_y_shift=moon_y_shift,
        lower_y_shift=lower_y_shift,
    )

    # 各点から curve 上の全点までの距離^2: [N, n_curve_points]
    dist2_upper = torch.sum((X_cont[:, None, :] - upper[None, :, :]) ** 2, dim=-1)
    dist2_lower = torch.sum((X_cont[:, None, :] - lower[None, :, :]) ** 2, dim=-1)

    # 最も近い curve 点の index
    idx_upper = torch.argmin(dist2_upper, dim=1)
    idx_lower = torch.argmin(dist2_lower, dim=1)

    # 最短距離^2
    row_idx = torch.arange(X_flat.shape[0], device=X_tensor.device)
    min_dist2_upper = dist2_upper[row_idx, idx_upper]
    min_dist2_lower = dist2_lower[row_idx, idx_lower]

    # 弧の中央ほど高い重み。t = pi / 2 付近が最大。
    center = X_tensor.new_tensor(math.pi / 2.0)
    center_weight_upper = torch.exp(-((t[idx_upper] - center) ** 2) / (2.0 * sigma_arc**2))
    center_weight_lower = torch.exp(-((t[idx_lower] - center) ** 2) / (2.0 * sigma_arc**2))

    # 曲線への近さ
    proximity_upper = torch.exp(-min_dist2_upper / (2.0 * sigma_dist**2))
    proximity_lower = torch.exp(-min_dist2_lower / (2.0 * sigma_dist**2))

    score_upper = proximity_upper * center_weight_upper
    score_lower = -proximity_lower * center_weight_lower

    if combine == "sum":
        y = score_upper + score_lower
    else:
        y = torch.maximum(score_upper, score_lower)

    y = _apply_mixed_effect(
        y,
        X_flat,
        mixed=mixed,
        categorical_dim=categorical_dim,
        categorical_weights=categorical_weight,
        mixed_biases=mixed_bias,
    )
    return y.reshape(output_shape)


def ellipse_center_high_function(
    X: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    a: float = 1.2,
    b: float = 0.7,
    center: tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
    mixed: bool = False,
    continuous_dims: tuple[int, int] = (0, 1),
    categorical_dim: int | None = 2,
    categorical_weight: float = -0.10,
    mixed_bias: float = 0.05,
) -> Tensor:
    """
    指定中心付近ほど高く、等高線が楕円になる関数。

    mixed=True の場合は、連続2次元で計算した ellipse score に対して
    mixed_bias + categorical_weight * X[..., categorical_dim] を加える。

    Args:
        X: 入力。shape = [..., d]。
        a: x1方向の広がり。
        b: x2方向の広がり。
        center: 楕円の中心。
        amplitude: 最大値スケール。
        mixed: True の場合、カテゴリ列の効果を加える。
        continuous_dims: ellipse 座標として使う2列の index。
        categorical_dim: mixed=True の場合に使うカテゴリ列の index。
        categorical_weight: カテゴリ列にかける係数。
        mixed_bias: mixed=True の場合に加える切片。

    Returns:
        y: 連続値。shape = X.shape[:-1]。
    """
    if a <= 0:
        raise ValueError("a must be positive.")
    if b <= 0:
        raise ValueError("b must be positive.")

    X_tensor = _as_tensor(X)
    _validate_x_and_dims(
        X_tensor,
        continuous_dims=continuous_dims,
        mixed=mixed,
        categorical_dim=categorical_dim,
    )

    X_flat, output_shape = _flatten_x(X_tensor)
    X_cont = X_flat[:, list(continuous_dims)]

    cx = X_tensor.new_tensor(center[0])
    cy = X_tensor.new_tensor(center[1])

    z = ((X_cont[:, 0] - cx) / a) ** 2 + ((X_cont[:, 1] - cy) / b) ** 2
    y = amplitude * torch.exp(-0.5 * z)

    y = _apply_mixed_effect(
        y,
        X_flat,
        mixed=mixed,
        categorical_dim=categorical_dim,
        categorical_weights=categorical_weight,
        mixed_biases=mixed_bias,
    )
    return y.reshape(output_shape)


def multi_objective_moon_ellipse_function(
    X: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    sigma_dist: float = 0.12,
    sigma_arc: float = 0.55,
    n_curve_points: int = 400,
    combine: CombineMode = "max",
    ellipse_a: float = 1.2,
    ellipse_b: float = 0.7,
    ellipse_center: tuple[float, float] = (0.0, 0.0),
    ellipse_amplitude: float = 1.0,
    mixed: bool = False,
    continuous_dims: tuple[int, int] = (0, 1),
    categorical_dim: int | None = 2,
    categorical_weights: tuple[float, float] = (0.15, -0.10),
    mixed_biases: tuple[float, float] = (-0.15, 0.05),
    moon_y_shift: float = 0.0,
    lower_y_shift: float = -0.5,
) -> Tensor:
    """
    多目的用 toy 関数。

    objective 1:
        moon_center_high_function

    objective 2:
        ellipse_center_high_function

    mixed=False の場合:
        X[..., continuous_dims] の2次元だけを使う。

    mixed=True の場合:
        各目的に mixed_biases[j] + categorical_weights[j] * X[..., categorical_dim]
        を加える。デフォルトではカテゴリ 0/1 が2目的に少し異なる方向の効果を持つ。

    Args:
        X: 入力。shape = [..., d]。
        sigma_dist: moon objective の曲線距離方向の広がり。
        sigma_arc: moon objective の弧方向の中央強調の広がり。
        n_curve_points: moon 曲線近似点数。
        combine: moon の upper/lower 寄与のまとめ方。"max" または "sum"。
        ellipse_a: ellipse objective の x1方向の広がり。
        ellipse_b: ellipse objective の x2方向の広がり。
        ellipse_center: ellipse objective の中心。
        ellipse_amplitude: ellipse objective の最大値スケール。
        mixed: True の場合、カテゴリ列の効果を加える。
        continuous_dims: 連続2変数として使う列の index。
        categorical_dim: mixed=True の場合に使うカテゴリ列の index。
        categorical_weights: mixed=True の場合の目的ごとのカテゴリ係数。長さ2。
        mixed_biases: mixed=True の場合の目的ごとの切片。長さ2。
        moon_y_shift: 上側 moon の y 方向シフト。
        lower_y_shift: 下側 moon の y 方向シフト。

    Returns:
        Y: 多目的連続値。shape = X.shape[:-1] + [2]。
    """
    X_tensor = _as_tensor(X)
    _validate_x_and_dims(
        X_tensor,
        continuous_dims=continuous_dims,
        mixed=mixed,
        categorical_dim=categorical_dim,
    )

    X_flat, output_shape = _flatten_x(X_tensor)

    # ここでは各単目的関数側の mixed 効果は OFF にし、最後に目的ごとの効果をまとめて加える。
    y1 = moon_center_high_function(
        X_tensor,
        sigma_dist=sigma_dist,
        sigma_arc=sigma_arc,
        n_curve_points=n_curve_points,
        combine=combine,
        mixed=False,
        continuous_dims=continuous_dims,
        moon_y_shift=moon_y_shift,
        lower_y_shift=lower_y_shift,
    ).reshape(-1)

    y2 = ellipse_center_high_function(
        X_tensor,
        a=ellipse_a,
        b=ellipse_b,
        center=ellipse_center,
        amplitude=ellipse_amplitude,
        mixed=False,
        continuous_dims=continuous_dims,
    ).reshape(-1)

    Y = torch.stack([y1, y2], dim=-1)
    Y = _apply_mixed_effect(
        Y,
        X_flat,
        mixed=mixed,
        categorical_dim=categorical_dim,
        categorical_weights=categorical_weights,
        mixed_biases=mixed_biases,
    )

    return Y.reshape(*output_shape, 2)


def binary_object_func(
    y: Tensor | Sequence[float],
    *,
    threshold: float = 0.5,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    連続値 y を 2値分類ラベルに変換する。

    Args:
        y: 連続値。
        threshold: y > threshold を class 1、それ以外を class 0 とする閾値。
        dtype: 出力 dtype。分類モデル用には torch.long 推奨。

    Returns:
        label: 0/1 ラベル。shape は y と同じ。
    """
    y_tensor = _as_tensor(y)
    return (y_tensor > threshold).to(dtype=dtype)


def classification_object_func(
    y: Tensor | Sequence[float],
    *,
    threshold: float = 0.5,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """binary_object_func の別名。"""
    return binary_object_func(y, threshold=threshold, dtype=dtype)


def object_func(
    y: Tensor | Sequence[float],
    thresh: float | None = None,
    *,
    threshold: float = 0.5,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    過去コードとの互換用 alias。

    旧コードの object_func(y, thresh=0.2) と、新しい object_func(y, threshold=0.2) の
    両方を受け付ける。
    """
    if thresh is not None:
        threshold = thresh
    return binary_object_func(y, threshold=threshold, dtype=dtype)


def ordinal_object_func(
    y: Tensor | Sequence[float],
    *,
    thresholds: Sequence[float] = (-0.2, 0.2),
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """
    連続値 y を ordinal ラベルに変換する。

    デフォルトでは以下の3クラス:
        class 0: y <= -0.2
        class 1: -0.2 < y <= 0.2
        class 2: y > 0.2

    Args:
        y: 連続値。
        thresholds: 昇順の閾値列。閾値数 + 1 がクラス数になる。
        dtype: 出力 dtype。分類モデル用には torch.long 推奨。

    Returns:
        label: ordinal ラベル。shape は y と同じ。
    """
    y_tensor = _as_tensor(y)
    thresholds_tuple = tuple(float(v) for v in thresholds)
    if len(thresholds_tuple) == 0:
        raise ValueError("thresholds must contain at least one value.")
    if any(a >= b for a, b in zip(thresholds_tuple, thresholds_tuple[1:])):
        raise ValueError("thresholds must be strictly increasing.")

    result = torch.zeros_like(y_tensor, dtype=dtype)
    for class_idx, threshold in enumerate(thresholds_tuple, start=1):
        result[y_tensor > threshold] = class_idx
    return result


def _multi_output_binary_object_func(
    Y: Tensor,
    *,
    thresholds: float | Sequence[float] = 0.5,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """多目的 Y[..., m] を目的ごとの閾値で 0/1 ラベルに変換する。"""
    n_objectives = Y.shape[-1]
    per_obj = _thresholds_per_objective(thresholds, n_objectives=n_objectives, name="classification_thresholds")
    th = Y.new_tensor([v[0] for v in per_obj]).view(*([1] * (Y.ndim - 1)), n_objectives)
    return (Y > th).to(dtype=dtype)


def _multi_output_ordinal_object_func(
    Y: Tensor,
    *,
    thresholds: Sequence[float] | Sequence[Sequence[float]] = (-0.2, 0.2),
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """多目的 Y[..., m] を目的ごとの ordinal ラベルに変換する。"""
    n_objectives = Y.shape[-1]
    per_obj = _thresholds_per_objective(thresholds, n_objectives=n_objectives, name="ordinal_thresholds")

    result = torch.zeros_like(Y, dtype=dtype)
    for obj_idx, obj_thresholds in enumerate(per_obj):
        y_obj = Y[..., obj_idx]
        for class_idx, threshold in enumerate(obj_thresholds, start=1):
            result[..., obj_idx][y_obj > threshold] = class_idx
    return result


def evaluate_moon_target(
    X: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    task: TaskType = "regression",
    as_column: bool = False,
    classification_threshold: float = 0.2,
    ordinal_thresholds: Sequence[float] = (-0.2, 0.2),
    sigma_dist: float = 0.12,
    sigma_arc: float = 0.55,
    n_curve_points: int = 400,
    combine: CombineMode = "max",
    mixed: bool = False,
    continuous_dims: tuple[int, int] = (0, 1),
    categorical_dim: int | None = 2,
    categorical_weight: float = 0.15,
    mixed_bias: float = -0.15,
    moon_y_shift: float = 0.0,
    lower_y_shift: float = -0.5,
) -> Tensor:
    """
    moon_center_high_function から単目的の regression / classification / ordinal 用ターゲットを生成する。

    Args:
        X: 入力。shape = [..., d]。
        task: "regression", "classification"/"binary", "ordinal" のいずれか。
              指定しない場合は "regression"。
        as_column: True の場合、最後に unsqueeze(-1) して shape = [..., 1] にする。
        classification_threshold: 2値分類化の閾値。
        ordinal_thresholds: ordinal 化の閾値。
        sigma_dist: moon score の距離方向の広がり。
        sigma_arc: moon score の弧方向の広がり。
        n_curve_points: 曲線近似点数。
        combine: upper/lower moon の寄与のまとめ方。"max" または "sum"。
        mixed: True の場合、カテゴリ列の効果を加える。
        continuous_dims: moon 座標として使う2列の index。
        categorical_dim: mixed=True の場合に使うカテゴリ列の index。
        categorical_weight: カテゴリ列にかける係数。
        mixed_bias: mixed=True の場合に加える切片。
        moon_y_shift: 上側 moon の y 方向シフト。
        lower_y_shift: 下側 moon の y 方向シフト。

    Returns:
        target: 指定 task のターゲット。as_column=True なら shape = [..., 1]。
    """
    y = moon_center_high_function(
        X,
        sigma_dist=sigma_dist,
        sigma_arc=sigma_arc,
        n_curve_points=n_curve_points,
        combine=combine,
        mixed=mixed,
        continuous_dims=continuous_dims,
        categorical_dim=categorical_dim,
        categorical_weight=categorical_weight,
        mixed_bias=mixed_bias,
        moon_y_shift=moon_y_shift,
        lower_y_shift=lower_y_shift,
    )

    if task == "regression":
        target = y
    elif task in {"classification", "binary"}:
        target = binary_object_func(y, threshold=classification_threshold)
    elif task == "ordinal":
        target = ordinal_object_func(y, thresholds=ordinal_thresholds)
    else:
        raise ValueError("task must be 'regression', 'classification', 'binary', or 'ordinal'.")

    if as_column:
        target = target.unsqueeze(-1)
    return target


def evaluate_multi_objective_target(
    X: Tensor | Sequence[Sequence[float]] | Sequence[float],
    *,
    task: TaskType = "regression",
    classification_thresholds: float | Sequence[float] = 0.5,
    ordinal_thresholds: Sequence[float] | Sequence[Sequence[float]] = (-0.2, 0.2),
    sigma_dist: float = 0.12,
    sigma_arc: float = 0.55,
    n_curve_points: int = 400,
    combine: CombineMode = "max",
    ellipse_a: float = 1.2,
    ellipse_b: float = 0.7,
    ellipse_center: tuple[float, float] = (0.0, 0.0),
    ellipse_amplitude: float = 1.0,
    mixed: bool = False,
    continuous_dims: tuple[int, int] = (0, 1),
    categorical_dim: int | None = 2,
    categorical_weights: tuple[float, float] = (0.15, -0.10),
    mixed_biases: tuple[float, float] = (-0.15, 0.05),
    moon_y_shift: float = 0.0,
    lower_y_shift: float = -0.5,
) -> Tensor:
    """
    moon + ellipse の多目的関数から regression / classification / ordinal 用ターゲットを生成する。

    Args:
        X: 入力。shape = [..., d]。
        task: "regression", "classification"/"binary", "ordinal" のいずれか。
              指定しない場合は "regression"。
        classification_thresholds: 2値分類化の閾値。
            - scalar: 全目的で同じ閾値。
            - 長さ2の sequence: 目的ごとの閾値。
        ordinal_thresholds: ordinal 化の閾値。
            - sequence[float]: 全目的で同じ ordinal 閾値。
            - sequence[sequence[float]]: 目的ごとの ordinal 閾値。
        sigma_dist: moon objective の曲線距離方向の広がり。
        sigma_arc: moon objective の弧方向の中央強調の広がり。
        n_curve_points: moon 曲線近似点数。
        combine: moon の upper/lower 寄与のまとめ方。"max" または "sum"。
        ellipse_a: ellipse objective の x1方向の広がり。
        ellipse_b: ellipse objective の x2方向の広がり。
        ellipse_center: ellipse objective の中心。
        ellipse_amplitude: ellipse objective の最大値スケール。
        mixed: True の場合、カテゴリ列の効果を加える。
        continuous_dims: 連続2変数として使う列の index。
        categorical_dim: mixed=True の場合に使うカテゴリ列の index。
        categorical_weights: mixed=True の場合の目的ごとのカテゴリ係数。長さ2。
        mixed_biases: mixed=True の場合の目的ごとの切片。長さ2。
        moon_y_shift: 上側 moon の y 方向シフト。
        lower_y_shift: 下側 moon の y 方向シフト。

    Returns:
        target:
            task="regression" の場合、連続値。shape = X.shape[:-1] + [2]。
            task="classification"/"binary" の場合、0/1 ラベル。shape = X.shape[:-1] + [2]。
            task="ordinal" の場合、ordinal ラベル。shape = X.shape[:-1] + [2]。
    """
    Y = multi_objective_moon_ellipse_function(
        X,
        sigma_dist=sigma_dist,
        sigma_arc=sigma_arc,
        n_curve_points=n_curve_points,
        combine=combine,
        ellipse_a=ellipse_a,
        ellipse_b=ellipse_b,
        ellipse_center=ellipse_center,
        ellipse_amplitude=ellipse_amplitude,
        mixed=mixed,
        continuous_dims=continuous_dims,
        categorical_dim=categorical_dim,
        categorical_weights=categorical_weights,
        mixed_biases=mixed_biases,
        moon_y_shift=moon_y_shift,
        lower_y_shift=lower_y_shift,
    )

    if task == "regression":
        return Y
    if task in {"classification", "binary"}:
        return _multi_output_binary_object_func(Y, thresholds=classification_thresholds)
    if task == "ordinal":
        return _multi_output_ordinal_object_func(Y, thresholds=ordinal_thresholds)
    raise ValueError("task must be 'regression', 'classification', 'binary', or 'ordinal'.")


# 分かりやすい別名。実体は evaluate_multi_objective_target と同じ。
evaluate_moon_ellipse_target = evaluate_multi_objective_target


def make_moon_grid(
    *,
    x1_min: float = -1.2,
    x1_max: float = 2.2,
    x2_min: float = -1.2,
    x2_max: float = 1.2,
    n_grid: int = 100,
    mixed: bool = False,
    categorical_value: float = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    可視化・予測確認用の2次元格子を作る。

    Args:
        x1_min: 1次元目の下限。
        x1_max: 1次元目の上限。
        x2_min: 2次元目の下限。
        x2_max: 2次元目の上限。
        n_grid: 各軸の分割数。
        mixed: True の場合、3列目に categorical_value を追加する。
        categorical_value: mixed=True のときに3列目へ入れる値。
        dtype: Tensor dtype。
        device: Tensor device。

    Returns:
        X_grid: mixed=False なら shape = [n_grid * n_grid, 2]。
                mixed=True なら shape = [n_grid * n_grid, 3]。
    """
    if n_grid <= 1:
        raise ValueError("n_grid must be greater than 1.")
    if dtype is None:
        dtype = torch.get_default_dtype()

    x1 = torch.linspace(x1_min, x1_max, n_grid, dtype=dtype, device=device)
    x2 = torch.linspace(x2_min, x2_max, n_grid, dtype=dtype, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing="xy")
    X_grid = torch.stack([xx1.reshape(-1), xx2.reshape(-1)], dim=-1)

    if mixed:
        cat = torch.full(
            (X_grid.shape[0], 1),
            fill_value=float(categorical_value),
            dtype=dtype,
            device=device,
        )
        X_grid = torch.cat([X_grid, cat], dim=-1)

    return X_grid
