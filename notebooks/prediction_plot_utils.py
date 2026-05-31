from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Sequence

import torch
from torch import Tensor
import matplotlib.pyplot as plt

try:
    from bochan.models.transforms.posterior import (
        aggregate_perturbed_posterior,
        aggregate_perturbed_posterior_chunked,
        aggregate_perturbed_ordinal_expected_utility,
        aggregate_perturbed_ordinal_expected_utility_chunked,
        aggregate_perturbed_ordinal_class_probs,
    )
except Exception:  # pragma: no cover - bochan 外でのimport確認用
    aggregate_perturbed_posterior = None
    aggregate_perturbed_posterior_chunked = None
    aggregate_perturbed_ordinal_expected_utility = None
    aggregate_perturbed_ordinal_expected_utility_chunked = None
    aggregate_perturbed_ordinal_class_probs = None


PlotMode = Literal["single", "categorical", "multi_objective"]
LabelMode = Literal["auto", "regression", "classification", "ordinal"]


def _get_base_model(model: Any, output_index: int = 0) -> Any:
    """ModelList 系の場合は output_index 番目の子モデルを返す。"""
    if hasattr(model, "models"):
        return model.models[output_index]
    return model


def _get_train_x(
    model: Any,
    output_index: int = 0,
    prefer_raw: bool = True,
) -> Tensor:
    """BoTorch 風モデルから train_X を取得する。

    優先順位:
        1. train_inputs_raw[0]
        2. train_inputs[0]
        3. train_X
    """
    base_model = _get_base_model(model, output_index=output_index)

    if prefer_raw and hasattr(base_model, "train_inputs_raw"):
        train_inputs_raw = getattr(base_model, "train_inputs_raw")
        if train_inputs_raw is not None:
            return train_inputs_raw[0]

    if hasattr(base_model, "train_inputs"):
        train_inputs = getattr(base_model, "train_inputs")
        if train_inputs is not None:
            return train_inputs[0]

    if hasattr(base_model, "train_X"):
        return getattr(base_model, "train_X")

    raise AttributeError(
        "train_X を取得できません。train_inputs_raw, train_inputs, train_X のいずれかを用意してください。"
    )


def _select_train_y_for_output(
    train_y: Tensor,
    *,
    output_index: int = 0,
    n_train: Optional[int] = None,
) -> Tensor:
    """train_y から指定 output の1次元ラベル/目的値を取り出す。

    BoTorch の multi-output ``SingleTaskGP`` では、内部の
    ``train_targets`` が ``[m, n]`` になることがある。
    一方、ユーザーが保持している ``train_y`` / ``train_Y`` は
    多くの場合 ``[n, m]`` である。

    この関数では ``n_train`` を基準に、
    - ``[n]`` / ``[n, 1]``
    - ``[n, m]``
    - ``[m, n]``
    のどれでも ``[n]`` に揃える。
    """
    y = train_y

    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)

    if y.ndim == 0:
        return y.reshape(1)

    if n_train is not None:
        # すでに単一 output の [n] / [n, 1] として渡されている場合
        if y.numel() == n_train:
            return y.reshape(-1)

        if y.ndim >= 2:
            # ユーザー側の典型: [n, m]
            if y.shape[0] == n_train:
                if y.shape[-1] == 1:
                    return y.squeeze(-1).reshape(-1)
                return y[..., output_index].reshape(-1)

            # BoTorch multi-output SingleTaskGP の内部表現の典型: [m, n]
            if y.shape[-1] == n_train:
                return y[output_index, ...].reshape(-1)

    # n_train が不明な場合の fallback。従来挙動を維持する。
    if y.ndim >= 2 and y.shape[-1] > 1:
        y = y[..., output_index]
    elif y.ndim >= 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    return y.reshape(-1)


def _get_train_y(model: Any, output_index: int = 0) -> Tensor:
    """BoTorch 風モデルから対象 output の train_y を取得する。"""
    base_model = _get_base_model(model, output_index=output_index)

    if hasattr(base_model, "train_targets"):
        y = getattr(base_model, "train_targets")
    elif hasattr(base_model, "train_Y"):
        y = getattr(base_model, "train_Y")
    else:
        raise AttributeError(
            "train_y を取得できません。train_targets または train_Y を用意してください。"
        )

    try:
        n_train = _get_train_x(model, output_index=output_index, prefer_raw=False).shape[-2]
    except Exception:
        n_train = None

    return _select_train_y_for_output(
        y,
        output_index=output_index,
        n_train=n_train,
    )


def _make_xy_grid(
    *,
    x1_range: tuple[float, float] = (-1.5, 2.5),
    x2_range: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 100,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """2次元の可視化グリッドを作成する。"""
    x1 = torch.linspace(x1_range[0], x1_range[1], n_grid, device=device, dtype=dtype)
    x2 = torch.linspace(x2_range[0], x2_range[1], n_grid, device=device, dtype=dtype)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    X = torch.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, X


def _make_mixed_grid_input(
    X_2d: Tensor,
    *,
    category_value: int | float,
    categorical_dim: int = 2,
    total_dim: Optional[int] = None,
    x_dims: tuple[int, int] = (0, 1),
) -> Tensor:
    """2Dグリッドにカテゴリ列を付与して mixed 入力を作る。

    Args:
        X_2d: shape = [N, 2] の可視化用連続入力。
        category_value: 付与するカテゴリ値。テスト関数では 0 または 1 を想定。
        categorical_dim: カテゴリ列の次元。
        total_dim: mixed 入力の総次元。None の場合は max(x_dims, categorical_dim)+1。
        x_dims: X_2d の2列を配置する連続次元。

    Returns:
        shape = [N, total_dim] の mixed 入力。
    """
    if total_dim is None:
        total_dim = max(max(x_dims), categorical_dim) + 1

    X = torch.zeros(
        X_2d.shape[0],
        total_dim,
        device=X_2d.device,
        dtype=X_2d.dtype,
    )
    X[:, x_dims[0]] = X_2d[:, 0]
    X[:, x_dims[1]] = X_2d[:, 1]
    X[:, categorical_dim] = torch.as_tensor(
        category_value,
        device=X_2d.device,
        dtype=X_2d.dtype,
    )
    return X


def _posterior(
    model: Any,
    X: Tensor,
    *,
    input_perturbation: bool = False,
    n_w: Optional[int] = None,
    chunk_size: int = 128,
    variance_mode: str = "total",
):
    """通常 posterior または入力摂動集約 posterior を返す。"""
    if not input_perturbation:
        return model.posterior(X)

    if aggregate_perturbed_posterior_chunked is None:
        raise ImportError(
            "aggregate_perturbed_posterior_chunked を import できません。"
            "bochan.models.transforms.posterior を確認してください。"
        )
    if n_w is None:
        raise ValueError("input_perturbation=True の場合は n_w を指定してください。")

    return aggregate_perturbed_posterior_chunked(
        model=model,
        X=X,
        n_w=n_w,
        variance_mode=variance_mode,
        chunk_size=chunk_size,
    )


def _select_output(values: Tensor, output_index: int = 0) -> Tensor:
    """posterior.mean / variance から指定 output の1次元値を取り出す。"""
    if values.ndim >= 2 and values.shape[-1] > 1:
        values = values[..., output_index]
    elif values.ndim >= 2 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    return values.reshape(-1)


def _to_numpy(x: Tensor):
    return x.detach().cpu().numpy()


def _scatter_train_points(
    ax,
    train_x: Tensor,
    train_y: Tensor,
    *,
    label_mode: LabelMode = "auto",
    x_dims: tuple[int, int] = (0, 1),
    point_size: int = 30,
):
    """学習点を散布図として重ねる。

    色分けルール:
        - classification: 2クラス想定 (0=blue, 1=red)
        - ordinal: 3クラス想定 (0=green, 1=blue, 2=red)
        - regression: train_y の連続値をそのまま色に使う

    Notes:
        label_mode="auto" の場合、整数ラベルかつ
        - ユニーク数 <= 2: classification
        - ユニーク数 == 3: ordinal
        - それ以外: regression
        として扱う。
    """
    train_x_np = _to_numpy(train_x)
    y = train_y.detach().cpu().reshape(-1)
    y_np = y.numpy()

    if label_mode == "auto":
        unique_y = torch.unique(y)
        is_integer_like = torch.allclose(y, y.round())
        if is_integer_like:
            if unique_y.numel() <= 2:
                label_mode = "classification"
            elif unique_y.numel() == 3:
                label_mode = "ordinal"
            else:
                label_mode = "regression"
        else:
            label_mode = "regression"

    if label_mode == "classification":
        color_map = {0: "blue", 1: "red"}
        point_colors = [color_map.get(int(v), "gray") for v in y_np]
        sc = ax.scatter(
            train_x_np[:, x_dims[0]],
            train_x_np[:, x_dims[1]],
            c=point_colors,
            edgecolors="k",
            s=point_size,
        )
    elif label_mode == "ordinal":
        color_map = {0: "green", 1: "blue", 2: "red"}
        point_colors = [color_map.get(int(v), "gray") for v in y_np]
        sc = ax.scatter(
            train_x_np[:, x_dims[0]],
            train_x_np[:, x_dims[1]],
            c=point_colors,
            edgecolors="k",
            s=point_size,
        )
    else:
        sc = ax.scatter(
            train_x_np[:, x_dims[0]],
            train_x_np[:, x_dims[1]],
            c=y_np,
            edgecolors="k",
            s=point_size,
        )
    return sc


def _scatter_candidates(
    ax,
    candidates: Optional[Tensor],
    *,
    x_dims: tuple[int, int] = (0, 1),
    point_size: int = 90,
    marker: str = "X",
):
    """提案候補点を散布図として重ねる。

    Args:
        ax: matplotlib の axis。
        candidates: 候補点。None の場合は何もしない。
        x_dims: 2D プロットに使う連続次元。
        point_size: 候補点のサイズ。
        marker: 候補点の marker。丸以外を推奨し、デフォルトは "X"。
    """
    if candidates is None:
        return None

    cand = candidates
    if not isinstance(cand, torch.Tensor):
        cand = torch.as_tensor(cand)

    if cand.numel() == 0:
        return None

    if cand.ndim == 1:
        cand = cand.unsqueeze(0)

    cand_np = _to_numpy(cand)
    sc = ax.scatter(
        cand_np[:, x_dims[0]],
        cand_np[:, x_dims[1]],
        marker=marker,
        c="yellow",
        edgecolors="k",
        linewidths=1.0,
        s=point_size,
        zorder=5,
    )
    return sc


def _plot_contour(
    ax,
    fig,
    X1: Tensor,
    X2: Tensor,
    Z: Tensor,
    *,
    title: str,
    colorbar_label: str,
    levels: int = 40,
    contour_levels: int = 12,
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
    candidates: Optional[Tensor] = None,
    label_mode: LabelMode = "auto",
    x_dims: tuple[int, int] = (0, 1),
    point_size: int = 30,
    candidate_point_size: int = 90,
    candidate_marker: str = "X",
):
    X1_np = _to_numpy(X1)
    X2_np = _to_numpy(X2)
    Z_np = _to_numpy(Z)

    cf = ax.contourf(X1_np, X2_np, Z_np, levels=levels)
    fig.colorbar(cf, ax=ax, label=colorbar_label)
    ax.contour(
        X1_np,
        X2_np,
        Z_np,
        levels=contour_levels,
        colors="k",
        linewidths=0.5,
        alpha=0.35,
    )

    if train_x is not None and train_y is not None and train_x.numel() > 0:
        _scatter_train_points(
            ax,
            train_x,
            train_y,
            label_mode=label_mode,
            x_dims=x_dims,
            point_size=point_size,
        )

    _scatter_candidates(
        ax,
        candidates,
        x_dims=x_dims,
        point_size=candidate_point_size,
        marker=candidate_marker,
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    return cf


def show_predict_single(
    model: Any,
    n: Optional[int] = None,
    *,
    train_x =None,
    train_y =None,
    input_perturbation: bool = False,
    n_w: Optional[int] = None,
    chunk_size: int = 128,
    variance_mode: str = "total",
    output_index: int = 0,
    x1_range: tuple[float, float] = (-1.5, 2.5),
    x2_range: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 100,
    label_mode: LabelMode = "auto",
    candidates: Optional[Tensor] = None,
    levels: int = 40,
    contour_levels: int = 12,
    point_size: int = 30,
    candidate_point_size: int = 90,
    candidate_marker: str = "X",
    figsize: tuple[float, float] = (12, 5),
    show: bool = True,
):
    """単目的・通常入力モデルの予測平均と分散を表示する。

    表示内容:
        - 左: posterior mean
        - 右: posterior variance

    入力摂動ありの場合は aggregate_perturbed_posterior_chunked で posterior を計算する。
    """
    if train_x is None:
        train_x = _get_train_x(model, output_index=output_index)
        train_y = _get_train_y(model, output_index=output_index)
    device = train_x.device
    dtype = train_x.dtype

    if candidates is not None and not isinstance(candidates, torch.Tensor):
        candidates = torch.as_tensor(candidates, device=device, dtype=dtype)

    X1, X2, X = _make_xy_grid(
        x1_range=x1_range,
        x2_range=x2_range,
        n_grid=n_grid,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        post = _posterior(
            model,
            X,
            input_perturbation=input_perturbation,
            n_w=n_w,
            chunk_size=chunk_size,
            variance_mode=variance_mode,
        )
        mean = _select_output(post.mean, output_index=output_index).reshape(X1.shape)
        variance = _select_output(post.variance, output_index=output_index).reshape(X1.shape)

    title_suffix = f" #{n + 1}" if n is not None else ""
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    _plot_contour(
        ax[0],
        fig,
        X1,
        X2,
        mean,
        title=f"contour of predict value{title_suffix}",
        colorbar_label="mean",
        levels=levels,
        contour_levels=contour_levels,
        train_x=train_x,
        train_y=train_y,
        candidates=candidates,
        label_mode=label_mode,
        point_size=point_size,
        candidate_point_size=candidate_point_size,
        candidate_marker=candidate_marker,
    )
    _plot_contour(
        ax[1],
        fig,
        X1,
        X2,
        variance,
        title=f"contour of variance{title_suffix}",
        colorbar_label="variance",
        levels=levels,
        contour_levels=contour_levels,
        train_x=train_x,
        train_y=train_y,
        candidates=candidates,
        label_mode=label_mode,
        point_size=point_size,
        candidate_point_size=candidate_point_size,
        candidate_marker=candidate_marker,
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def show_predict_categorical(
    model: Any,
    n: Optional[int] = None,
    *,
    train_x =None,
    train_y =None,
    categories: Sequence[int | float] = (0, 1),
    categorical_dim: int = 2,
    input_perturbation: bool = False,
    n_w: Optional[int] = None,
    chunk_size: int = 128,
    variance_mode: str = "total",
    output_index: int = 0,
    x_dims: tuple[int, int] = (0, 1),
    x1_range: tuple[float, float] = (-1.5, 2.5),
    x2_range: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 100,
    label_mode: LabelMode = "auto",
    candidates: Optional[Tensor] = None,
    levels: int = 40,
    contour_levels: int = 12,
    point_size: int = 30,
    candidate_point_size: int = 90,
    candidate_marker: str = "X",
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
):
    """単目的・mixed 入力モデルについて、各カテゴリの posterior mean を表示する。

    テスト関数向けに categories=(0, 1) をデフォルトにしている。
    入力摂動ありの場合は aggregate_perturbed_posterior_chunked で posterior を計算する。
    """
    if train_x is None:
        train_x = _get_train_x(model, output_index=output_index)
        train_y = _get_train_y(model, output_index=output_index)
    device = train_x.device
    dtype = train_x.dtype
    total_dim = train_x.shape[-1]

    if candidates is not None and not isinstance(candidates, torch.Tensor):
        candidates = torch.as_tensor(candidates, device=device, dtype=dtype)

    X1, X2, X_2d = _make_xy_grid(
        x1_range=x1_range,
        x2_range=x2_range,
        n_grid=n_grid,
        device=device,
        dtype=dtype,
    )

    if figsize is None:
        figsize = (6 * len(categories), 5)
    fig, ax = plt.subplots(1, len(categories), figsize=figsize)
    if len(categories) == 1:
        ax = [ax]

    title_suffix = f" #{n + 1}" if n is not None else ""

    for i, cat in enumerate(categories):
        X_cat = _make_mixed_grid_input(
            X_2d,
            category_value=cat,
            categorical_dim=categorical_dim,
            total_dim=total_dim,
            x_dims=x_dims,
        )

        with torch.no_grad():
            post = _posterior(
                model,
                X_cat,
                input_perturbation=input_perturbation,
                n_w=n_w,
                chunk_size=chunk_size,
                variance_mode=variance_mode,
            )
            mean = _select_output(post.mean, output_index=output_index).reshape(X1.shape)

        mask = train_x[:, categorical_dim] == torch.as_tensor(
            cat,
            device=train_x.device,
            dtype=train_x.dtype,
        )

        candidate_mask = None
        if candidates is not None:
            candidate_mask = candidates[:, categorical_dim] == torch.as_tensor(
                cat,
                device=candidates.device,
                dtype=candidates.dtype,
            )

        _plot_contour(
            ax[i],
            fig,
            X1,
            X2,
            mean,
            title=f"contour of predict value: cat={cat}{title_suffix}",
            colorbar_label="mean",
            levels=levels,
            contour_levels=contour_levels,
            train_x=train_x[mask],
            train_y=train_y[mask],
            candidates=candidates[candidate_mask] if candidate_mask is not None else None,
            label_mode=label_mode,
            x_dims=x_dims,
            point_size=point_size,
            candidate_point_size=candidate_point_size,
            candidate_marker=candidate_marker,
        )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def show_predict_multi_objective(
    model: Any,
    n: Optional[int] = None,
    *,
    train_x=None,
    train_y=None,
    output_indices: Sequence[int] = (0, 1),
    objective_names: Optional[Sequence[str]] = None,
    mixed: bool = False,
    category_value: int | float = 0,
    categorical_dim: int = 2,
    input_perturbation: bool = False,
    n_w: Optional[int] = None,
    chunk_size: int = 128,
    variance_mode: str = "total",
    x_dims: tuple[int, int] = (0, 1),
    x1_range: tuple[float, float] = (-1.5, 2.5),
    x2_range: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 100,
    label_mode: LabelMode = "auto",
    candidates: Optional[Tensor] = None,
    levels: int = 40,
    contour_levels: int = 12,
    point_size: int = 30,
    candidate_point_size: int = 90,
    candidate_marker: str = "X",
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
):
    """多目的モデルについて、各目的変数の posterior mean を表示する。

    テスト関数向けに output_indices=(0, 1) をデフォルトにしている。
    mixed=True の場合は、category_value で指定したカテゴリに固定して可視化する。
    入力摂動ありの場合は aggregate_perturbed_posterior_chunked で posterior を計算する。
    """
    if train_x is None:
        train_x = _get_train_x(model, output_index=output_indices[0])
    device = train_x.device
    dtype = train_x.dtype
    total_dim = train_x.shape[-1]

    if candidates is not None and not isinstance(candidates, torch.Tensor):
        candidates = torch.as_tensor(candidates, device=device, dtype=dtype)

    if train_y is not None and not isinstance(train_y, torch.Tensor):
        train_y = torch.as_tensor(train_y, device=device, dtype=dtype)
    elif isinstance(train_y, torch.Tensor):
        train_y = train_y.to(device=device)

    X1, X2, X_2d = _make_xy_grid(
        x1_range=x1_range,
        x2_range=x2_range,
        n_grid=n_grid,
        device=device,
        dtype=dtype,
    )

    if mixed:
        X = _make_mixed_grid_input(
            X_2d,
            category_value=category_value,
            categorical_dim=categorical_dim,
            total_dim=total_dim,
            x_dims=x_dims,
        )
        train_mask = train_x[:, categorical_dim] == torch.as_tensor(
            category_value,
            device=train_x.device,
            dtype=train_x.dtype,
        )
        title_mixed_suffix = f", cat={category_value}"
    else:
        X = X_2d
        train_mask = torch.ones(train_x.shape[0], dtype=torch.bool, device=train_x.device)
        title_mixed_suffix = ""

    candidate_mask = None
    if candidates is not None:
        if mixed:
            candidate_mask = candidates[:, categorical_dim] == torch.as_tensor(
                category_value,
                device=candidates.device,
                dtype=candidates.dtype,
            )
        else:
            candidate_mask = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)

    with torch.no_grad():
        post = _posterior(
            model,
            X,
            input_perturbation=input_perturbation,
            n_w=n_w,
            chunk_size=chunk_size,
            variance_mode=variance_mode,
        )

    if objective_names is None:
        objective_names = [f"objective {i}" for i in output_indices]

    if figsize is None:
        figsize = (6 * len(output_indices), 5)
    fig, ax = plt.subplots(1, len(output_indices), figsize=figsize)
    if len(output_indices) == 1:
        ax = [ax]

    title_suffix = f" #{n + 1}" if n is not None else ""

    for j, output_index in enumerate(output_indices):
        mean = _select_output(post.mean, output_index=output_index).reshape(X1.shape)

        if train_y is None:
            train_y_i = _get_train_y(model, output_index=output_index)
        else:
            train_y_i = _select_train_y_for_output(
                train_y,
                output_index=output_index,
                n_train=train_x.shape[0],
            )

        if train_y_i.shape[0] != train_x.shape[0]:
            raise ValueError(
                "train_x と train_y の行数が一致しません。"
                f" train_x.shape[0]={train_x.shape[0]}, "
                f"selected train_y.shape[0]={train_y_i.shape[0]}, "
                f"output_index={output_index}"
            )

        _plot_contour(
            ax[j],
            fig,
            X1,
            X2,
            mean,
            title=f"contour of {objective_names[j]} mean{title_mixed_suffix}{title_suffix}",
            colorbar_label="mean",
            levels=levels,
            contour_levels=contour_levels,
            train_x=train_x[train_mask],
            train_y=train_y_i[train_mask],
            candidates=candidates[candidate_mask] if candidate_mask is not None else None,
            label_mode=label_mode,
            x_dims=x_dims,
            point_size=point_size,
            candidate_point_size=candidate_point_size,
            candidate_marker=candidate_marker,
        )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def show_predict(
    model: Any,
    n: Optional[int] = None,
    *,
    train_x = None,
    train_y = None,
    plot_mode: PlotMode = "single",
    input_perturbation: bool = False,
    n_w: Optional[int] = None,
    chunk_size: int = 128,
    variance_mode: str = "total",
    # categorical / mixed 用
    categories: Sequence[int | float] = (0, 1),
    categorical_dim: int = 2,
    category_value: int | float = 0,
    mixed: bool = False,
    # multi-objective 用
    output_indices: Sequence[int] = (0, 1),
    objective_names: Optional[Sequence[str]] = None,
    output_index: int = 0,
    # grid / plot 共通
    x_dims: tuple[int, int] = (0, 1),
    x1_range: tuple[float, float] = (-1.5, 2.5),
    x2_range: tuple[float, float] = (-1.5, 1.5),
    n_grid: int = 100,
    label_mode: LabelMode = "auto",
    candidates: Optional[Tensor] = None,
    levels: int = 40,
    contour_levels: int = 12,
    point_size: int = 30,
    candidate_point_size: int = 90,
    candidate_marker: str = "X",
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
):
    """予測結果可視化の統一入口。

    Args:
        model: BoTorch 風モデル。
        n: 反復番号。タイトル表示用。None なら表示しない。
        plot_mode:
            - "single": 単目的・通常入力。mean と variance を表示。
            - "categorical": 単目的・mixed 入力。各カテゴリの mean を表示。
            - "multi_objective": 多目的。各目的変数の mean を表示。
        input_perturbation:
            True の場合、aggregate_perturbed_posterior_chunked で posterior を集約する。
        n_w:
            入力摂動数。input_perturbation=True の場合に必須。
        chunk_size:
            入力摂動集約時の chunk サイズ。
        variance_mode:
            aggregate_perturbed_posterior_chunked に渡す variance_mode。
        categories:
            plot_mode="categorical" のときに可視化するカテゴリ値。
        categorical_dim:
            カテゴリ列の次元。テスト関数では 2 を想定。
        category_value:
            plot_mode="multi_objective" かつ mixed=True のときに固定するカテゴリ値。
        mixed:
            plot_mode="multi_objective" で mixed 入力を使う場合 True。
        output_indices:
            plot_mode="multi_objective" で表示する目的変数 index。
        objective_names:
            目的変数名。None の場合は objective 0, objective 1, ...。
        output_index:
            plot_mode="single" / "categorical" で表示する output index。
        x_dims:
            2Dプロットに使う連続次元。
        x1_range, x2_range:
            可視化範囲。
        n_grid:
            グリッド分割数。
        label_mode:
            学習点の色付け方法。auto, regression, classification, ordinal。
        candidates:
            提案候補点。None の場合は描画しない。描画時は丸以外の marker
            (デフォルトは "X") で重ねる。
        levels:
            contourf のレベル数。
        contour_levels:
            contour のレベル数。
        point_size:
            学習点のサイズ。
        figsize:
            図サイズ。None の場合は plot_mode に応じて自動設定。
        show:
            True の場合 plt.show() を実行。

    Returns:
        (fig, ax)
    """
    if plot_mode == "single":
        return show_predict_single(
            model=model,
            n=n,
            train_x=train_x,
            train_y=train_y,
            input_perturbation=input_perturbation,
            n_w=n_w,
            chunk_size=chunk_size,
            variance_mode=variance_mode,
            output_index=output_index,
            x1_range=x1_range,
            x2_range=x2_range,
            n_grid=n_grid,
            label_mode=label_mode,
            candidates=candidates,
            levels=levels,
            contour_levels=contour_levels,
            point_size=point_size,
            candidate_point_size=candidate_point_size,
            candidate_marker=candidate_marker,
            figsize=figsize or (12, 5),
            show=show,
        )

    if plot_mode == "categorical":
        return show_predict_categorical(
            model=model,
            n=n,
            train_x=train_x,
            train_y=train_y,
            categories=categories,
            categorical_dim=categorical_dim,
            input_perturbation=input_perturbation,
            n_w=n_w,
            chunk_size=chunk_size,
            variance_mode=variance_mode,
            output_index=output_index,
            x_dims=x_dims,
            x1_range=x1_range,
            x2_range=x2_range,
            n_grid=n_grid,
            label_mode=label_mode,
            candidates=candidates,
            levels=levels,
            contour_levels=contour_levels,
            point_size=point_size,
            candidate_point_size=candidate_point_size,
            candidate_marker=candidate_marker,
            figsize=figsize,
            show=show,
        )

    if plot_mode == "multi_objective":
        return show_predict_multi_objective(
            model=model,
            n=n,
            train_x=train_x,
            train_y=train_y,
            output_indices=output_indices,
            objective_names=objective_names,
            mixed=mixed,
            category_value=category_value,
            categorical_dim=categorical_dim,
            input_perturbation=input_perturbation,
            n_w=n_w,
            chunk_size=chunk_size,
            variance_mode=variance_mode,
            x_dims=x_dims,
            x1_range=x1_range,
            x2_range=x2_range,
            n_grid=n_grid,
            label_mode=label_mode,
            candidates=candidates,
            levels=levels,
            contour_levels=contour_levels,
            point_size=point_size,
            candidate_point_size=candidate_point_size,
            candidate_marker=candidate_marker,
            figsize=figsize,
            show=show,
        )

    raise ValueError(f"Unknown plot_mode: {plot_mode}")


__all__ = [
    "show_predict",
    "show_predict_single",
    "show_predict_categorical",
    "show_predict_multi_objective",
]
