from __future__ import annotations

"""PCA / REMBO などの射影 wrapper で共通利用する補助関数群。

このモジュールは regression / classification / ordinal の各タスクに依存しない
処理だけを集約する。主な責務は以下。

- raw-space / preproject-space / projected-space の変換補助
- InputPerturbation を含む input_transform の train/eval 適用
- mixed 入力でカテゴリ列が変換されていないことの検査
- PCA / REMBO transformer の clone
- condition_on_observations 用の raw-space データ整形
"""

import copy
from typing import Optional, Sequence

import torch
from torch import Tensor

from botorch.models.transforms.input import InputTransform, Normalize

from bochan.models.components.decomposition import (
    PCAConfig,
    REMBOConfig,
    PCATransformer,
    REMBOTransformer,
)


__all__ = [
    "_clone_input_transform",
    "_ensure_2d_train_Y",
    "_ensure_Y_last_dim",
    "_flatten_targets",
    "_flatten_optional_noise",
    "_normalize_dims",
    "_get_cont_dims",
    "_expand_raw_X_to_match_transformed_q",
    "_check_categorical_columns_unchanged",
    "_apply_input_transform_for_training",
    "_apply_input_transform_for_eval",
    "_prepare_raw_input_transform_for_mixed",
    "_prepare_original_space_conditioning_data",
    "_concat_optional_noise",
    "_clone_fitted_pca",
    "_clone_fitted_rembo",
    "_resolve_latent_dim",
    "PCAConfig",
    "REMBOConfig",
    "PCATransformer",
    "REMBOTransformer",
]


def _clone_input_transform(
    input_transform: Optional[InputTransform],
) -> Optional[InputTransform]:
    """input_transform を安全に複製する。

    Args:
        input_transform: 複製したい BoTorch の input transform。

    Returns:
        複製された input_transform。``None`` の場合は ``None``。
    """
    return None if input_transform is None else copy.deepcopy(input_transform)


def _ensure_2d_train_Y(train_Y: Tensor) -> Tensor:
    """訓練用 Y を ``[n, m]`` 形状にそろえる。"""
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    return train_Y


def _ensure_Y_last_dim(Y: Tensor) -> Tensor:
    """condition_on_observations 用に Y の最後の出力次元を保証する。"""
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    return Y


def _flatten_targets(Y: Tensor, *, dtype: Optional[torch.dtype] = None) -> Tensor:
    """分類・順序回帰用 target を ``[n]`` にそろえる。"""
    if Y.ndim > 1 and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)
    Y = Y.reshape(-1)
    return Y if dtype is None else Y.to(dtype=dtype)


def _flatten_optional_noise(noise: Optional[Tensor]) -> Optional[Tensor]:
    """optional noise を ``[n]`` にそろえる。"""
    if noise is None:
        return None
    if noise.ndim > 1 and noise.shape[-1] == 1:
        noise = noise.squeeze(-1)
    return noise.reshape(-1)


def _normalize_dims(dims: Sequence[int], d: int) -> list[int]:
    """負の次元指定を正規化し、重複を除いた昇順 list にする。"""
    out: list[int] = []
    for idx in dims:
        j = int(idx)
        if j < 0:
            j = d + j
        if j < 0 or j >= d:
            raise ValueError(f"dim index {idx} is out of range for input dim {d}.")
        out.append(j)
    return sorted(set(out))


def _get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    """カテゴリ列以外の連続列 index を返す。"""
    cat = set(_normalize_dims(cat_dims, d))
    return [i for i in range(d) if i not in cat]


def _expand_raw_X_to_match_transformed_q(X: Tensor, X_tf: Tensor) -> Tensor:
    """InputPerturbation 後の q 展開に合わせて raw X を repeat する。

    Args:
        X: raw-space 入力。shape は ``batch_shape x q x d`` を想定。
        X_tf: input_transform 後の入力。InputPerturbation により
            ``q`` が ``q * n_w`` へ展開されることがある。

    Returns:
        X_tf と比較できる shape に揃えた X。
    """
    if X.shape == X_tf.shape:
        return X
    if X.ndim < 2 or X_tf.ndim < 2:
        return X
    if X.shape[-1] != X_tf.shape[-1]:
        return X
    if X.shape[:-2] == X_tf.shape[:-2]:
        q = X.shape[-2]
        q_like = X_tf.shape[-2]
        if q_like == q:
            return X
        if q > 0 and q_like % q == 0:
            return X.repeat_interleave(q_like // q, dim=-2)
    if X.numel() == X_tf.numel():
        return X.reshape_as(X_tf)
    return X


def _check_categorical_columns_unchanged(
    X: Tensor,
    X_tf: Tensor,
    cat_dims: Optional[Sequence[int]],
) -> None:
    """input_transform がカテゴリ列を変更していないことを確認する。

    mixed model では、カテゴリ列は raw encoding のまま base model に渡す必要がある。
    InputPerturbation によって q が増える場合は、raw X 側も repeat してから比較する。
    """
    if cat_dims is None or len(cat_dims) == 0:
        return

    cat_idx = [int(i) for i in cat_dims]
    X_cmp = _expand_raw_X_to_match_transformed_q(X, X_tf)

    if X_cmp.shape[:-1] != X_tf.shape[:-1]:
        raise RuntimeError(
            "Could not align raw X with transformed X for categorical column check. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}, "
            f"X_cmp.shape={tuple(X_cmp.shape)}."
        )

    if not torch.allclose(X_tf[..., cat_idx], X_cmp[..., cat_idx]):
        raise ValueError(
            "input_transform must not modify categorical columns. "
            "For mixed projected models, transform only continuous columns. "
            f"X_cat.shape={tuple(X_cmp[..., cat_idx].shape)}, "
            f"X_tf_cat.shape={tuple(X_tf[..., cat_idx].shape)}."
        )


def _apply_input_transform_for_training(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
    name: str = "input_transform",
) -> Tensor:
    """学習データ準備用に input_transform を適用する。

    Notes:
        InputPerturbation は eval mode で候補点を q*n_w に展開することがある。
        学習時に train_Y と点数がずれるのを避けるため、ここでは transform を
        train mode で適用し、点数が増えていないことを検査する。
    """
    if input_transform is None:
        return X.detach().clone()

    was_training = getattr(input_transform, "training", None)
    if hasattr(input_transform, "train"):
        input_transform.train()

    with torch.no_grad():
        X_tf = input_transform(X).detach().clone()

    if was_training is False and hasattr(input_transform, "eval"):
        input_transform.eval()

    if X_tf.shape[-2] != X.shape[-2]:
        raise RuntimeError(
            f"{name} expanded training inputs. "
            f"X.shape={tuple(X.shape)}, X_tf.shape={tuple(X_tf.shape)}. "
            "For InputPerturbation, ensure transform_on_train=False."
        )

    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _apply_input_transform_for_eval(
    X: Tensor,
    input_transform: Optional[InputTransform],
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    """posterior / acquisition 評価用に input_transform を適用する。

    Notes:
        ここでは候補点 X への勾配を残す。InputPerturbation による q 展開も許す。
    """
    if input_transform is None:
        return X

    was_training = getattr(input_transform, "training", None)
    if hasattr(input_transform, "eval"):
        input_transform.eval()

    X_tf = input_transform(X)

    if was_training is True and hasattr(input_transform, "train"):
        input_transform.train()

    _check_categorical_columns_unchanged(X=X, X_tf=X_tf, cat_dims=cat_dims)
    return X_tf


def _prepare_raw_input_transform_for_mixed(
    input_transform: Optional[InputTransform],
    *,
    input_dim: int,
    cont_dims: Sequence[int],
    cat_dims: Sequence[int],
) -> Optional[InputTransform]:
    """mixed 用 input_transform を検査・補正する。

    Normalize が indices=None の場合、明示 bounds があれば連続列のみを対象にした
    Normalize に差し替える。カテゴリ列を正規化対象に含む transform はエラーにする。
    """
    if input_transform is None:
        return None

    if isinstance(input_transform, Normalize):
        indices = getattr(input_transform, "indices", None)
        if indices is None:
            bounds = getattr(input_transform, "bounds", None)
            if bounds is None:
                raise ValueError(
                    "For mixed projected models, Normalize without indices requires explicit bounds."
                )
            return Normalize(d=input_dim, bounds=bounds, indices=list(cont_dims))

        if isinstance(indices, Tensor):
            idx_list = [int(i) for i in indices.view(-1).tolist()]
        else:
            idx_list = [int(i) for i in indices]

        bad = sorted(set(idx_list).intersection(set(int(i) for i in cat_dims)))
        if bad:
            raise ValueError(
                "raw-space Normalize indices must exclude categorical dims. "
                f"Got categorical indices: {bad}."
            )

    return input_transform


def _prepare_original_space_conditioning_data(
    X: Tensor,
    Y: Tensor,
    noise: Optional[Tensor],
    *,
    expected_input_dim: int,
    force_2d_Y: bool = False,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """condition_on_observations 用に raw-space 新規観測を整形する。

    Args:
        X: 新規観測点。最後の次元は raw-space の入力次元。
        Y: 新規観測値。
        noise: 観測ノイズ。未指定なら ``None``。
        expected_input_dim: 期待される raw-space 入力次元。
        force_2d_Y: ``True`` の場合、戻り値 Y を ``[n, 1]`` にする。

    Returns:
        ``(X_flat, Y_flat, noise_flat)``。
    """
    if isinstance(X, tuple):
        X = X[0]

    if X.shape[-1] != expected_input_dim:
        raise ValueError(
            "Projected wrapper condition_on_observations expects X in the original "
            f"input space with last dim = {expected_input_dim}, got {X.shape[-1]}."
        )

    expected_y_shape = X.shape[:-1]
    if Y.ndim == X.ndim and Y.shape[-1] == 1:
        Y = Y.squeeze(-1)

    if Y.shape != expected_y_shape:
        raise NotImplementedError(
            "This wrapper supports non-fantasy observations with "
            "Y.shape == X.shape[:-1] or trailing singleton dim. "
            f"Got X.shape={tuple(X.shape)}, Y.shape={tuple(Y.shape)}."
        )

    if noise is not None:
        if noise.ndim == X.ndim and noise.shape[-1] == 1:
            noise = noise.squeeze(-1)
        if noise.shape != expected_y_shape:
            raise ValueError(
                "noise must match X.shape[:-1] or have trailing singleton dim. "
                f"Got X.shape={tuple(X.shape)}, noise.shape={tuple(noise.shape)}."
            )

    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Y.reshape(-1).to(dtype=X.dtype, device=X.device)
    noise_flat = None if noise is None else noise.reshape(-1).to(dtype=X.dtype, device=X.device)

    if force_2d_Y:
        Y_flat = Y_flat.unsqueeze(-1)
        if noise_flat is not None:
            noise_flat = noise_flat.unsqueeze(-1)

    return X_flat, Y_flat, noise_flat


def _concat_optional_noise(
    old_Y: Tensor,
    old_Yvar: Optional[Tensor],
    new_Y: Tensor,
    new_Yvar: Optional[Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """old/new の train_Yvar を連結する。片方だけある場合は 0 埋めする。"""
    if old_Yvar is None and new_Yvar is None:
        return None

    old_shape = old_Y.shape
    new_shape = new_Y.shape

    if old_Yvar is None:
        old_Yvar = torch.zeros(old_shape, dtype=dtype, device=device)
    else:
        old_Yvar = old_Yvar.to(dtype=dtype, device=device)
        if old_Yvar.shape != old_shape:
            old_Yvar = old_Yvar.reshape(old_shape)

    if new_Yvar is None:
        new_Yvar = torch.zeros(new_shape, dtype=dtype, device=device)
    else:
        new_Yvar = new_Yvar.to(dtype=dtype, device=device)
        if new_Yvar.shape != new_shape:
            new_Yvar = new_Yvar.reshape(new_shape)

    return torch.cat([old_Yvar, new_Yvar], dim=0)


def _clone_fitted_pca(pca: PCATransformer) -> PCATransformer:
    """fit 済み PCA transformer を複製する。"""
    new_pca = PCATransformer(copy.deepcopy(pca.config))
    for name in ("mean_", "scale_", "components_"):
        value = getattr(pca, name, None)
        setattr(new_pca, name, None if value is None else value.detach().clone())
    return new_pca


def _clone_fitted_rembo(rembo: REMBOTransformer) -> REMBOTransformer:
    """fit 済み REMBO transformer を複製する。"""
    new_rembo = REMBOTransformer(copy.deepcopy(rembo.config))
    for name in ("mean_", "scale_", "projection_"):
        value = getattr(rembo, name, None)
        setattr(new_rembo, name, None if value is None else value.detach().clone())
    return new_rembo


def _resolve_latent_dim(
    *,
    latent_dim: Optional[int],
    n_components: Optional[int],
    default: int,
) -> int:
    """latent_dim / n_components の後方互換を解決する。"""
    if latent_dim is not None and n_components is not None and latent_dim != n_components:
        raise ValueError(
            f"latent_dim and n_components are both specified but inconsistent: "
            f"latent_dim={latent_dim}, n_components={n_components}."
        )
    value = n_components if n_components is not None else latent_dim
    return int(default if value is None else value)
