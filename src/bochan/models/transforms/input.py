from typing import List, Optional, Union, Literal
import torch
from torch import Tensor

from botorch.utils.sampling import draw_sobol_normal_samples
from botorch.models.transforms.input import Normalize, InputPerturbation, ChainedInputTransform

STD_DEV = 0.1
N_W = 16

RiskType = Literal["none", "var", "cvar"]


def _normalize_cat_dims(
    cat_dims: Optional[List[int]],
    dim: int,
) -> List[int]:
    """カテゴリ列 index を検証して、重複なし・昇順に整形する。"""
    if cat_dims is None:
        return []

    cat_dims = sorted(set(int(i) for i in cat_dims))

    invalid = [i for i in cat_dims if i < 0 or i >= dim]
    if invalid:
        raise ValueError(
            f"cat_dims contains invalid indices: {invalid}. "
            f"Valid range is [0, {dim - 1}]."
        )

    return cat_dims


def _continuous_indices(
    dim: int,
    cat_dims: Optional[List[int]],
) -> List[int]:
    """カテゴリ列を除いた連続変数 index を返す。"""
    cat_dims = _normalize_cat_dims(cat_dims, dim)
    cat_set = set(cat_dims)
    return [i for i in range(dim) if i not in cat_set]


def _infer_bounds_from_train_X(train_X: Tensor) -> Tensor:
    """train_X から BoTorch 形式の bounds を推定する。"""
    if train_X.ndim < 2:
        raise ValueError("train_X must have shape n x d or batch_shape x n x d to infer bounds.")
    return torch.stack(
        [
            train_X.min(dim=-2).values,
            train_X.max(dim=-2).values,
        ],
        dim=-2,
    )


def setup_input_perturbation(
    dim: int,
    bounds: Tensor,
    n: int = N_W,
    std: float = STD_DEV,
    perturbation: bool = False,
    cat_dims: Optional[List[int]] = None,
    **tkwargs,
) -> Optional[InputPerturbation]:
    """
    入力空間における不確実性（摂動）を設定する。

    Args:
        dim:
            入力次元数。
        bounds:
            摂動後の入力を clamp する境界。
            Normalize 後に使う場合、連続変数は [0, 1]、
            カテゴリ変数は元スケールの bounds にしておく。
        n:
            摂動サンプル数、つまり n_w。
        std:
            摂動の標準偏差。
            Normalize 後に使う場合は、正規化空間での標準偏差。
        perturbation:
            摂動を有効にするかどうか。
        cat_dims:
            カテゴリ変数の index。
            ここには摂動を与えない。
        **tkwargs:
            dtype, device など。
    """
    if not perturbation:
        return None

    cat_dims = _normalize_cat_dims(cat_dims, dim)

    raw_perturbation_set = draw_sobol_normal_samples(
        d=dim,
        n=n,
        **tkwargs,
    ) * std

    perturbation_set = raw_perturbation_set.clone()

    # カテゴリ変数には摂動を与えない
    if cat_dims:
        perturbation_set[:, cat_dims] = 0.0

    return InputPerturbation(
        perturbation_set=perturbation_set,
        bounds=bounds,
    )


def _extract_transform_metadata(bounds: Tensor | dict | None, normalize: bool) -> tuple[Tensor | None, bool]:
    """API 側から渡された lightweight metadata を取り出す。

    既存の factory との互換を保つため、InputTransformConfig 側では normalize フラグを
    bounds に metadata として同梱できるようにしている。
    """
    if isinstance(bounds, dict) and bounds.get("__bochan_input_transform_config__", False):
        return bounds.get("bounds"), bool(bounds.get("normalize", normalize))
    return bounds, normalize


def build_input_transform(
    train_X: Tensor,
    bounds: Tensor | dict | None,
    perturbation: bool,
    categorical_idx: Optional[List[int]] = None,
    n_w: int = N_W,
    std: float = STD_DEV,
    normalize: bool = True,
) -> Optional[Union[Normalize, InputPerturbation, ChainedInputTransform]]:
    """
    Normalize と必要に応じた InputPerturbation をまとめて構築する。

    Args:
        train_X:
            学習入力。
        bounds:
            raw-space の bounds。API の `InputTransformConfig` 経由では metadata dict が
            渡される場合もある。None の場合は train_X から自動推定する。
        perturbation:
            InputPerturbation を使うか。
        categorical_idx:
            カテゴリ変数 index。Normalize / perturbation の対象から除外する。
        n_w:
            摂動サンプル数。
        std:
            摂動標準偏差。normalize=True なら Normalize 後空間、normalize=False なら raw 空間。
        normalize:
            Normalize を使うか。

    重要:
        - カテゴリ変数は Normalize しない。
        - normalize=True かつ perturbation=True の場合、InputPerturbation は Normalize 後に適用する。
        - normalize=False かつ perturbation=True の場合、InputPerturbation は raw 空間に適用する。
        - normalize=False かつ perturbation=False の場合、None を返す。
    """
    bounds, normalize = _extract_transform_metadata(bounds, normalize)
    if bounds is None:
        bounds = _infer_bounds_from_train_X(train_X)

    bounds = bounds.to(dtype=train_X.dtype, device=train_X.device)
    dim = train_X.shape[-1]
    categorical_idx = _normalize_cat_dims(categorical_idx, dim)
    continuous_idx = _continuous_indices(dim, categorical_idx)

    if normalize:
        # 連続変数だけ Normalize する
        tf_normalize = Normalize(
            d=dim,
            bounds=bounds,
            indices=continuous_idx,
        )

        if not perturbation:
            return tf_normalize

        # Normalize 後の空間における perturbation bounds
        # 連続変数は [0, 1]、カテゴリ変数は元スケールの bounds を使う
        pert_bounds = bounds.clone().to(dtype=train_X.dtype, device=train_X.device)

        if continuous_idx:
            pert_bounds[0, continuous_idx] = 0.0
            pert_bounds[1, continuous_idx] = 1.0

        tf_perturb = setup_input_perturbation(
            dim=dim,
            bounds=pert_bounds,
            n=n_w,
            std=std,
            perturbation=True,
            cat_dims=categorical_idx,
            dtype=train_X.dtype,
            device=train_X.device,
        )

        return ChainedInputTransform(
            normalize=tf_normalize,
            perturb=tf_perturb,
        )

    if not perturbation:
        return None

    # raw 空間における perturbation bounds
    pert_bounds = bounds.clone().to(dtype=train_X.dtype, device=train_X.device)
    return setup_input_perturbation(
        dim=dim,
        bounds=pert_bounds,
        n=n_w,
        std=std,
        perturbation=True,
        cat_dims=categorical_idx,
        dtype=train_X.dtype,
        device=train_X.device,
    )
