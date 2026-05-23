from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL


def get_batch_dimensions(
    train_X: Tensor,
    train_Y: Tensor,
) -> Tuple[torch.Size, torch.Size]:
    """
    train_X / train_Y から BoTorch-style の batch_shape を推定する。

    Args:
        train_X:
            入力データ。
            shape: batch_shape x n x d

        train_Y:
            出力データ。
            shape: batch_shape x n x m

    Returns:
        input_batch_shape:
            入力側の batch_shape。

        aug_batch_shape:
            出力次元 m を batch として扱うための拡張 batch_shape。
            m == 1 の場合は input_batch_shape のまま。
            m > 1 の場合は input_batch_shape + torch.Size([m])。
    """
    _validate_train_shapes(train_X=train_X, train_Y=train_Y)

    input_batch_shape = train_X.shape[:-2]
    num_outputs = train_Y.shape[-1]

    if num_outputs > 1:
        aug_batch_shape = input_batch_shape + torch.Size([num_outputs])
    else:
        aug_batch_shape = input_batch_shape

    return input_batch_shape, aug_batch_shape


def _validate_train_shapes(
    train_X: Tensor,
    train_Y: Tensor,
) -> None:
    """
    likelihood 構築に必要な最低限の shape 検証を行う。
    """
    if train_X.ndim < 2:
        raise ValueError(
            "`train_X` must have shape `batch_shape x n x d`."
        )

    if train_Y.ndim < 2:
        raise ValueError(
            "`train_Y` must have shape `batch_shape x n x m`. "
            "For single-output regression, use shape `n x 1`, not `n`."
        )

    x_batch_shape = train_X.shape[:-2]
    y_batch_shape = train_Y.shape[:-2]

    if x_batch_shape != y_batch_shape:
        raise ValueError(
            "`train_X` and `train_Y` must have the same batch shape. "
            f"Got train_X batch_shape={x_batch_shape}, "
            f"train_Y batch_shape={y_batch_shape}."
        )

    if train_X.shape[-2] != train_Y.shape[-2]:
        raise ValueError(
            "`train_X` and `train_Y` must have the same number of observations. "
            f"Got train_X n={train_X.shape[-2]}, train_Y n={train_Y.shape[-2]}."
        )


def _num_outputs(train_Y: Tensor) -> int:
    """
    train_Y の最終次元から出力数を取得する。
    """
    if train_Y.ndim < 2:
        raise ValueError(
            "`train_Y` must have shape `batch_shape x n x m`."
        )

    return train_Y.shape[-1]


def _default_noise_prior() -> LogNormalPrior:
    """
    BoTorch SingleTaskGP に近い Gaussian noise prior を返す。

    Note:
        Prior オブジェクトを使い回さないため、毎回新しいインスタンスを返す。
    """
    return LogNormalPrior(loc=-4.0, scale=1.0)


def _make_noise_constraint(
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
    noise_prior: Optional[LogNormalPrior] = None,
) -> GreaterThan:
    """
    Gaussian likelihood 用の noise_constraint を作成する。

    Args:
        alpha:
            noise の下限値。

        noise_prior:
            noise prior。
            指定された場合は prior.mode を initial_value として使う。

    Returns:
        GreaterThan constraint。
    """
    if noise_prior is None:
        return GreaterThan(lower_bound=alpha)

    return GreaterThan(
        lower_bound=alpha,
        transform=None,
        initial_value=noise_prior.mode,
    )


def _make_noise_prior_and_constraint(
    use_noise_prior: bool,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> tuple[Optional[LogNormalPrior], GreaterThan]:
    """
    noise_prior と noise_constraint をまとめて作成する。

    Args:
        use_noise_prior:
            True の場合は LogNormalPrior を使う。
            False の場合は prior なし。

        alpha:
            noise の下限値。

    Returns:
        noise_prior:
            use_noise_prior=True の場合は LogNormalPrior。
            False の場合は None。

        noise_constraint:
            GreaterThan constraint。
    """
    noise_prior = _default_noise_prior() if use_noise_prior else None
    noise_constraint = _make_noise_constraint(
        alpha=alpha,
        noise_prior=noise_prior,
    )

    return noise_prior, noise_constraint


def build_single_task_likelihood(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    deep: bool = False,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> GaussianLikelihood:
    """
    SingleTaskGP / deep GP 用の GaussianLikelihood を構築する。

    Args:
        train_X:
            学習入力。
            shape: batch_shape x n x d

        train_Y:
            学習出力。
            shape: batch_shape x n x m

        deep:
            True の場合、noise_prior を使わない。
            DeepGP / Variational GP などで prior を外したい場合に使う。

        alpha:
            noise_constraint の下限値。

    Returns:
        GaussianLikelihood。
    """
    _, aug_batch_shape = get_batch_dimensions(
        train_X=train_X,
        train_Y=train_Y,
    )

    noise_prior, noise_constraint = _make_noise_prior_and_constraint(
        use_noise_prior=not deep,
        alpha=alpha,
    )

    kwargs = {
        "batch_shape": aug_batch_shape,
        "noise_constraint": noise_constraint,
    }

    if noise_prior is not None:
        kwargs["noise_prior"] = noise_prior

    return GaussianLikelihood(**kwargs)


def build_multitask_likelihood(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    deep: bool = False,
    rank: Optional[int] = None,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> MultitaskGaussianLikelihood:
    """
    MultiTaskGP / 多出力 GP 用の MultitaskGaussianLikelihood を構築する。

    Args:
        train_X:
            学習入力。
            shape: batch_shape x n x d

        train_Y:
            学習出力。
            shape: batch_shape x n x m

        deep:
            True の場合、noise_prior を使わない。

        rank:
            タスク間ノイズ共分散の低ランク表現のランク。
            None の場合は num_tasks と同じにする。

        alpha:
            noise_constraint の下限値。

    Returns:
        MultitaskGaussianLikelihood。
    """
    batch_shape, _ = get_batch_dimensions(
        train_X=train_X,
        train_Y=train_Y,
    )

    num_tasks = _num_outputs(train_Y)

    if rank is None:
        rank = num_tasks

    if rank < 0:
        raise ValueError(f"`rank` must be non-negative. Got rank={rank}.")

    if rank > num_tasks:
        raise ValueError(
            "`rank` must be less than or equal to `num_tasks`. "
            f"Got rank={rank}, num_tasks={num_tasks}."
        )

    noise_prior, noise_constraint = _make_noise_prior_and_constraint(
        use_noise_prior=not deep,
        alpha=alpha,
    )

    kwargs = {
        "num_tasks": num_tasks,
        "batch_shape": batch_shape,
        "rank": rank,
        "noise_constraint": noise_constraint,
    }

    if noise_prior is not None:
        kwargs["noise_prior"] = noise_prior

    return MultitaskGaussianLikelihood(**kwargs)


# ---------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------

def singletasklikelihood(
    train_X: Tensor,
    train_Y: Tensor,
    deep: bool = False,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> GaussianLikelihood:
    """
    後方互換用の alias。

    新規コードでは `build_single_task_likelihood` の使用を推奨。
    """
    return build_single_task_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        deep=deep,
        alpha=alpha,
    )


def multitasklikelihood(
    train_X: Tensor,
    train_Y: Tensor,
    deep: bool = False,
    rank: Optional[int] = None,
    alpha: float = MIN_INFERRED_NOISE_LEVEL,
) -> MultitaskGaussianLikelihood:
    """
    後方互換用の alias。

    新規コードでは `build_multitask_likelihood` の使用を推奨。
    """
    return build_multitask_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        deep=deep,
        rank=rank,
        alpha=alpha,
    )