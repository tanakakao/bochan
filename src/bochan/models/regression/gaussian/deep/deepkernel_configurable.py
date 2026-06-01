from __future__ import annotations

from typing import Optional, Sequence

from torch import Tensor
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood

from botorch.utils.transforms import normalize_indices

from bochan.models.components.layers import DeepKernel, DeepKernelMixed
from .deepkernel import (
    _BaseDeepKernelGPModel,
    InputTransformArg,
    OutcomeTransformArg,
)


class DeepKernelGPModel(_BaseDeepKernelGPModel):
    """連続入力向け Deep Kernel GP 回帰モデル。

    Args:
        hidden_dims: DeepKernel feature extractor の隠れ層次元。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        _, train_X_tf, prepared_train_Y = self._setup_common(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        if likelihood is None:
            if self._num_outputs == 1:
                likelihood = GaussianLikelihood()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)

        self.likelihood = likelihood
        self.deepkernel = DeepKernel(
            train_x=train_X_tf,
            train_y=prepared_train_Y,
            likelihood=self.likelihood,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
        )
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        self.to(train_X)


class DeepKernelMixedGPModel(_BaseDeepKernelGPModel):
    """混合入力（連続 + カテゴリ）向け Deep Kernel GP 回帰モデル。

    Args:
        hidden_dims: 連続変数側 feature extractor の隠れ層次元。
            None の場合は従来通り [cont_dim * 8, cont_dim * 4, cont_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Optional[Tensor] = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        _, train_X_tf, prepared_train_Y = self._setup_common(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )

        if likelihood is None:
            if self._num_outputs == 1:
                from botorch.models.utils.gpytorch_modules import (
                    get_gaussian_likelihood_with_lognormal_prior,
                )
                likelihood = get_gaussian_likelihood_with_lognormal_prior()
            else:
                likelihood = MultitaskGaussianLikelihood(num_tasks=self._num_outputs)

        self.likelihood = likelihood
        self.deepkernel = DeepKernelMixed(
            train_x=train_X_tf,
            train_y=prepared_train_Y,
            cat_dims=cat_dims,
            likelihood=self.likelihood,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
        )
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        self.to(train_X)


__all__ = ["DeepKernelGPModel", "DeepKernelMixedGPModel"]
