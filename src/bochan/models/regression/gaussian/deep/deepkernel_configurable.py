from __future__ import annotations

from collections.abc import Sequence

from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from torch import Tensor, nn

from bochan.models.components.layers import DeepKernel, DeepKernelMixed

from .deepkernel import (
    InputTransformArg,
    OutcomeTransformArg,
    _BaseDeepKernelGPModel,
)


class DeepKernelGaussianGPModel(_BaseDeepKernelGPModel):
    """連続入力向け Deep Kernel GP 回帰モデル。

    Args:
        hidden_dims: DeepKernel feature extractor の隠れ層次元。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
        feature_extractor: ``R^input_dim -> R^latent_dim`` の任意の module。
            None の場合は既存の MLP / skip MLP を使う。
        latent_dim: GP kernel が受け取る特徴次元。None の場合、既存 MLP では
            input_dim、任意 module では ``output_dim`` 属性または sample forward
            から解決する。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Sequence[int] | None = None,
        feature_extractor: nn.Module | None = None,
        latent_dim: int | None = None,
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
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        self.latent_dim = self.deepkernel.latent_dim
        self.to(train_X)


class DeepKernelGaussianMixedGPModel(_BaseDeepKernelGPModel):
    """混合入力（連続 + カテゴリ）向け Deep Kernel GP 回帰モデル。

    Args:
        hidden_dims: 連続変数側 feature extractor の隠れ層次元。
            None の場合は従来通り [cont_dim * 8, cont_dim * 4, cont_dim * 2] を使う。
        feature_extractor: 連続変数から任意の latent 表現を作る module。
        latent_dim: feature extractor の出力次元。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        likelihood=None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        ext_type: str = "DEFAULT",
        hidden_dims: Sequence[int] | None = None,
        feature_extractor: nn.Module | None = None,
        latent_dim: int | None = None,
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
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self.ext_type = str(ext_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        self.latent_dim = self.deepkernel.latent_dim
        self.to(train_X)


__all__ = ["DeepKernelGaussianGPModel", "DeepKernelGaussianMixedGPModel"]
