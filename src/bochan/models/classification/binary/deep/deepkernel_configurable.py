from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from botorch.models.transforms.input import InputTransform
from botorch.utils.transforms import normalize_indices
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean

from bochan.models.components.layers.feature_extractor import (
    LargeFeatureExtractor,
    SkipLargeFeatureExtractor,
)
from .deepkernel import (
    _BaseDeepKernelBinaryClassificationModel,
    _DeepKernelLatentBinaryMixedSVGP,
    _DeepKernelLatentBinarySVGP,
    _make_train_X_tf,
    _prepare_binary_targets,
    _select_inducing_points,
    _to_device_dtype_transform,
)


def _default_feature_extractor(
    input_dim: int,
    model_type: str = "DEFAULT",
    hidden_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    """既定の特徴抽出器を返す。"""
    hidden_dims = (
        [input_dim * 8, input_dim * 4, input_dim * 2]
        if hidden_dims is None
        else [int(h) for h in hidden_dims]
    )
    if model_type.lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims,
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )

    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dims=hidden_dims,
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


class DeepKernelBinaryClassificationGPModel(_BaseDeepKernelBinaryClassificationModel):
    """連続入力向け DeepKernel binary classification model。

    Args:
        hidden_dims: DeepKernel feature extractor の隠れ層次元。
            None の場合は従来通り [input_dim * 8, input_dim * 4, input_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 64,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        model_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        train_Y = _prepare_binary_targets(train_Y, train_X)
        input_transform = _to_device_dtype_transform(input_transform, train_X)
        train_X_tf = _make_train_X_tf(
            train_X,
            input_transform,
            name="DeepKernelClassificationGPModel.input_transform",
        )

        if feature_extractor is None:
            feature_extractor = _default_feature_extractor(
                input_dim=train_X.shape[-1],
                model_type=model_type,
                hidden_dims=hidden_dims,
            )
        feature_extractor = feature_extractor.to(train_X)

        inducing_points = _select_inducing_points(
            train_X_tf,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _DeepKernelLatentBinarySVGP(
            inducing_points=inducing_points,
            feature_extractor=feature_extractor,
            train_inputs=train_X_tf,
            train_targets=train_Y,
            mean_module=mean_module,
            covar_module=covar_module,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            input_transform=input_transform,
            train_X=train_X,
            train_Y=train_Y,
        )
        self.model_type = str(model_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]


class DeepKernelBinaryClassificationMixedGPModel(_BaseDeepKernelBinaryClassificationModel):
    """混合入力（連続 + カテゴリ）向け DeepKernel binary classification model。

    Args:
        hidden_dims: 連続変数側 feature extractor の隠れ層次元。
            None の場合は従来通り [cont_dim * 8, cont_dim * 4, cont_dim * 2] を使う。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        likelihood: Optional[BernoulliLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 64,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        model_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        _ = mean_module, covar_module
        if len(cat_dims) == 0:
            raise ValueError("カテゴリ次元を指定する必要があります (cat_dims)。")

        train_Y = _prepare_binary_targets(train_Y, train_X)

        d = train_X.shape[-1]
        cat_dims = list(normalize_indices(indices=cat_dims, d=d))
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims

        input_transform = _to_device_dtype_transform(input_transform, train_X)
        train_X_tf = _make_train_X_tf(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="DeepKernelClassificationMixedGPModel.input_transform",
        )

        if feature_extractor is None:
            if len(ord_dims) > 0:
                feature_extractor = _default_feature_extractor(
                    input_dim=len(ord_dims),
                    model_type=model_type,
                    hidden_dims=hidden_dims,
                )
            else:
                feature_extractor = nn.Identity()
        feature_extractor = feature_extractor.to(train_X)

        inducing_points = _select_inducing_points(
            train_X_tf,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )

        latent_model = _DeepKernelLatentBinaryMixedSVGP(
            inducing_points=inducing_points,
            cat_dims=cat_dims,
            feature_extractor=feature_extractor,
            train_inputs=train_X_tf,
            train_targets=train_Y,
            learn_inducing_locations=learn_inducing_locations,
        )

        likelihood = likelihood or BernoulliLikelihood()

        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            input_transform=input_transform,
            train_X=train_X,
            train_Y=train_Y,
        )
        self.cat_dims = cat_dims
        self.ord_dims = ord_dims
        self._ignore_X_dims_scaling_check = cat_dims
        self.model_type = str(model_type)
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]


__all__ = [
    "DeepKernelBinaryClassificationGPModel",
    "DeepKernelBinaryClassificationMixedGPModel",
]
