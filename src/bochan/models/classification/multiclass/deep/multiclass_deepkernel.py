from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from botorch.models.transforms.input import InputTransform
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from bochan.models.components.multiclass import (
    apply_input_transform_for_training,
    clone_input_transform,
    get_cont_dims,
    infer_num_classes,
    normalize_dims,
    prepare_class_targets,
    select_inducing_points,
    to_device_dtype_transform,
)
from bochan.models.classification.multiclass import (
    _BaseMulticlassClassificationModel,
    build_mixed_multiclass_kernel,
)


class _DefaultFeatureExtractor(nn.Module):
    """DeepKernel 用の簡易 MLP feature extractor。"""

    def __init__(self, input_dim: int, output_dim: Optional[int] = None) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else int(output_dim)
        h1 = max(8, input_dim * 8)
        h2 = max(8, input_dim * 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LeakyReLU(),
            nn.Linear(h2, output_dim),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)


def make_multiclass_feature_extractor(
    input_dim: int,
    output_dim: Optional[int] = None,
    ext_type: str = "DEFAULT",
) -> nn.Module:
    """多クラス DeepKernel 用 feature extractor を作る。"""
    # ユーザー環境に components.layers がある場合は、それを優先する。
    try:
        from bochan.models.components.layers.feature_extractor import (  # type: ignore
            LargeFeatureExtractor,
            SkipLargeFeatureExtractor,
        )

        output_dim = input_dim if output_dim is None else int(output_dim)
        if str(ext_type).lower() == "skip":
            return SkipLargeFeatureExtractor(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
                activation="leaky_relu",
                dropout=0.0,
                use_bn=False,
                use_global_skip=True,
            )
        return LargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[input_dim * 8, input_dim * 4, input_dim * 2],
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
        )
    except Exception:
        return _DefaultFeatureExtractor(input_dim=input_dim, output_dim=output_dim)


class _DeepKernelMulticlassSVGP(ApproximateGP):
    """Deep Kernel 多クラス分類用の class-wise latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.num_classes = int(num_classes)
        batch_shape = torch.Size([self.num_classes])
        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            num_classes=self.num_classes,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        input_dim = train_X.shape[-1]
        self.feature_extractor = feature_extractor or make_multiclass_feature_extractor(
            input_dim=input_dim,
            output_dim=input_dim,
            ext_type=ext_type,
        )
        self.deepkernel = self.feature_extractor
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)

        with torch.no_grad():
            z = self.scale_to_bounds(self.deepkernel(train_X[:1]))
        latent_dim = z.shape[-1]

        self.mean_module = mean_module or ConstantMean(batch_shape=batch_shape)
        self.covar_module = covar_module or ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=latent_dim, batch_shape=batch_shape),
            batch_shape=batch_shape,
        ).to(train_X)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        Z = self.scale_to_bounds(self.deepkernel(X))
        return MultivariateNormal(self.mean_module(Z), self.covar_module(Z))


class _DeepKernelMixedMulticlassSVGP(ApproximateGP):
    """mixed Deep Kernel 多クラス分類用の class-wise latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: int,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
    ) -> None:
        self.num_classes = int(num_classes)
        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)
        batch_shape = torch.Size([self.num_classes])

        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
            num_classes=self.num_classes,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        if len(self.cont_dims) > 0:
            self.feature_extractor = feature_extractor or make_multiclass_feature_extractor(
                input_dim=len(self.cont_dims),
                output_dim=len(self.cont_dims),
                ext_type=ext_type,
            )
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        else:
            self.feature_extractor = nn.Identity()
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = nn.Identity()

        self.mean_module = mean_module or ConstantMean(batch_shape=batch_shape)
        self.covar_module = covar_module or build_mixed_multiclass_kernel(
            d=d,
            cat_dims=self.cat_dims,
            num_classes=self.num_classes,
        )
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def _combine_cont_cat(self, X: Tensor) -> Tensor:
        if len(self.cont_dims) == 0:
            return X
        out = torch.empty_like(X)
        out[..., self.cont_dims] = self.scale_to_bounds(self.deepkernel(X[..., self.cont_dims]))
        out[..., self.cat_dims] = X[..., self.cat_dims]
        return out

    def forward(self, X: Tensor) -> MultivariateNormal:
        Z = self._combine_cont_cat(X)
        return MultivariateNormal(self.mean_module(Z), self.covar_module(Z))


class DeepKernelMulticlassClassificationGPModel(_BaseMulticlassClassificationModel):
    """Deep Kernel Learning 版の多クラス分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        num_classes = infer_num_classes(train_Y, num_classes)
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            name="DeepKernelMulticlassClassificationGPModel.input_transform",
        )
        latent_model = _DeepKernelMulticlassSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            num_classes=num_classes,
            ext_type=ext_type,
            feature_extractor=feature_extractor,
            mean_module=mean_module,
            covar_module=covar_module,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        likelihood = likelihood or SoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            mixing_weights=False,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            input_transform=input_transform,
            cat_dims=None,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )
        self.ext_type = str(ext_type)


class DeepKernelMulticlassClassificationMixedGPModel(_BaseMulticlassClassificationModel):
    """mixed Deep Kernel Learning 版の多クラス分類 GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        d = train_X.shape[-1]
        cat_dims = normalize_dims(cat_dims, d)
        num_classes = infer_num_classes(train_Y, num_classes)
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="DeepKernelMulticlassClassificationMixedGPModel.input_transform",
        )
        latent_model = _DeepKernelMixedMulticlassSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            cat_dims=cat_dims,
            num_classes=num_classes,
            ext_type=ext_type,
            feature_extractor=feature_extractor,
            mean_module=mean_module,
            covar_module=covar_module,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        likelihood = likelihood or SoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            mixing_weights=False,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            input_transform=input_transform,
            cat_dims=cat_dims,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )
        self.ext_type = str(ext_type)


__all__ = [
    "DeepKernelMulticlassClassificationGPModel",
    "DeepKernelMulticlassClassificationMixedGPModel",
    "make_multiclass_feature_extractor",
]
