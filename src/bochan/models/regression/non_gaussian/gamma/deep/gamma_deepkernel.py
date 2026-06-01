from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.models import ApproximateGP
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from botorch.models.transforms.input import InputTransform

from bochan.models.components.layers.feature_extractor import LargeFeatureExtractor, SkipLargeFeatureExtractor
from bochan.models.components.gamma import (
    GammaLink,
    GammaLogLikelihood,
    apply_input_transform_for_training,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_positive_targets,
    select_inducing_points,
    to_device_dtype_transform,
)
from bochan.models.regression.non_gaussian.gamma import _BaseGammaGPModel, build_mixed_gamma_kernel


def make_gamma_feature_extractor(
    input_dim: int,
    output_dim: Optional[int] = None,
    ext_type: str = "DEFAULT",
    hidden_dims: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Gamma DeepKernel 用 feature extractor を作る。"""
    output_dim = input_dim if output_dim is None else int(output_dim)
    hidden_dims = (
        [input_dim * 8, input_dim * 4, input_dim * 2]
        if hidden_dims is None
        else [int(h) for h in hidden_dims]
    )
    if str(ext_type).lower() == "skip":
        return SkipLargeFeatureExtractor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation="leaky_relu",
            dropout=0.0,
            use_bn=False,
            use_global_skip=True,
        )
    return LargeFeatureExtractor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="leaky_relu",
        dropout=0.0,
        use_bn=False,
    )


class _DeepKernelGammaSVGP(ApproximateGP):
    """Deep Kernel Gamma 用 latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
    ) -> None:
        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        input_dim = train_X.shape[-1]
        self.feature_extractor = feature_extractor or make_gamma_feature_extractor(
            input_dim=input_dim,
            output_dim=input_dim,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
        )
        self.deepkernel = self.feature_extractor
        self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)

        with torch.no_grad():
            z = self.scale_to_bounds(self.deepkernel(train_X[:1]))
        latent_dim = z.shape[-1]

        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=latent_dim)
        ).to(train_X)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        Z = self.scale_to_bounds(self.deepkernel(X))
        return MultivariateNormal(self.mean_module(Z), self.covar_module(Z))


class _DeepKernelMixedGammaSVGP(ApproximateGP):
    """mixed Deep Kernel Gamma 用 latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
    ) -> None:
        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)

        inducing_points = select_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2]
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        if len(self.cont_dims) > 0:
            self.feature_extractor = feature_extractor or make_gamma_feature_extractor(
                input_dim=len(self.cont_dims),
                output_dim=len(self.cont_dims),
                ext_type=ext_type,
                hidden_dims=hidden_dims,
            )
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = ScaleToBounds(-1.0, 1.0)
        else:
            self.feature_extractor = nn.Identity()
            self.deepkernel = self.feature_extractor
            self.scale_to_bounds = nn.Identity()

        self.mean_module = mean_module or ConstantMean()
        self.covar_module = covar_module or build_mixed_gamma_kernel(
            d=d,
            cat_dims=self.cat_dims,
            batch_shape=torch.Size(),
        )
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def _combine_cont_cat(self, X: Tensor) -> Tensor:
        if len(self.cont_dims) == 0:
            return X
        out = torch.empty_like(X)
        z_cont = self.scale_to_bounds(self.deepkernel(X[..., self.cont_dims]))
        out[..., self.cont_dims] = z_cont
        out[..., self.cat_dims] = X[..., self.cat_dims]
        return out

    def forward(self, X: Tensor) -> MultivariateNormal:
        Z = self._combine_cont_cat(X)
        return MultivariateNormal(self.mean_module(Z), self.covar_module(Z))


class DeepKernelGammaGPModel(_BaseGammaGPModel):
    """Deep Kernel Learning 版 Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            name="DeepKernelGammaGPModel.input_transform",
        )
        likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )
        latent_model = _DeepKernelGammaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
            feature_extractor=feature_extractor,
            mean_module=mean_module,
            covar_module=covar_module,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            cat_dims=None,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )
        self.ext_type = ext_type
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]


class DeepKernelGammaMixedGPModel(_BaseGammaGPModel):
    """mixed Deep Kernel Learning 版 Gamma GP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: Optional[GammaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        ext_type: str = "DEFAULT",
        hidden_dims: Optional[Sequence[int]] = None,
        feature_extractor: Optional[nn.Module] = None,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        cat_dims = normalize_dims(cat_dims, train_X.shape[-1])
        train_Y = prepare_positive_targets(train_Y, train_X, min_value=min_mean)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name="DeepKernelGammaMixedGPModel.input_transform",
        )
        likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )
        latent_model = _DeepKernelMixedGammaSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            cat_dims=cat_dims,
            ext_type=ext_type,
            hidden_dims=hidden_dims,
            feature_extractor=feature_extractor,
            mean_module=mean_module,
            covar_module=covar_module,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(
            latent_model=latent_model,
            likelihood=likelihood,
            train_X=train_X,
            train_Y=train_Y,
            input_transform=input_transform,
            cat_dims=cat_dims,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            link=link,
            exp_clip=exp_clip,
            min_mean=min_mean,
        )
        self.ext_type = ext_type
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]


__all__ = [
    "DeepKernelGammaGPModel",
    "DeepKernelGammaMixedGPModel",
    "make_gamma_feature_extractor",
]
