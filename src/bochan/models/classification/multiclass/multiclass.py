from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ProductKernel, ScaleKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from bochan.models.components.multiclass import (
    MulticlassProbsPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    build_default_multiclass_covar_module,
    check_categorical_columns_unchanged,
    clone_input_transform,
    get_cont_dims,
    infer_num_classes,
    normalize_dims,
    prepare_class_targets,
    select_inducing_points,
    to_device_dtype_transform,
)


def _make_cat_kernel(cat_dims: Sequence[int], batch_shape: torch.Size) -> ScaleKernel:
    return ScaleKernel(
        CategoricalKernel(active_dims=tuple(cat_dims), ard_num_dims=len(cat_dims), batch_shape=batch_shape),
        batch_shape=batch_shape,
    )


def _make_cont_kernel(cont_dims: Sequence[int], batch_shape: torch.Size) -> Kernel:
    return get_covar_module_with_dim_scaled_prior(
        batch_shape=batch_shape,
        ard_num_dims=len(cont_dims),
        active_dims=tuple(cont_dims),
    )


def build_mixed_multiclass_kernel(d: int, cat_dims: Sequence[int], *, num_classes: int) -> Kernel:
    """多クラス mixed model 用の continuous + categorical kernel を作る。"""
    cat_dims = normalize_dims(cat_dims, d)
    cont_dims = get_cont_dims(d, cat_dims)
    batch_shape = torch.Size([int(num_classes)])
    if len(cat_dims) == 0:
        return _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    if len(cont_dims) == 0:
        return _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    cont_1 = _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    cont_2 = _make_cont_kernel(cont_dims, batch_shape=batch_shape)
    cat_1 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    cat_2 = _make_cat_kernel(cat_dims, batch_shape=batch_shape)
    return cont_1 + cat_1 + ProductKernel(cont_2, cat_2)


class _LatentMulticlassSVGP(ApproximateGP):
    """多クラス分類用の class-wise latent SVGP。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        inducing_points: Optional[Tensor] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        covar_module: Optional[Kernel] = None,
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
        self.mean_module = mean_module or ConstantMean(batch_shape=batch_shape)
        self.covar_module = covar_module or build_default_multiclass_covar_module(
            train_X,
            num_classes=self.num_classes,
        )
        self.train_inputs = (train_X,)
        self.train_targets = train_Y

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _BaseMulticlassClassificationModel(ApproximateGPyTorchModel):
    """多クラス分類 wrapper の共通基底。"""

    def __init__(
        self,
        *,
        latent_model: ApproximateGP,
        likelihood: SoftmaxLikelihood,
        train_X: Tensor,
        train_Y: Tensor,
        num_classes: int,
        input_transform: Optional[InputTransform],
        cat_dims: Optional[Sequence[int]] = None,
        num_inducing_points: int = 128,
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model=latent_model, likelihood=likelihood, num_outputs=int(num_classes))
        self.num_classes = int(num_classes)
        self.input_transform = input_transform
        self.cat_dims = None if cat_dims is None else list(cat_dims)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X.detach().clone(),)
        self.train_targets = prepare_class_targets(train_Y, train_X, num_classes=self.num_classes)
        self.transformed_train_inputs = (self.model.train_inputs[0].detach().clone(),)
        self.num_inducing_points = int(num_inducing_points)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.temperature = float(temperature)
        self.to(train_X)

    def _set_transformed_inputs(self) -> None:
        return None

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(X, self.input_transform, cat_dims=self.cat_dims)

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(f'{self.__class__.__name__} does not support output_indices.')
        if isinstance(X, tuple):
            X = X[0]
        self.eval()
        X_tf = self.transform_inputs(X)
        latent_dist = self.model(X_tf)
        posterior = GPyTorchPosterior(latent_dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> MulticlassProbsPosterior:
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(f'{self.__class__.__name__} does not support tensor observation_noise.')
        latent_post = self.latent_posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
        posterior = MulticlassProbsPosterior(
            latent_posterior=latent_post,
            num_classes=self.num_classes,
            temperature=self.temperature,
        )
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def class_probs(self, X: Tensor) -> Tensor:
        return self.posterior(X).mean

    def predict_class(self, X: Tensor) -> Tensor:
        return self.class_probs(X).argmax(dim=-1)

    def make_mll(self) -> VariationalELBO:
        return VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.train_inputs_raw[0].shape[-2],
        )


class MulticlassClassificationGPModel(_BaseMulticlassClassificationModel):
    """連続入力用の多クラス SVGP 分類モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
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
            name='MulticlassClassificationGPModel.input_transform',
        )
        latent_model = _LatentMulticlassSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            num_classes=num_classes,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
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

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> 'MulticlassClassificationGPModel':
        if kwargs.get('noise') is not None:
            raise NotImplementedError('MulticlassClassificationGPModel does not support noise in condition_on_observations.')
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_class_targets(Y, X, num_classes=self.num_classes)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            num_classes=self.num_classes,
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            temperature=self.temperature,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        return new_model


class MulticlassClassificationMixedGPModel(_BaseMulticlassClassificationModel):
    """連続 + カテゴリ mixed 入力用の多クラス SVGP 分類モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
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
        if len(cat_dims) == 0:
            raise ValueError('cat_dims must be non-empty for MulticlassClassificationMixedGPModel.')
        num_classes = infer_num_classes(train_Y, num_classes)
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            input_transform,
            cat_dims=cat_dims,
            name='MulticlassClassificationMixedGPModel.input_transform',
        )
        check_categorical_columns_unchanged(train_X, train_X_tf, cat_dims=cat_dims)
        covar_module = covar_module or build_mixed_multiclass_kernel(
            d=d,
            cat_dims=cat_dims,
            num_classes=num_classes,
        )
        latent_model = _LatentMulticlassSVGP(
            train_X=train_X_tf,
            train_Y=train_Y,
            num_classes=num_classes,
            inducing_points=inducing_points,
            num_inducing_points=num_inducing_points,
            learn_inducing_locations=learn_inducing_locations,
            mean_module=mean_module,
            covar_module=covar_module,
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
        self.cat_dims = list(cat_dims)


__all__ = [
    'MulticlassClassificationGPModel',
    'MulticlassClassificationMixedGPModel',
    'build_mixed_multiclass_kernel',
]
