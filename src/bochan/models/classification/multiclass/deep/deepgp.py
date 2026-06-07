from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from bochan.models.components.multiclass import (
    MulticlassProbsPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    clone_input_transform,
    get_cont_dims,
    infer_num_classes,
    normalize_dims,
    prepare_class_targets,
    to_device_dtype_transform,
)


class _MulticlassDeepGPLayer(DeepGPLayer):
    """多クラス DeepGP 用の汎用 hidden / output layer。"""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: Optional[int],
        num_inducing: int,
        input_data: Optional[Tensor] = None,
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
    ) -> None:
        self.input_dims = int(input_dims)
        self.output_dims = None if output_dims is None else int(output_dims)
        batch_shape = torch.Size([]) if output_dims is None else torch.Size([int(output_dims)])

        if input_data is not None and input_data.shape[-2] > 0:
            m = min(int(num_inducing), int(input_data.shape[-2]))
            perm = torch.randperm(input_data.shape[-2], device=input_data.device)[:m]
            Z = input_data[perm].detach().clone()
            if output_dims is not None:
                Z = Z.unsqueeze(0).expand(int(output_dims), *Z.shape).contiguous()
        else:
            if output_dims is None:
                Z = torch.randn(int(num_inducing), self.input_dims)
            else:
                Z = torch.randn(int(output_dims), int(num_inducing), self.input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=Z.shape[-2],
            batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=Z,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy, input_dims=input_dims, output_dims=output_dims)

        if mean_type == "linear" and output_dims is None:
            self.mean_module = LinearMean(input_size=input_dims)
        else:
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=input_dims, batch_shape=batch_shape),
            batch_shape=batch_shape,
        )

    def forward(self, X: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(X), self.covar_module(X))


class _BaseMulticlassDeepGPModel(DeepGP, GPyTorchModel):
    """多クラス DeepGP wrapper の共通基底。"""

    _num_outputs = 1

    def _set_transformed_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> int:
        return self.num_classes

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=getattr(self, "cat_dims", None),
        )

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices is not None:
            raise NotImplementedError(f"{self.__class__.__name__} does not support output_indices.")
        if isinstance(X, tuple):
            X = X[0]
        self.eval()
        X_tf = self.transform_inputs(X)
        dist = self(X_tf)
        posterior = GPyTorchPosterior(dist)
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
            raise NotImplementedError(f"{self.__class__.__name__} does not support tensor observation_noise.")
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

    def make_mll(self) -> DeepApproximateMLL:
        base_mll = VariationalELBO(
            likelihood=self.likelihood,
            model=self,
            num_data=self.train_inputs_raw[0].shape[-2],
        )
        return DeepApproximateMLL(base_mll)


class MulticlassDeepGPModel(_BaseMulticlassDeepGPModel):
    """true DeepGP + SoftmaxLikelihood の多クラス分類モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: Optional[int] = None,
        hidden_dim: int = 4,
        num_inducing: int = 128,
        list_hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        num_classes = infer_num_classes(train_Y, num_classes)
        train_Y = prepare_class_targets(train_Y, train_X, num_classes=num_classes)
        self.num_classes = int(num_classes)
        self.input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name="MulticlassDeepGPModel.input_transform",
        )

        d = train_X_tf.shape[-1]
        if list_hidden_dims is None:
            list_hidden_dims = [int(hidden_dim)]
        list_hidden_dims = [int(h) for h in list_hidden_dims]

        self.hidden_layer = _MulticlassDeepGPLayer(
            input_dims=d,
            output_dims=list_hidden_dims[0],
            num_inducing=num_inducing,
            input_data=train_X_tf,
            mean_type=mean_type,
            learn_inducing_locations=learn_inducing_locations,
        )
        current_dim = list_hidden_dims[0]
        extra_layers = []
        for h in list_hidden_dims[1:]:
            extra_layers.append(
                _MulticlassDeepGPLayer(
                    input_dims=current_dim,
                    output_dims=int(h),
                    num_inducing=num_inducing,
                    input_data=None,
                    mean_type=mean_type,
                    learn_inducing_locations=learn_inducing_locations,
                )
            )
            current_dim = int(h)
        self.extra_layers = torch.nn.ModuleList(extra_layers)
        self.last_layer = _MulticlassDeepGPLayer(
            input_dims=current_dim,
            output_dims=self.num_classes,
            num_inducing=num_inducing,
            input_data=None,
            mean_type="constant",
            learn_inducing_locations=learn_inducing_locations,
        )
        self.likelihood = likelihood or SoftmaxLikelihood(
            num_features=self.num_classes,
            num_classes=self.num_classes,
            mixing_weights=False,
        )
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X.detach().clone(),)
        self.transformed_train_inputs = (train_X_tf.detach().clone(),)
        self.train_targets = train_Y
        self.hidden_dim = int(hidden_dim)
        self.list_hidden_dims = list(list_hidden_dims)
        self.num_inducing = int(num_inducing)
        self.mean_type = str(mean_type)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.temperature = float(temperature)
        self.to(train_X)

    def forward(self, X: Tensor):
        h = self.hidden_layer(X)
        for layer in self.extra_layers:
            h = layer(h)
        return self.last_layer(h)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "MulticlassDeepGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("MulticlassDeepGPModel does not support noise in condition_on_observations.")
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
            hidden_dim=self.hidden_dim,
            num_inducing=self.num_inducing,
            list_hidden_dims=self.list_hidden_dims,
            input_transform=clone_input_transform(self.input_transform),
            likelihood=copy.deepcopy(self.likelihood),
            mean_type=self.mean_type,
            learn_inducing_locations=self.learn_inducing_locations,
            temperature=self.temperature,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class MulticlassMixedDeepGPModel(MulticlassDeepGPModel):
    """mixed 入力版の多クラス DeepGP。

    Notes:
        カテゴリ列は input_transform で変更しないことを検査します。
        ここでは DeepGP layer 自体は数値入力として扱います。カテゴリ専用 kernel を使う
        true mixed DeepGP が必要な場合は components.layers の mixed hidden layer へ置き換えてください。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        num_classes: Optional[int] = None,
        hidden_dim: int = 4,
        num_inducing: int = 128,
        list_hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[SoftmaxLikelihood] = None,
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
        temperature: float = 1.0,
    ) -> None:
        train_X_tmp = torch.as_tensor(train_X)
        self.cat_dims = normalize_dims(cat_dims, train_X_tmp.shape[-1])
        self.cont_dims = get_cont_dims(train_X_tmp.shape[-1], self.cat_dims)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_inducing=num_inducing,
            list_hidden_dims=list_hidden_dims,
            input_transform=input_transform,
            likelihood=likelihood,
            mean_type=mean_type,
            learn_inducing_locations=learn_inducing_locations,
            temperature=temperature,
        )


__all__ = [
    "MulticlassDeepGPModel",
    "MulticlassMixedDeepGPModel",
]
