from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP

from bochan.models.components.layers.hidden_layers import (
    DeepGPHiddenLayer,
    DeepMixedGPHiddenLayer,
    DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer,
)
from bochan.models.components.beta import (
    BetaMeanLink,
    BetaLogLikelihood,
    BetaPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    prepare_beta_targets,
    to_device_dtype_transform,
)


class _BaseBetaDeepGPModel(DeepGP, GPyTorchModel):
    """Beta DeepGP wrapper の共通基底。"""

    _num_outputs = 1

    def _set_transformed_inputs(self) -> None:
        return None

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def transform_inputs(self, X: Tensor) -> Tensor:
        return apply_input_transform_for_eval(X, self.input_transform, cat_dims=getattr(self, "cat_dims", None))

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
        observation_noise: bool | Tensor = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> BetaPosterior:
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(f"{self.__class__.__name__} does not support tensor observation_noise.")
        latent_post = self.latent_posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
        posterior = BetaPosterior(latent_posterior=latent_post, likelihood=self.likelihood, add_observation_noise=bool(observation_noise))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def predict_mean(self, X: Tensor) -> Tensor:
        return self.posterior(X).mean

    def predict_concentration(self) -> Tensor:
        return self.likelihood.concentration

    def predict_beta_params(self, X: Tensor):
        mu = self.predict_mean(X).clamp(min=self.eps, max=1.0 - self.eps)
        phi = self.predict_concentration().to(device=mu.device, dtype=mu.dtype)
        return (mu * phi).clamp_min(self.eps), ((1.0 - mu) * phi).clamp_min(self.eps)

    def make_mll(self) -> DeepApproximateMLL:
        base_mll = VariationalELBO(likelihood=self.likelihood, model=self, num_data=self.train_inputs_raw[0].shape[-2])
        return DeepApproximateMLL(base_mll)


class BetaDeepGPModel(_BaseBetaDeepGPModel):
    """true DeepGP + Beta likelihood の割合回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        hidden_dim: int = 4,
        num_inducing: int = 128,
        list_hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[BetaLogLikelihood] = None,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
        layer_type: str = "default",
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        self.input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(train_X, self.input_transform, name="BetaDeepGPModel.input_transform")
        d = train_X_tf.shape[-1]
        if list_hidden_dims is None:
            list_hidden_dims = [int(hidden_dim)]
        list_hidden_dims = [int(h) for h in list_hidden_dims]
        first_out = list_hidden_dims[0]
        if str(layer_type).lower() == "deepkernel":
            self.hidden_layer = DeepKernelDeepGPHiddenLayer(
                input_dims=d,
                output_dims=first_out,
                num_inducing=num_inducing,
                mean_type="constant",
                input_data=train_X_tf,
                learn_inducing_locations=learn_inducing_locations,
            )
        else:
            self.hidden_layer = DeepGPHiddenLayer(
                input_dims=d,
                output_dims=first_out,
                num_inducing=num_inducing,
                mean_type=mean_type,
                input_data=train_X_tf,
                learn_inducing_locations=learn_inducing_locations,
            )
        current_dim = first_out
        layers = []
        for h in list_hidden_dims[1:]:
            layers.append(DeepGPHiddenLayer(input_dims=current_dim, output_dims=int(h), num_inducing=num_inducing, mean_type=mean_type, input_data=None, learn_inducing_locations=learn_inducing_locations))
            current_dim = int(h)
        self.extra_layers = torch.nn.ModuleList(layers)
        self.last_layer = DeepGPHiddenLayer(input_dims=current_dim, output_dims=None, num_inducing=num_inducing, mean_type=mean_type, input_data=None, learn_inducing_locations=learn_inducing_locations)
        self.likelihood = likelihood or BetaLogLikelihood(link=link, init_concentration=init_concentration, learn_concentration=learn_concentration, eps=eps, min_concentration=min_concentration)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.transformed_train_inputs = (train_X_tf.detach().clone(),)
        self.train_targets = train_Y
        self.hidden_dim = int(hidden_dim)
        self.list_hidden_dims = list(list_hidden_dims)
        self.num_inducing = int(num_inducing)
        self.layer_type = str(layer_type)
        self.mean_type = str(mean_type)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.eps = float(eps)
        self.min_concentration = float(min_concentration)
        self.clip_targets = bool(clip_targets)
        self.to(train_X)

    def forward(self, X: Tensor):
        h = self.hidden_layer(X)
        for layer in self.extra_layers:
            h = layer(h)
        return self.last_layer(h)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "BetaDeepGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("BetaDeepGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_beta_targets(Y, X, eps=self.eps, clip=self.clip_targets)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets, Y], dim=0)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            hidden_dim=self.hidden_dim,
            num_inducing=self.num_inducing,
            list_hidden_dims=self.list_hidden_dims,
            input_transform=clone_input_transform(self.input_transform),
            likelihood=copy.deepcopy(self.likelihood),
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            eps=self.eps,
            min_concentration=self.min_concentration,
            clip_targets=self.clip_targets,
            layer_type=self.layer_type,
            mean_type=self.mean_type,
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class BetaMixedDeepGPModel(_BaseBetaDeepGPModel):
    """mixed 入力版 true DeepGP + Beta likelihood。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        hidden_dim: int = 4,
        num_inducing: int = 128,
        input_transform: Optional[InputTransform] = None,
        likelihood: Optional[BetaLogLikelihood] = None,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
        layer_type: str = "default",
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)
        self.input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(train_X, self.input_transform, cat_dims=self.cat_dims, name="BetaMixedDeepGPModel.input_transform")
        if str(layer_type).lower() == "deepkernel":
            self.hidden_layer = DeepKernelDeepMixedGPHiddenLayer(input_dims=d, output_dims=int(hidden_dim), ord_dims=self.cont_dims, cat_dims=self.cat_dims, num_inducing=num_inducing, mean_type="constant", input_data=train_X_tf, learn_inducing_locations=learn_inducing_locations)
        else:
            self.hidden_layer = DeepMixedGPHiddenLayer(input_dims=d, output_dims=int(hidden_dim), ord_dims=self.cont_dims, cat_dims=self.cat_dims, num_inducing=num_inducing, mean_type=mean_type, input_data=train_X_tf, learn_inducing_locations=learn_inducing_locations)
        self.last_layer = DeepGPHiddenLayer(input_dims=int(hidden_dim), output_dims=None, num_inducing=num_inducing, mean_type=mean_type, input_data=None, learn_inducing_locations=learn_inducing_locations)
        self.likelihood = likelihood or BetaLogLikelihood(link=link, init_concentration=init_concentration, learn_concentration=learn_concentration, eps=eps, min_concentration=min_concentration)
        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.transformed_train_inputs = (train_X_tf.detach().clone(),)
        self.train_targets = train_Y
        self.hidden_dim = int(hidden_dim)
        self.num_inducing = int(num_inducing)
        self.layer_type = str(layer_type)
        self.mean_type = str(mean_type)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.eps = float(eps)
        self.min_concentration = float(min_concentration)
        self.clip_targets = bool(clip_targets)
        self.to(train_X)

    def forward(self, X: Tensor):
        return self.last_layer(self.hidden_layer(X))


__all__ = ["BetaDeepGPModel", "BetaMixedDeepGPModel"]
