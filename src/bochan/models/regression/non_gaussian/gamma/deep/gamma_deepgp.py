from __future__ import annotations

import copy
from math import prod, sqrt
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP
from linear_operator.operators import PsdSumLinearOperator, RootLinearOperator

from bochan.models.components.layers.hidden_layers import (
    DeepGPHiddenLayer,
    DeepMixedGPHiddenLayer,
    DeepKernelDeepGPHiddenLayer,
    DeepKernelDeepMixedGPHiddenLayer,
)
from bochan.models.components.gamma import (
    GammaLink,
    GammaLogLikelihood,
    GammaPosterior,
    apply_input_transform_for_eval,
    apply_input_transform_for_training,
    clone_input_transform,
    get_cont_dims,
    normalize_dims,
    to_device_dtype_transform,
)
from bochan.models.regression.non_gaussian.gamma.base.gamma import (
    apply_outcome_transform_for_training,
    clone_outcome_transform,
)


class _BaseGammaDeepGPModel(DeepGP, GPyTorchModel):
    """Gamma DeepGP wrapper の共通基底。"""

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
        return apply_input_transform_for_eval(
            X,
            self.input_transform,
            cat_dims=getattr(self, "cat_dims", None),
        )

    @staticmethod
    def _moment_match_deepgp_distribution(
        dist: MultivariateNormal,
        X: Tensor,
    ) -> MultivariateNormal:
        """DeepGP の内部サンプル次元をモーメントマッチングで集約する。

        BoTorch の単一出力モデルは、入力 ``batch_shape x q x d`` に対して
        posterior の event shape が ``q x 1`` となることを期待する。一方、
        GPyTorch の DeepGP は likelihood sample 次元を先頭に追加するため、
        その次元を公開 posterior に残さないように集約する。

        各 DeepGP 成分を ``N(mu_s, Sigma_s)`` としたとき、

        ``mu = E_s[mu_s]``
        ``Sigma = E_s[Sigma_s] + Cov_s(mu_s)``

        を用いて単一の MultivariateNormal に近似する。
        """
        mean = dist.mean
        expected_mean_shape = torch.Size(X.shape[:-1])
        extra_ndim = mean.ndim - len(expected_mean_shape)

        if extra_ndim < 0:
            raise RuntimeError(
                "DeepGP posterior mean has fewer dimensions than expected. "
                f"mean.shape={tuple(mean.shape)}, X.shape={tuple(X.shape)}."
            )

        if extra_ndim == 0:
            return dist

        if torch.Size(mean.shape[extra_ndim:]) != expected_mean_shape:
            raise RuntimeError(
                "DeepGP posterior mean cannot be aligned with X. "
                f"mean.shape={tuple(mean.shape)}, X.shape={tuple(X.shape)}, "
                f"extra_ndim={extra_ndim}."
            )

        q = int(X.shape[-2])
        expected_covar_shape = torch.Size((*X.shape[:-2], q, q))
        lazy_covar = dist.lazy_covariance_matrix
        if torch.Size(lazy_covar.shape[extra_ndim:]) != expected_covar_shape:
            raise RuntimeError(
                "DeepGP posterior covariance cannot be aligned with X. "
                f"covariance.shape={tuple(lazy_covar.shape)}, X.shape={tuple(X.shape)}, "
                f"extra_ndim={extra_ndim}."
            )

        component_shape = mean.shape[:extra_ndim]
        n_components = prod(int(size) for size in component_shape)
        component_mean = mean.reshape(n_components, *expected_mean_shape)
        matched_mean = component_mean.mean(dim=0)

        centered_mean = component_mean - matched_mean.unsqueeze(0)
        between_component_root = centered_mean.movedim(0, -1) / sqrt(n_components)
        between_component_covar = RootLinearOperator(between_component_root)

        within_component_covar = lazy_covar
        for _ in range(extra_ndim):
            within_component_covar = within_component_covar.sum(dim=0)
        within_component_covar = within_component_covar * (1.0 / n_components)

        matched_covar = PsdSumLinearOperator(
            within_component_covar,
            between_component_covar,
        )
        return MultivariateNormal(matched_mean, matched_covar)

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
        dist = self._moment_match_deepgp_distribution(dist, X=X_tf)
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
    ) -> Posterior:
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(f"{self.__class__.__name__} does not support tensor observation_noise.")
        latent_post = self.latent_posterior(X, output_indices=output_indices, posterior_transform=None, **kwargs)
        posterior: Posterior = GammaPosterior(
            latent_posterior=latent_post,
            likelihood=self.likelihood,
            add_observation_noise=bool(observation_noise),
        )
        if getattr(self, "outcome_transform", None) is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def predict_mean(self, X: Tensor) -> Tensor:
        return self.posterior(X).mean

    def predict_concentration(self) -> Tensor:
        return self.likelihood.concentration

    def make_mll(self) -> DeepApproximateMLL:
        base_mll = VariationalELBO(
            likelihood=self.likelihood,
            model=self,
            num_data=self.train_inputs_raw[0].shape[-2],
        )
        return DeepApproximateMLL(base_mll)


class DeepGammaGPModel(_BaseGammaDeepGPModel):
    """true DeepGP + Gamma likelihood の正値連続値回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        hidden_dim: int = 4,
        num_inducing: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
        layer_type: str = "default",
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        self.input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        self.outcome_transform = clone_outcome_transform(outcome_transform)
        raw_train_Y, train_Y, self.outcome_transform = apply_outcome_transform_for_training(
            train_Y=train_Y,
            train_X=train_X,
            outcome_transform=self.outcome_transform,
            min_mean=min_mean,
            name="DeepGammaGPModel.outcome_transform",
        )

        train_X_tf = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name="DeepGammaGPModel.input_transform",
        )

        d = train_X_tf.shape[-1]
        if hidden_dims is None:
            hidden_dims = [int(hidden_dim)]
        hidden_dims = [int(h) for h in hidden_dims]

        first_out = hidden_dims[0]
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
        extra_layers = []
        for h in hidden_dims[1:]:
            extra_layers.append(
                DeepGPHiddenLayer(
                    input_dims=current_dim,
                    output_dims=int(h),
                    num_inducing=num_inducing,
                    mean_type=mean_type,
                    input_data=None,
                    learn_inducing_locations=learn_inducing_locations,
                )
            )
            current_dim = int(h)
        self.extra_layers = torch.nn.ModuleList(extra_layers)

        self.last_layer = DeepGPHiddenLayer(
            input_dims=current_dim,
            output_dims=None,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=None,
            learn_inducing_locations=learn_inducing_locations,
        )

        self.likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )

        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.transformed_train_inputs = (train_X_tf.detach().clone(),)
        self.train_targets_raw = raw_train_Y
        self.train_targets = train_Y

        self.hidden_dim = int(hidden_dim)
        self.hidden_dims = list(hidden_dims)
        self.num_inducing = int(num_inducing)
        self.layer_type = str(layer_type)
        self.mean_type = str(mean_type)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)
        self.min_concentration = float(min_concentration)

        self.to(train_X)

    def forward(self, X: Tensor):
        h = self.hidden_layer(X)
        for layer in self.extra_layers:
            h = layer(h)
        return self.last_layer(h)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> "DeepGammaGPModel":
        if kwargs.get("noise") is not None:
            raise NotImplementedError("DeepGammaGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = torch.as_tensor(Y, device=X.device, dtype=X.dtype)
        if Y.ndim > 1 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        Y = Y.clamp_min(self.min_mean)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets_raw, Y], dim=0)
        new_model = self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            hidden_dim=self.hidden_dim,
            num_inducing=self.num_inducing,
            hidden_dims=self.hidden_dims,
            input_transform=clone_input_transform(self.input_transform),
            outcome_transform=clone_outcome_transform(self.outcome_transform),
            likelihood=copy.deepcopy(self.likelihood),
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
            min_concentration=self.min_concentration,
            layer_type=self.layer_type,
            mean_type=self.mean_type,
            learn_inducing_locations=self.learn_inducing_locations,
        )
        new_model.load_state_dict(copy.deepcopy(self.state_dict()), strict=False)
        new_model.eval()
        new_model.likelihood.eval()
        return new_model


class DeepGammaMixedGPModel(_BaseGammaDeepGPModel):
    """mixed 入力版 true DeepGP + Gamma likelihood。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        hidden_dim: int = 4,
        num_inducing: int = 128,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        likelihood: Optional[GammaLogLikelihood] = None,
        link: GammaLink = "softplus",
        init_concentration: float = 10.0,
        learn_concentration: bool = True,
        exp_clip: float = 20.0,
        min_mean: float = 1e-8,
        min_concentration: float = 1e-6,
        layer_type: str = "default",
        mean_type: str = "linear",
        learn_inducing_locations: bool = True,
    ) -> None:
        super().__init__()
        train_X = torch.as_tensor(train_X)
        self.outcome_transform = clone_outcome_transform(outcome_transform)
        raw_train_Y, train_Y, self.outcome_transform = apply_outcome_transform_for_training(
            train_Y=train_Y,
            train_X=train_X,
            outcome_transform=self.outcome_transform,
            min_mean=min_mean,
            name="DeepGammaMixedGPModel.outcome_transform",
        )

        d = train_X.shape[-1]
        self.cat_dims = normalize_dims(cat_dims, d)
        self.cont_dims = get_cont_dims(d, self.cat_dims)

        self.input_transform = to_device_dtype_transform(clone_input_transform(input_transform), train_X)
        train_X_tf = apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name="DeepGammaMixedGPModel.input_transform",
        )

        if str(layer_type).lower() == "deepkernel":
            self.hidden_layer = DeepKernelDeepMixedGPHiddenLayer(
                input_dims=d,
                output_dims=int(hidden_dim),
                ord_dims=self.cont_dims,
                cat_dims=self.cat_dims,
                num_inducing=num_inducing,
                mean_type="constant",
                input_data=train_X_tf,
                learn_inducing_locations=learn_inducing_locations,
            )
        else:
            self.hidden_layer = DeepMixedGPHiddenLayer(
                input_dims=d,
                output_dims=int(hidden_dim),
                ord_dims=self.cont_dims,
                cat_dims=self.cat_dims,
                num_inducing=num_inducing,
                mean_type=mean_type,
                input_data=train_X_tf,
                learn_inducing_locations=learn_inducing_locations,
            )

        self.last_layer = DeepGPHiddenLayer(
            input_dims=int(hidden_dim),
            output_dims=None,
            num_inducing=num_inducing,
            mean_type=mean_type,
            input_data=None,
            learn_inducing_locations=learn_inducing_locations,
        )

        self.likelihood = likelihood or GammaLogLikelihood(
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            exp_clip=exp_clip,
            min_mean=min_mean,
            min_concentration=min_concentration,
        )

        self.train_inputs_raw = (train_X.detach().clone(),)
        self.train_inputs = (train_X,)
        self.transformed_train_inputs = (train_X_tf.detach().clone(),)
        self.train_targets_raw = raw_train_Y
        self.train_targets = train_Y

        self.hidden_dim = int(hidden_dim)
        self.num_inducing = int(num_inducing)
        self.layer_type = str(layer_type)
        self.mean_type = str(mean_type)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.link = link
        self.init_concentration = float(init_concentration)
        self.learn_concentration = bool(learn_concentration)
        self.exp_clip = float(exp_clip)
        self.min_mean = float(min_mean)
        self.min_concentration = float(min_concentration)

        self.to(train_X)

    def forward(self, X: Tensor):
        h = self.hidden_layer(X)
        return self.last_layer(h)


__all__ = [
    "DeepGammaGPModel",
    "DeepGammaMixedGPModel",
]
