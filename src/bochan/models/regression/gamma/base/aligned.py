from __future__ import annotations

import copy
from typing import Any

import torch
from botorch import settings
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.base import MCSampler
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from bochan.models.regression.gamma._components import clone_input_transform, prepare_positive_targets

from .models import GammaGPModel as _GammaGPModel
from .models import GammaMixedGPModel as _GammaMixedGPModel
from .models import GammaPosterior, clone_outcome_transform


def _expand_batch(tensor: Tensor, batch_shape: torch.Size, *, event_ndims: int) -> Tensor:
    """Tensor の batch 次元を broadcast 後の shape に展開する。"""
    if event_ndims < 0 or event_ndims > tensor.ndim:
        raise ValueError(f"Invalid event_ndims={event_ndims} for shape={tuple(tensor.shape)}.")

    event_shape = tensor.shape[-event_ndims:] if event_ndims > 0 else torch.Size()
    source_batch_shape = tensor.shape[: tensor.ndim - event_ndims] if event_ndims > 0 else tensor.shape
    if len(source_batch_shape) > len(batch_shape):
        raise RuntimeError(
            "Cannot expand a tensor with more batch dimensions than the target. "
            f"source_batch_shape={tuple(source_batch_shape)}, target_batch_shape={tuple(batch_shape)}."
        )

    pad = (1,) * (len(batch_shape) - len(source_batch_shape))
    view_shape = torch.Size(pad) + source_batch_shape + event_shape
    return tensor.reshape(view_shape).expand(batch_shape + event_shape)


def _expand_sample_batch(
    tensor: Tensor,
    *,
    sample_shape: torch.Size,
    batch_shape: torch.Size,
    event_ndims: int,
) -> Tensor:
    """先頭の sample 次元を保持しながら batch 次元を展開する。"""
    if tensor.shape[: len(sample_shape)] != sample_shape:
        raise RuntimeError(
            "Fantasy sample dimensions do not match sampler.sample_shape. "
            f"tensor.shape={tuple(tensor.shape)}, sample_shape={tuple(sample_shape)}."
        )

    event_shape = tensor.shape[-event_ndims:] if event_ndims > 0 else torch.Size()
    batch_end = tensor.ndim - event_ndims if event_ndims > 0 else tensor.ndim
    source_batch_shape = tensor.shape[len(sample_shape) : batch_end]
    if len(source_batch_shape) > len(batch_shape):
        raise RuntimeError(
            "Cannot expand fantasy samples to the requested batch shape. "
            f"source_batch_shape={tuple(source_batch_shape)}, target_batch_shape={tuple(batch_shape)}."
        )

    pad = (1,) * (len(batch_shape) - len(source_batch_shape))
    view_shape = sample_shape + torch.Size(pad) + source_batch_shape + event_shape
    return tensor.reshape(view_shape).expand(sample_shape + batch_shape + event_shape)


def _gamma_latent_observation_noise(model, latent_mean: Tensor) -> Tensor:
    """Gamma 観測分散を delta method で latent 空間の分散へ変換する。"""
    likelihood = model.likelihood
    mean = likelihood.mean_from_f(latent_mean)
    concentration = likelihood.concentration.to(device=mean.device, dtype=mean.dtype)

    if likelihood.link == "softplus":
        derivative = torch.sigmoid(latent_mean)
    elif likelihood.link == "exp":
        derivative = mean
    else:  # pragma: no cover - GammaLogLikelihood validates the link.
        raise ValueError(f"Unknown Gamma link: {likelihood.link!r}.")

    eps = torch.finfo(mean.dtype).eps
    observation_variance = mean.square() / concentration.clamp_min(likelihood.min_concentration)
    latent_noise = observation_variance / derivative.square().clamp_min(eps)
    return torch.nan_to_num(
        latent_noise,
        nan=1.0,
        posinf=1.0 / eps,
        neginf=eps,
    ).clamp_min(eps)


def _stable_cholesky(matrix: Tensor) -> Tensor:
    """候補点の条件付け行列に段階的 jitter を加えて Cholesky 分解する。"""
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    eps = torch.finfo(matrix.dtype).eps
    jitter = max(1e-8, 10.0 * eps)

    for _ in range(7):
        chol, info = torch.linalg.cholesky_ex(matrix + jitter * eye)
        if bool(torch.all(info == 0)):
            return chol
        jitter *= 10.0

    return torch.linalg.cholesky(matrix + jitter * eye)


def _apply_posterior_transform(posterior_transform, posterior: Posterior, X: Tensor) -> Posterior:
    """BoTorch バージョン差を吸収して posterior transform を適用する。"""
    try:
        return posterior_transform(posterior, X=X)
    except TypeError:
        return posterior_transform(posterior)


class _GammaFantasyModel(Model):
    """Gamma SVGP の局所 Gaussian 条件付けによる fantasy model。

    Sparse variational GP は exact GP の ``get_fantasy_model`` を利用できないため、
    学習済み latent posterior を candidate 点で Gaussian 条件付けする。Gamma の
    観測ノイズは delta method で latent 空間へ写像する。この近似は候補観測後の
    posterior variance を微分可能に評価するためのもので、モデルを再学習しない。
    """

    def __init__(
        self,
        *,
        base_model,
        fantasy_X: Tensor,
        fantasy_latents: Tensor,
        sample_shape: torch.Size,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.fantasy_X = fantasy_X
        self.fantasy_latents = fantasy_latents
        self.sample_shape = torch.Size(sample_shape)

    @property
    def num_outputs(self) -> int:
        return self.base_model.num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        return self.sample_shape + torch.broadcast_shapes(
            self.base_model.batch_shape,
            self.fantasy_X.shape[:-2],
        )

    def transform_inputs(self, X: Tensor, input_transform=None) -> Tensor:
        if input_transform is None:
            return self.base_model.transform_inputs(X)
        if hasattr(input_transform, "to"):
            input_transform = input_transform.to(device=X.device, dtype=X.dtype)
        return input_transform(X)

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        if output_indices is not None:
            raise NotImplementedError("Gamma fantasy models support a single output only.")
        if torch.is_tensor(observation_noise):
            raise NotImplementedError(
                "Tensor-valued observation_noise is not supported by Gamma fantasy posterior."
            )
        if isinstance(X, tuple):
            X = X[0]

        X = torch.as_tensor(
            X,
            device=self.fantasy_X.device,
            dtype=self.fantasy_X.dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)

        candidate_batch_shape = self.fantasy_X.shape[:-2]
        evaluation_batch_shape = X.shape[:-2]
        common_batch_shape = torch.broadcast_shapes(
            self.base_model.batch_shape,
            candidate_batch_shape,
            evaluation_batch_shape,
        )

        candidate_X = _expand_batch(self.fantasy_X, common_batch_shape, event_ndims=2)
        evaluation_X = _expand_batch(X, common_batch_shape, event_ndims=2)

        self.base_model.eval()
        candidate_X_tf = self.base_model.transform_inputs(candidate_X)
        evaluation_X_tf = self.base_model.transform_inputs(evaluation_X)
        n_candidate = candidate_X_tf.shape[-2]
        n_evaluation = evaluation_X_tf.shape[-2]

        latent_dist = self.base_model.model(torch.cat([candidate_X_tf, evaluation_X_tf], dim=-2))
        latent_mean = latent_dist.mean
        latent_covariance = latent_dist.lazy_covariance_matrix.to_dense()

        candidate_mean = latent_mean[..., :n_candidate]
        evaluation_mean = latent_mean[..., n_candidate:]
        candidate_covariance = latent_covariance[..., :n_candidate, :n_candidate]
        candidate_evaluation_covariance = latent_covariance[..., :n_candidate, n_candidate:]
        evaluation_candidate_covariance = latent_covariance[..., n_candidate:, :n_candidate]
        evaluation_covariance = latent_covariance[..., n_candidate:, n_candidate:]

        latent_noise = _gamma_latent_observation_noise(self.base_model, candidate_mean)
        conditioning_covariance = candidate_covariance + torch.diag_embed(latent_noise)
        chol = _stable_cholesky(conditioning_covariance)

        solved_cross_covariance = torch.cholesky_solve(candidate_evaluation_covariance, chol)
        conditioned_covariance = evaluation_covariance - (
            evaluation_candidate_covariance @ solved_cross_covariance
        )
        conditioned_covariance = 0.5 * (
            conditioned_covariance + conditioned_covariance.transpose(-1, -2)
        )
        covariance_jitter = max(1e-8, 10.0 * torch.finfo(conditioned_covariance.dtype).eps)
        identity = torch.eye(
            n_evaluation,
            device=conditioned_covariance.device,
            dtype=conditioned_covariance.dtype,
        )
        conditioned_covariance = conditioned_covariance + covariance_jitter * identity

        fantasy_latents = self.fantasy_latents
        if fantasy_latents.shape[-1] != 1:
            raise RuntimeError(
                "Gamma fantasy samples must have a singleton output dimension. "
                f"Got shape={tuple(fantasy_latents.shape)}."
            )
        fantasy_latents = fantasy_latents.squeeze(-1)
        fantasy_latents = _expand_sample_batch(
            fantasy_latents,
            sample_shape=self.sample_shape,
            batch_shape=common_batch_shape,
            event_ndims=1,
        )

        sample_prefix = (1,) * len(self.sample_shape)
        candidate_mean = candidate_mean.reshape(sample_prefix + candidate_mean.shape).expand(
            self.sample_shape + common_batch_shape + torch.Size([n_candidate])
        )
        evaluation_mean = evaluation_mean.reshape(sample_prefix + evaluation_mean.shape).expand(
            self.sample_shape + common_batch_shape + torch.Size([n_evaluation])
        )
        chol = chol.reshape(sample_prefix + chol.shape).expand(
            self.sample_shape + common_batch_shape + torch.Size([n_candidate, n_candidate])
        )
        evaluation_candidate_covariance = evaluation_candidate_covariance.reshape(
            sample_prefix + evaluation_candidate_covariance.shape
        ).expand(
            self.sample_shape
            + common_batch_shape
            + torch.Size([n_evaluation, n_candidate])
        )

        residual = fantasy_latents - candidate_mean
        solved_residual = torch.cholesky_solve(residual.unsqueeze(-1), chol)
        conditioned_mean = evaluation_mean + (
            evaluation_candidate_covariance @ solved_residual
        ).squeeze(-1)

        conditioned_covariance = conditioned_covariance.reshape(
            sample_prefix + conditioned_covariance.shape
        ).expand(
            self.sample_shape
            + common_batch_shape
            + torch.Size([n_evaluation, n_evaluation])
        )

        latent_posterior = GPyTorchPosterior(
            MultivariateNormal(
                mean=conditioned_mean,
                covariance_matrix=conditioned_covariance,
            )
        )
        posterior: Posterior = GammaPosterior(
            latent_posterior=latent_posterior,
            likelihood=self.base_model.likelihood,
            add_observation_noise=bool(observation_noise),
        )

        outcome_transform = getattr(self.base_model, "outcome_transform", None)
        if outcome_transform is not None:
            posterior = outcome_transform.untransform_posterior(posterior, X=X)
        if posterior_transform is not None:
            posterior = _apply_posterior_transform(posterior_transform, posterior, X)
        return posterior


class _GammaFantasizeMixin:
    """Gamma SVGP に微分可能な近似 ``fantasize`` を追加する。"""

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Tensor | None = None,
        **kwargs: Any,
    ) -> _GammaFantasyModel:
        if observation_noise is not None:
            raise NotImplementedError(
                "GammaGPModel.fantasize derives observation noise from the fitted Gamma "
                "likelihood and does not accept tensor-valued observation_noise."
            )

        propagate_grads = bool(
            kwargs.pop("propagate_grads", settings.propagate_grads.on())
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected fantasize keyword arguments: {unknown}.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_inputs_raw[0].device,
            dtype=self.train_inputs_raw[0].dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)

        with settings.propagate_grads(propagate_grads):
            latent_posterior = self.latent_posterior(X)
            fantasy_latents = sampler(latent_posterior)

        return _GammaFantasyModel(
            base_model=self,
            fantasy_X=X,
            fantasy_latents=fantasy_latents,
            sample_shape=sampler.sample_shape,
        )


class _AlignedGammaMixin:
    """Align non-Gaussian Gamma wrappers with regression / ordinal conventions."""

    @property
    def batch_shape(self) -> torch.Size:
        """Return the model I/O batch shape expected by BoTorch."""
        return self.train_inputs_raw[0].shape[:-2]

    @property
    def train_inputs(self) -> tuple[Tensor, ...]:
        return self.model.train_inputs

    @train_inputs.setter
    def train_inputs(self, value) -> None:
        self._train_inputs_outer = value

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GammaPosterior:
        return super().posterior(
            X=X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )


class GammaGPModel(_GammaFantasizeMixin, _AlignedGammaMixin, _GammaGPModel):
    """Gamma GP with transformed inputs and approximate fantasy conditioning."""


class GammaMixedGPModel(_GammaFantasizeMixin, _AlignedGammaMixin, _GammaMixedGPModel):
    """Mixed-input Gamma GP with approximate fantasy conditioning."""

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> GammaMixedGPModel:
        if kwargs.get("noise") is not None:
            raise NotImplementedError("GammaMixedGPModel does not support noise in condition_on_observations.")
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(X, device=self.train_inputs_raw[0].device, dtype=self.train_inputs_raw[0].dtype)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        Y = prepare_positive_targets(Y, X, min_value=self.min_mean)
        new_X = torch.cat([self.train_inputs_raw[0], X], dim=-2)
        new_Y = torch.cat([self.train_targets_raw, Y], dim=0)
        return self.__class__(
            train_X=new_X,
            train_Y=new_Y,
            cat_dims=list(self.cat_dims),
            likelihood=copy.deepcopy(self.likelihood),
            input_transform=clone_input_transform(self.input_transform),
            outcome_transform=clone_outcome_transform(self.outcome_transform),
            mean_module=copy.deepcopy(self.model.mean_module),
            covar_module=copy.deepcopy(self.model.covar_module),
            num_inducing_points=self.num_inducing_points,
            inducing_points=self.model.variational_strategy.inducing_points.detach().clone(),
            learn_inducing_locations=self.learn_inducing_locations,
            link=self.link,
            init_concentration=float(self.likelihood.concentration.detach().cpu()),
            learn_concentration=self.learn_concentration,
            exp_clip=self.exp_clip,
            min_mean=self.min_mean,
            min_concentration=self.min_concentration,
        )


__all__ = ["GammaGPModel", "GammaMixedGPModel"]
