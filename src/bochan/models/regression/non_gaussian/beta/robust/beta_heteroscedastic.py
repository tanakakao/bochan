from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.components.beta import (
    BetaMeanLink,
    BetaPosterior,
    align_like,
    ensure_2d_col,
    extract_normalize_only_transform,
    prepare_beta_targets,
)
from bochan.models.regression.non_gaussian.beta import BetaGPModel, BetaLogLikelihood, BetaMixedGPModel


class HeteroscedasticBetaPosterior(Posterior):
    """BetaPosterior に追加の heteroscedastic variance を加える wrapper。"""

    def __init__(self, base_posterior: BetaPosterior, extra_noise_var: Optional[Tensor] = None) -> None:
        super().__init__()
        self.base_posterior = base_posterior
        self.extra_noise_var = extra_noise_var

    @property
    def device(self) -> torch.device:
        return self.base_posterior.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base_posterior.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self.base_posterior.event_shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.base_posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self.base_posterior.batch_range

    @property
    def mean(self) -> Tensor:
        return self.base_posterior.mean

    @property
    def variance(self) -> Tensor:
        var = self.base_posterior.variance
        if self.extra_noise_var is None:
            return var
        return var + align_like(self.extra_noise_var, var)

    def rsample(self, sample_shape: Optional[torch.Size] = None, base_samples: Optional[Tensor] = None) -> Tensor:
        return self.base_posterior.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def sample_observations(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        return self.base_posterior.sample_observations(sample_shape=sample_shape)


def _fit_variational_beta_mll(
    model: BetaGPModel | BetaMixedGPModel,
    *,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> None:
    """補助 Beta モデルを簡易 training loop で fit する。"""
    mll = model.make_mll()
    mll.train()
    x_tensor = model.model.train_inputs[0]
    y_tensor = model.model.train_targets
    if y_tensor.ndim > 1 and y_tensor.shape[-1] == 1:
        y_tensor = y_tensor.squeeze(-1)
    if batch_size is None:
        batch_size = x_tensor.shape[-2]
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    optimizer = torch.optim.Adam(mll.parameters(), lr=float(lr))
    for _ in range(int(num_epochs)):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = mll.model(xb)
            loss = -mll(output, yb)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
    mll.eval()
    model.eval()
    model.likelihood.eval()


def _fit_noise_model_single(train_X: Tensor, noise_targets: Tensor, input_transform: Optional[InputTransform]) -> SingleTaskGP:
    model = SingleTaskGP(train_X=train_X, train_Y=noise_targets.log(), input_transform=input_transform)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def _fit_noise_model_mixed(train_X: Tensor, noise_targets: Tensor, cat_dims: Sequence[int], input_transform: Optional[InputTransform]) -> MixedSingleTaskGP:
    model = MixedSingleTaskGP(train_X=train_X, train_Y=noise_targets.log(), cat_dims=list(cat_dims), input_transform=input_transform)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model


def _estimate_beta_noise_targets(model: BetaGPModel | BetaMixedGPModel, train_X: Tensor, train_Y: Tensor, *, eps: float, min_noise: float = 1e-6) -> Tensor:
    """Beta mean 予測と observed proportion の残差から追加分散 target を作る。"""
    with torch.no_grad():
        mean = model.predict_mean(train_X)
        if mean.ndim > 1 and mean.shape[-1] == 1:
            mean = mean.squeeze(-1)
        y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=True).to(mean)
        extra = (y - mean).pow(2).clamp_min(float(min_noise))
    return ensure_2d_col(extra)


class _HeteroscedasticBetaMixin:
    """Beta heteroscedastic model 用 mixin。"""

    def predict_noise_logvar(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        logvar = self.noise_model.posterior(X).mean
        if ref_like is not None:
            logvar = align_like(logvar, ref_like)
        return logvar

    def predict_noise_var(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        return self.predict_noise_logvar(X, ref_like=ref_like).exp().clamp_min(1e-12)

    def predict_noise_std(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        return self.predict_noise_var(X, ref_like=ref_like).sqrt()

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = True,
        posterior_transform=None,
        **kwargs: Any,
    ) -> HeteroscedasticBetaPosterior:
        if torch.is_tensor(observation_noise):
            extra_noise = observation_noise
            base_post = super().posterior(X, output_indices=output_indices, observation_noise=True, posterior_transform=None, **kwargs)
        else:
            base_post = super().posterior(X, output_indices=output_indices, observation_noise=True, posterior_transform=None, **kwargs)
            extra_noise = self.predict_noise_var(X, ref_like=base_post.mean) if observation_noise else None
        posterior = HeteroscedasticBetaPosterior(base_posterior=base_post, extra_noise_var=extra_noise)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class HeteroscedasticBetaGPModel(_HeteroscedasticBetaMixin, BetaGPModel):
    """追加分散 GP を持つ Beta 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: Optional[BetaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        self.min_noise = float(min_noise)
        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size
        self.aux_shuffle = bool(aux_shuffle)
        noise_tf = extract_normalize_only_transform(input_transform)
        if train_Yvar is None:
            aux_model = BetaGPModel(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=copy.deepcopy(noise_tf),
                num_inducing_points=num_inducing_points,
                link=link,
                init_concentration=init_concentration,
                learn_concentration=learn_concentration,
                eps=eps,
                min_concentration=min_concentration,
                clip_targets=clip_targets,
            )
            _fit_variational_beta_mll(aux_model, lr=aux_lr, num_epochs=aux_num_epochs, batch_size=aux_batch_size, shuffle=aux_shuffle)
            noise_targets = _estimate_beta_noise_targets(aux_model, train_X, train_Y, eps=eps, min_noise=min_noise)
        else:
            noise_targets = ensure_2d_col(torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype)).clamp_min(float(min_noise))
        self.noise_model = _fit_noise_model_single(train_X=train_X, noise_targets=noise_targets, input_transform=copy.deepcopy(noise_tf))
        self.noise_input_transform = noise_tf
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
            clip_targets=clip_targets,
        )


class HeteroscedasticBetaMixedGPModel(_HeteroscedasticBetaMixin, BetaMixedGPModel):
    """mixed 入力版の heteroscedastic Beta 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: Optional[BetaLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: BetaMeanLink = "sigmoid",
        init_concentration: float = 20.0,
        learn_concentration: bool = True,
        eps: float = 1e-6,
        min_concentration: float = 1e-6,
        clip_targets: bool = True,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_beta_targets(train_Y, train_X, eps=eps, clip=clip_targets)
        self.min_noise = float(min_noise)
        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size
        self.aux_shuffle = bool(aux_shuffle)
        noise_tf = extract_normalize_only_transform(input_transform)
        if train_Yvar is None:
            aux_model = BetaMixedGPModel(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                input_transform=copy.deepcopy(noise_tf),
                num_inducing_points=num_inducing_points,
                link=link,
                init_concentration=init_concentration,
                learn_concentration=learn_concentration,
                eps=eps,
                min_concentration=min_concentration,
                clip_targets=clip_targets,
            )
            _fit_variational_beta_mll(aux_model, lr=aux_lr, num_epochs=aux_num_epochs, batch_size=aux_batch_size, shuffle=aux_shuffle)
            noise_targets = _estimate_beta_noise_targets(aux_model, train_X, train_Y, eps=eps, min_noise=min_noise)
        else:
            noise_targets = ensure_2d_col(torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype)).clamp_min(float(min_noise))
        self.noise_model = _fit_noise_model_mixed(train_X=train_X, noise_targets=noise_targets, cat_dims=cat_dims, input_transform=copy.deepcopy(noise_tf))
        self.noise_input_transform = noise_tf
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            likelihood=likelihood,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            link=link,
            init_concentration=init_concentration,
            learn_concentration=learn_concentration,
            eps=eps,
            min_concentration=min_concentration,
            clip_targets=clip_targets,
        )


__all__ = ["HeteroscedasticBetaPosterior", "HeteroscedasticBetaGPModel", "HeteroscedasticBetaMixedGPModel"]
