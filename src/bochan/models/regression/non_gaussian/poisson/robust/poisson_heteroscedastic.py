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

from bochan.models.components.poisson import (
    PoissonLink,
    PoissonPosterior,
    align_like,
    ensure_2d_col,
    extract_normalize_only_transform,
    prepare_count_targets,
)
from bochan.models.regression.non_gaussian.poisson import (
    PoissonGPModel,
    PoissonLogLikelihood,
    PoissonMixedGPModel,
)


class HeteroscedasticPoissonPosterior(Posterior):
    """PoissonPosterior に追加の heteroscedastic variance を加える wrapper。

    `mean` は Poisson rate λ のままです。`variance` だけ base variance に
    `extra_noise_var` を足します。
    """

    def __init__(self, base_posterior: PoissonPosterior, extra_noise_var: Optional[Tensor] = None) -> None:
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

    def sample_counts(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        return self.base_posterior.sample_counts(sample_shape=sample_shape)


def fit_variational_poisson_mll(
    model: PoissonGPModel | PoissonMixedGPModel,
    *,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> None:
    """補助 Poisson モデルを簡易 training loop で fit する。"""
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
            loss.backward()
            optimizer.step()

    mll.eval()
    model.eval()
    model.likelihood.eval()


def _fit_noise_model_single(train_X: Tensor, noise_targets: Tensor, input_transform: Optional[InputTransform]) -> SingleTaskGP:
    model = SingleTaskGP(train_X=train_X, train_Y=noise_targets.log(), input_transform=input_transform)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval(); model.likelihood.eval()
    return model


def _fit_noise_model_mixed(
    train_X: Tensor,
    noise_targets: Tensor,
    cat_dims: Sequence[int],
    input_transform: Optional[InputTransform],
) -> MixedSingleTaskGP:
    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=noise_targets.log(),
        cat_dims=list(cat_dims),
        input_transform=input_transform,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval(); model.likelihood.eval()
    return model


def _estimate_poisson_noise_targets(
    model: PoissonGPModel | PoissonMixedGPModel,
    train_X: Tensor,
    train_Y: Tensor,
    *,
    min_noise: float = 1e-6,
) -> Tensor:
    """rate 予測と observed count の残差から追加分散 target を作る。"""
    with torch.no_grad():
        rate = model.predict_rate(train_X)
        if rate.ndim > 1 and rate.shape[-1] == 1:
            rate = rate.squeeze(-1)
        y = prepare_count_targets(train_Y, train_X).to(rate)
        extra = (y - rate).pow(2).clamp_min(float(min_noise))
    return ensure_2d_col(extra)


class _HeteroscedasticPoissonMixin:
    """Poisson heteroscedastic model 用 mixin。"""

    def predict_noise_logvar(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        """追加分散の log variance を予測する。"""
        logvar = self.noise_model.posterior(X).mean
        if ref_like is not None:
            logvar = align_like(logvar, ref_like)
        return logvar

    def predict_noise_var(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        """追加分散を予測する。"""
        return self.predict_noise_logvar(X, ref_like=ref_like).exp().clamp_min(1e-12)

    def predict_noise_std(self, X: Tensor, ref_like: Optional[Tensor] = None) -> Tensor:
        """追加分散の標準偏差を予測する。"""
        return self.predict_noise_var(X, ref_like=ref_like).sqrt()

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool | Tensor = True,
        posterior_transform=None,
        **kwargs: Any,
    ) -> HeteroscedasticPoissonPosterior:
        if torch.is_tensor(observation_noise):
            extra_noise = observation_noise
            add_base_obs_noise = True
        else:
            add_base_obs_noise = True
            extra_noise = None

        base_post = super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=add_base_obs_noise,
            posterior_transform=None,
            **kwargs,
        )

        if not torch.is_tensor(observation_noise) and observation_noise:
            extra_noise = self.predict_noise_var(X, ref_like=base_post.mean)
        elif not torch.is_tensor(observation_noise):
            extra_noise = None

        posterior = HeteroscedasticPoissonPosterior(base_posterior=base_post, extra_noise_var=extra_noise)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


class HeteroscedasticPoissonGPModel(_HeteroscedasticPoissonMixin, PoissonGPModel):
    """追加分散 GP を持つ Poisson 回帰モデル。

    Poisson の基本分散 `Var[y|x]=λ` に加えて、rate 残差から推定した
    extra variance を `observation_noise=True` のときに加えます。
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_count_targets(train_Y, train_X)
        self.min_noise = float(min_noise)
        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size
        self.aux_shuffle = bool(aux_shuffle)

        noise_tf = extract_normalize_only_transform(input_transform)

        if train_Yvar is None:
            aux_model = PoissonGPModel(
                train_X=train_X,
                train_Y=train_Y,
                input_transform=copy.deepcopy(noise_tf),
                num_inducing_points=num_inducing_points,
                link=link,
                exp_clip=exp_clip,
                min_rate=min_rate,
            )
            fit_variational_poisson_mll(aux_model, lr=aux_lr, num_epochs=aux_num_epochs, batch_size=aux_batch_size, shuffle=aux_shuffle)
            noise_targets = _estimate_poisson_noise_targets(aux_model, train_X, train_Y, min_noise=min_noise)
        else:
            noise_targets = ensure_2d_col(torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype)).clamp_min(float(min_noise))

        self.noise_model = _fit_noise_model_single(train_X, noise_targets, copy.deepcopy(noise_tf))
        self.noise_input_transform = noise_tf

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )


class HeteroscedasticPoissonMixedGPModel(_HeteroscedasticPoissonMixin, PoissonMixedGPModel):
    """mixed 入力版の heteroscedastic Poisson 回帰モデル。"""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: Sequence[int],
        likelihood: Optional[PoissonLogLikelihood] = None,
        input_transform: Optional[InputTransform] = None,
        num_inducing_points: int = 128,
        link: PoissonLink = "softplus",
        exp_clip: float = 20.0,
        min_rate: float = 1e-8,
        aux_lr: float = 0.01,
        aux_num_epochs: int = 300,
        aux_batch_size: Optional[int] = None,
        aux_shuffle: bool = True,
        min_noise: float = 1e-6,
        train_Yvar: Optional[Tensor] = None,
    ) -> None:
        train_X = torch.as_tensor(train_X)
        train_Y = prepare_count_targets(train_Y, train_X)
        self.min_noise = float(min_noise)
        self.aux_lr = float(aux_lr)
        self.aux_num_epochs = int(aux_num_epochs)
        self.aux_batch_size = aux_batch_size
        self.aux_shuffle = bool(aux_shuffle)

        noise_tf = extract_normalize_only_transform(input_transform)

        if train_Yvar is None:
            aux_model = PoissonMixedGPModel(
                train_X=train_X,
                train_Y=train_Y,
                cat_dims=cat_dims,
                input_transform=copy.deepcopy(noise_tf),
                num_inducing_points=num_inducing_points,
                link=link,
                exp_clip=exp_clip,
                min_rate=min_rate,
            )
            fit_variational_poisson_mll(aux_model, lr=aux_lr, num_epochs=aux_num_epochs, batch_size=aux_batch_size, shuffle=aux_shuffle)
            noise_targets = _estimate_poisson_noise_targets(aux_model, train_X, train_Y, min_noise=min_noise)
        else:
            noise_targets = ensure_2d_col(torch.as_tensor(train_Yvar, device=train_X.device, dtype=train_X.dtype)).clamp_min(float(min_noise))

        self.noise_model = _fit_noise_model_mixed(train_X, noise_targets, cat_dims, copy.deepcopy(noise_tf))
        self.noise_input_transform = noise_tf

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            likelihood=likelihood,
            input_transform=input_transform,
            num_inducing_points=num_inducing_points,
            link=link,
            exp_clip=exp_clip,
            min_rate=min_rate,
        )


__all__ = [
    "HeteroscedasticPoissonPosterior",
    "HeteroscedasticPoissonGPModel",
    "HeteroscedasticPoissonMixedGPModel",
    "fit_variational_poisson_mll",
]
