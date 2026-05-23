from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform


def _ensure_q_batch(X: Tensor) -> Tensor:
    if X.ndim == 2:
        return X.unsqueeze(-2)
    return X


def _normalize_pointwise(t: Tensor, X: Tensor, *, name: str) -> Tensor:
    """Normalize posterior mean / variance to (*batch, q_like, 1)."""
    X = _ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    q = X.shape[-2]

    if t.ndim >= 1 and t.shape[-1] == 1:
        if t.shape[:-1] == batch_shape + torch.Size([q]):
            return t
        t_s = t.squeeze(-1)
    else:
        t_s = t

    expected = batch_shape + torch.Size([q])
    if t_s.shape == expected:
        return t_s.unsqueeze(-1)

    if t_s.ndim == len(batch_shape) + 1 and t_s.shape[:-1] == batch_shape:
        q_like = t_s.shape[-1]
        if q_like >= q and q_like % q == 0:
            return t_s.unsqueeze(-1)

    batch_numel = math.prod(batch_shape) if len(batch_shape) > 0 else 1
    if t_s.numel() % batch_numel == 0:
        q_like = t_s.numel() // batch_numel
        if q_like >= q and q_like % q == 0:
            return t_s.reshape(*batch_shape, q_like, 1)

    raise RuntimeError(
        f"{name}: unsupported shape. X.shape={tuple(X.shape)}, t.shape={tuple(t.shape)}"
    )


def _get_noise_posterior(model: Model, X: Tensor):
    if hasattr(model, "posterior_noise") and callable(getattr(model, "posterior_noise")):
        return model.posterior_noise(X)

    if hasattr(model, "noise_posterior") and callable(getattr(model, "noise_posterior")):
        return model.noise_posterior(X)

    noise_model = getattr(model, "noise_model", None)
    if noise_model is None:
        inner = getattr(model, "model", None)
        if inner is not None:
            noise_model = getattr(inner, "noise_model", None)

    if noise_model is None:
        raise AttributeError(
            "Noise posterior was not found. Expected model.posterior_noise(X), "
            "model.noise_posterior(X), model.noise_model.posterior(X), "
            "or model.model.noise_model.posterior(X)."
        )

    return noise_model.posterior(X)


def _get_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    X = _ensure_q_batch(X)
    try:
        noise_post = _get_noise_posterior(model, X)
        noise_mean = _normalize_pointwise(noise_post.mean, X, name="noise_mean")
        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)
        return noise_var.sqrt().clamp_min(eps)
    except Exception:
        post = model.posterior(X)
        mean = _normalize_pointwise(post.mean, X, name="posterior.mean")
        return torch.full_like(mean, float(default_sigma))


def hetero_adjust_regression_samples(
    model: Model,
    X: Tensor,
    samples: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-9,
    posterior=None,
) -> Tensor:
    """
    Heteroscedastic regression 用 robust samples.

    robust = μ + beta * (sample - μ) - noise_penalty * σ_noise
    """
    X = _ensure_q_batch(X)
    if posterior is None:
        posterior = model.posterior(X)

    mean = _normalize_pointwise(posterior.mean, X, name="posterior.mean")
    if samples.ndim >= 1 and samples.shape[-1] != 1:
        samples = samples.unsqueeze(-1)

    sigma_noise = _get_noise_std(
        model,
        X,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        eps=eps,
    )

    if sigma_noise.shape != mean.shape:
        if sigma_noise.numel() == mean.numel():
            sigma_noise = sigma_noise.reshape_as(mean)
        else:
            sigma_noise = sigma_noise.expand_as(mean)

    robust = mean.unsqueeze(0) + float(beta) * (samples - mean.unsqueeze(0))
    robust = robust - float(noise_penalty) * sigma_noise.unsqueeze(0)
    return robust


def compute_hetero_regression_best_f(
    model: Model,
    train_X: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
) -> Tensor:
    with torch.no_grad():
        train_X = _ensure_q_batch(train_X).squeeze(-2)
        post = model.posterior(train_X)
        mean = _normalize_pointwise(post.mean, train_X, name="posterior.mean")
        sigma_noise = _get_noise_std(
            model,
            train_X,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
        )
        robust_y = mean - float(noise_penalty) * sigma_noise
        return robust_y.max().detach()


class _HeteroRegressionBOBase(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        *,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        sampler: Optional[SobolQMCNormalSampler] = None,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        super().__init__(model=model, sampler=sampler, **kwargs)

        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("noise_penalty", torch.as_tensor(noise_penalty))
        self.register_buffer("default_sigma", torch.as_tensor(default_sigma))
        self.noise_is_log_var = bool(noise_is_log_var)

    def _robust_samples(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)
        return hetero_adjust_regression_samples(
            self.model,
            X,
            samples,
            beta=float(self.beta),
            noise_penalty=float(self.noise_penalty),
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            posterior=post,
        )


class qHeteroRegressionUpperConfidenceBound(_HeteroRegressionBOBase):
    """heteroscedastic regression 用 upper confidence bound acquisition。平均と不確実性を組み合わせて探索します。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        robust = self._robust_samples(X)
        best_q = robust.max(dim=-2).values.squeeze(-1)
        return best_q.mean(dim=0)


class qHeteroRegressionExpectedImprovement(_HeteroRegressionBOBase):
    """heteroscedastic regression 用 expected improvement acquisition。現在の best_f からの改善量を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        robust = self._robust_samples(X)
        best_q = robust.max(dim=-2).values.squeeze(-1)
        best_f = self.best_f.to(best_q)
        return (best_q - best_f).clamp_min(0.0).mean(dim=0)


class qHeteroRegressionProbabilityOfImprovement(_HeteroRegressionBOBase):
    """heteroscedastic regression 用 probability of improvement acquisition。best_f を上回る確率を評価します。
    
    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        tau: soft PI や境界近傍重み付けに使う温度・幅パラメータ。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。
    
    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。
    
    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    
    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        *,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        robust = self._robust_samples(X)
        best_q = robust.max(dim=-2).values.squeeze(-1)
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(1e-9)
        return torch.sigmoid((best_q - best_f) / tau).mean(dim=0)

__all__ = [
    "qHeteroRegressionUpperConfidenceBound",
    "qHeteroRegressionExpectedImprovement",
    "qHeteroRegressionProbabilityOfImprovement",
]
