from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform

from ._utils import (
    ensure_q_batch,
    normalize_binary_mean_shape,
    reshape_binary_samples,
    to_probability,
)


def _get_noise_posterior(model: Model, X: Tensor):
    """Return posterior of the heteroscedastic noise model."""
    if hasattr(model, "posterior_noise") and callable(getattr(model, "posterior_noise")):
        return model.posterior_noise(X)

    if hasattr(model, "noise_posterior") and callable(getattr(model, "noise_posterior")):
        return model.noise_posterior(X)

    noise_model = getattr(model, "noise_model", None)
    if noise_model is None:
        inner_model = getattr(model, "model", None)
        if inner_model is not None:
            noise_model = getattr(inner_model, "noise_model", None)

    if noise_model is None:
        raise AttributeError(
            "Noise posterior was not found. Expected model.posterior_noise(X), "
            "model.noise_posterior(X), model.noise_model.posterior(X), "
            "or model.model.noise_model.posterior(X)."
        )

    return noise_model.posterior(X)


def _align_pointwise_to_reference(
    value: Tensor,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    """Align a pointwise tensor to the shape of a reference tensor."""
    if value.ndim >= 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)

    if value.shape == reference.shape:
        return value

    ref_shape = reference.shape
    batch_shape = ref_shape[:-1]
    q_ref = ref_shape[-1]

    if value.shape == batch_shape:
        return value.unsqueeze(-1).expand_as(reference)

    if value.ndim == len(ref_shape) and value.shape[:-1] == batch_shape:
        q_value = value.shape[-1]
        if q_ref % q_value == 0:
            return value.repeat_interleave(q_ref // q_value, dim=-1)
        if q_value % q_ref == 0:
            return value.reshape(*batch_shape, q_ref, q_value // q_ref).mean(dim=-1)

    if value.numel() == reference.numel():
        return value.reshape_as(reference)

    raise RuntimeError(
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}"
    )


def _get_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Return noise standard deviation for heteroscedastic classification."""
    X = ensure_q_batch(X)
    try:
        noise_post = _get_noise_posterior(model, X)
        noise_mean = normalize_binary_mean_shape(noise_post.mean, X)

        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)

        return noise_var.sqrt().clamp_min(eps)

    except Exception:
        post = model.posterior(X)
        mean = normalize_binary_mean_shape(post.mean, X)
        return torch.full_like(mean, float(default_sigma))


def hetero_adjust_binary_classification_samples(
    model: Model,
    X: Tensor,
    samples: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    samples_are_probs: bool = True,
    apply_sigmoid_if_needed: bool = False,
    eps: float = 1e-6,
    posterior=None,
) -> Tensor:
    """Apply heteroscedastic noise adjustment to binary classification samples."""
    X = ensure_q_batch(X)

    if posterior is None:
        posterior = model.posterior(X)

    mean_prob = normalize_binary_mean_shape(
        posterior.mean,
        X,
        perturbation_reduction="mean",
    )
    mean_prob = to_probability(
        mean_prob,
        apply_sigmoid_if_needed=apply_sigmoid_if_needed,
        eps=eps,
        name="posterior.mean",
    )

    samples = reshape_binary_samples(
        samples,
        X,
        perturbation_reduction="mean",
    )
    samples = to_probability(
        samples,
        apply_sigmoid_if_needed=(not samples_are_probs) or apply_sigmoid_if_needed,
        eps=eps,
        name="posterior samples",
    )

    sigma_noise = _get_noise_std(
        model,
        X,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        eps=eps,
    )
    sigma_noise = _align_pointwise_to_reference(
        sigma_noise,
        mean_prob,
        name="sigma_noise",
    )

    adjusted = mean_prob.unsqueeze(0) + float(beta) * (
        samples - mean_prob.unsqueeze(0)
    )
    adjusted = adjusted - float(noise_penalty) * sigma_noise.unsqueeze(0)

    return adjusted.clamp(eps, 1.0 - eps)


def compute_hetero_binary_classification_best_f(
    model: Model,
    train_X: Tensor,
    *,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    apply_sigmoid_if_needed: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Compute best_f for heteroscedastic binary classification acquisition."""
    train_X = ensure_q_batch(train_X).squeeze(-2)

    with torch.no_grad():
        post = model.posterior(train_X)
        mean_prob = normalize_binary_mean_shape(post.mean, train_X)
        mean_prob = to_probability(
            mean_prob,
            apply_sigmoid_if_needed=apply_sigmoid_if_needed,
            eps=eps,
            name="posterior.mean",
        )

        sigma_noise = _get_noise_std(
            model,
            train_X,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            eps=eps,
        )
        sigma_noise = _align_pointwise_to_reference(
            sigma_noise,
            mean_prob,
            name="sigma_noise",
        )

        return (
            mean_prob - float(noise_penalty) * sigma_noise
        ).clamp(eps, 1.0 - eps).max()


class _HeteroBinaryBOBase(MCAcquisitionFunction):
    """Base class for heteroscedastic binary classification MC acquisitions."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        samples_are_probs: bool = True,
        apply_sigmoid_if_needed: bool = False,
        eps: float = 1e-6,
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
        self.samples_are_probs = bool(samples_are_probs)
        self.apply_sigmoid_if_needed = bool(apply_sigmoid_if_needed)
        self.eps = float(eps)

    def _hetero_samples(self, X: Tensor) -> Tensor:
        post = self.model.posterior(X)
        samples = self.get_posterior_samples(post)

        return hetero_adjust_binary_classification_samples(
            self.model,
            X,
            samples,
            beta=float(self.beta),
            noise_penalty=float(self.noise_penalty),
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            samples_are_probs=self.samples_are_probs,
            apply_sigmoid_if_needed=self.apply_sigmoid_if_needed,
            eps=self.eps,
            posterior=post,
        )


class qHeteroBinaryUpperConfidenceBound(_HeteroBinaryBOBase):
    """heteroscedastic classification 用 upper confidence bound acquisition。

    平均と不確実性を組み合わせて探索します。

    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。

    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。

    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける
        robust / noise-aware score に調整します。
    """

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        hetero = self._hetero_samples(X)
        return hetero.max(dim=-1).values.mean(dim=0)


class qHeteroBinaryExpectedImprovement(_HeteroBinaryBOBase):
    """heteroscedastic classification 用 expected improvement acquisition。

    現在の best_f からの改善量を評価します。

    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        best_f: 既存観測点または baseline から計算した現在の最良値。
        **kwargs: 親クラスまたは BoTorch acquisition に渡す追加 keyword arguments。

    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。

    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。

    Notes:
        hetero 版では noise posterior を使い、noise が大きい領域を避ける
        robust / noise-aware score に調整します。
    """

    def __init__(self, model: Model, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        hetero = self._hetero_samples(X)
        best_q = hetero.max(dim=-1).values
        best_f = self.best_f.to(best_q)

        return (best_q - best_f).clamp_min(0.0).mean(dim=0)


class qHeteroBinaryProbabilityOfImprovement(_HeteroBinaryBOBase):
    """heteroscedastic classification 用 probability of improvement acquisition。

    best_f を上回る確率を評価します。

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
        hetero 版では noise posterior を使い、noise が大きい領域を避ける
        robust / noise-aware score に調整します。
    """

    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        hetero = self._hetero_samples(X)
        best_q = hetero.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(1e-9)

        return torch.sigmoid((best_q - best_f) / tau).mean(dim=0)


__all__ = [
    "hetero_adjust_binary_classification_samples",
    "compute_hetero_binary_classification_best_f",
    "qHeteroBinaryUpperConfidenceBound",
    "qHeteroBinaryExpectedImprovement",
    "qHeteroBinaryProbabilityOfImprovement",
]
