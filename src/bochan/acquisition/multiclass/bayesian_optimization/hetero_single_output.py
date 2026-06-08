from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Optional

import torch
from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ClassReductionType

from .single_output import (
    _MulticlassProbabilityBOBase,
    _finalize_multiclass_acq_output_to_batch,
    _mean_over_sample_dims,
    _select_class_probs,
    _std_over_sample_dims,
    compute_multiclass_target_probability_values,
    ensure_q_batch,
)

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract", "add"]


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


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Align a pointwise tensor to a reference ``batch_shape x q`` tensor."""

    if value.ndim >= 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    while value.ndim > reference.ndim:
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        else:
            value = value.mean(dim=-1)

    if value.shape == reference.shape:
        return value.to(reference)

    if value.shape == reference.shape[:-1]:
        return value.unsqueeze(-1).expand_as(reference).to(reference)

    if value.ndim == reference.ndim and value.shape[:-1] == reference.shape[:-1]:
        q_ref = reference.shape[-1]
        q_value = value.shape[-1]
        if q_ref % q_value == 0:
            return value.repeat_interleave(q_ref // q_value, dim=-1).to(reference)
        if q_value % q_ref == 0:
            return value.reshape(*reference.shape[:-1], q_ref, q_value // q_ref).mean(dim=-1).to(reference)

    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)

    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)

    raise RuntimeError(
        f"{name}: cannot align value to reference. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


def _get_noise_std(
    model: Model,
    X: Tensor,
    *,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Return heteroscedastic noise standard deviation for multiclass BO."""

    Xq = ensure_q_batch(X)
    try:
        noise_post = _get_noise_posterior(model, Xq)
        noise_mean = noise_post.mean
        while noise_mean.ndim > len(Xq.shape[:-1]):
            if noise_mean.shape[-1] == 1:
                noise_mean = noise_mean.squeeze(-1)
            else:
                noise_mean = noise_mean.mean(dim=-1)
        if noise_is_log_var:
            noise_var = torch.exp(noise_mean.clamp(min=math.log(eps), max=30.0))
        else:
            noise_var = noise_mean.clamp_min(eps)
        return noise_var.sqrt().clamp_min(eps)
    except Exception:
        return torch.full(Xq.shape[:-1], float(default_sigma), device=Xq.device, dtype=Xq.dtype)


def hetero_adjust_multiclass_target_probability_samples(
    model: Model,
    X: Tensor,
    target_samples: Tensor,
    target_mean: Tensor,
    *,
    beta: float = 0.0,
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Apply heteroscedastic adjustment to target-class probability samples.

    Args:
        model: Heteroscedastic multiclass model.
        X: Candidate tensor with shape ``batch_shape x q x d``.
        target_samples: Target probability samples with shape
            ``sample_shape x batch_shape x q``.
        target_mean: Target probability mean with shape ``batch_shape x q``.
        beta: Scale of posterior sample deviation from the mean.
        noise_penalty: Penalty coefficient for predicted observation noise.
        default_sigma: Fallback noise standard deviation when no noise model is found.
        noise_is_log_var: Whether the noise model output is log variance.
        eps: Numerical stability constant.

    Returns:
        Hetero-adjusted target probability samples with the same shape as
        ``target_samples``.
    """

    Xq = ensure_q_batch(X)
    target_mean = _align_pointwise_to_reference(target_mean, target_samples.mean(dim=0), name="target_mean")
    sigma_noise = _get_noise_std(
        model,
        Xq,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        eps=eps,
    )
    sigma_noise = _align_pointwise_to_reference(sigma_noise, target_mean, name="sigma_noise")

    adjusted = target_mean.unsqueeze(0) + float(beta) * (target_samples - target_mean.unsqueeze(0))
    adjusted = adjusted - float(noise_penalty) * sigma_noise.unsqueeze(0)
    return adjusted.clamp(eps, 1.0 - eps)


def compute_hetero_multiclass_target_probability_values(
    model: Model,
    X: Tensor,
    *,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReductionType = "mean",
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    apply_softmax_if_needed: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Compute noise-penalized target-class probabilities."""

    Xq = ensure_q_batch(X)
    with torch.no_grad():
        values = compute_multiclass_target_probability_values(
            model=model,
            X=Xq,
            target_class=target_class,
            class_reduction=class_reduction,
            apply_softmax_if_needed=apply_softmax_if_needed,
            eps=eps,
        ).reshape(Xq.shape[:-1])
        sigma_noise = _get_noise_std(
            model,
            Xq,
            default_sigma=default_sigma,
            noise_is_log_var=noise_is_log_var,
            eps=eps,
        )
        sigma_noise = _align_pointwise_to_reference(sigma_noise, values, name="sigma_noise")
        values = (values - float(noise_penalty) * sigma_noise).clamp(eps, 1.0 - eps)
    return values.detach().reshape(-1)


def compute_hetero_multiclass_target_probability_best_f(
    model: Model,
    train_X: Tensor,
    *,
    target_class: int | Sequence[int] | None,
    class_reduction: ClassReductionType = "mean",
    noise_penalty: float = 0.0,
    default_sigma: float = 0.0,
    noise_is_log_var: bool = True,
    apply_softmax_if_needed: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Compute best_f for heteroscedastic multiclass target-probability BO."""

    values = compute_hetero_multiclass_target_probability_values(
        model=model,
        X=train_X,
        target_class=target_class,
        class_reduction=class_reduction,
        noise_penalty=noise_penalty,
        default_sigma=default_sigma,
        noise_is_log_var=noise_is_log_var,
        apply_softmax_if_needed=apply_softmax_if_needed,
        eps=eps,
    )
    return values.max().detach()


class _HeteroMulticlassBOBase(_MulticlassProbabilityBOBase):
    """Base class for heteroscedastic multiclass MC acquisitions."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float = 0.0,
        noise_penalty: float = 0.0,
        default_sigma: float = 0.0,
        noise_is_log_var: bool = True,
        eps: float = 1e-6,
        sampler: Optional[SobolQMCNormalSampler] = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, sampler=sampler, eps=eps, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("noise_penalty", torch.as_tensor(noise_penalty))
        self.register_buffer("default_sigma", torch.as_tensor(default_sigma))
        self.noise_is_log_var = bool(noise_is_log_var)

    def _hetero_target_samples(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        target_samples = self._target_prob_samples(Xq)
        target_mean = self._target_prob_mean(Xq)
        return hetero_adjust_multiclass_target_probability_samples(
            self.model,
            Xq,
            target_samples,
            target_mean,
            beta=float(self.beta),
            noise_penalty=float(self.noise_penalty),
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            eps=self.eps,
        )

    def _hetero_target_mean(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        mean = self._target_prob_mean(Xq)
        sigma_noise = _get_noise_std(
            self.model,
            Xq,
            default_sigma=float(self.default_sigma),
            noise_is_log_var=self.noise_is_log_var,
            eps=self.eps,
        )
        sigma_noise = _align_pointwise_to_reference(sigma_noise, mean, name="sigma_noise")
        return (mean - float(self.noise_penalty) * sigma_noise).clamp(self.eps, 1.0 - self.eps)


class qHeteroMulticlassProbabilityOfFeasibility(_HeteroMulticlassBOBase):
    """Noise-aware target-class probability / probability of feasibility."""

    def __init__(
        self,
        model: Model,
        *,
        target_class: int | Sequence[int] | None,
        threshold: float | None = None,
        tau: float = 0.02,
        q_feas_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.threshold = None if threshold is None else float(threshold)
        self.tau = float(tau)
        self.q_feas_mode = q_feas_mode

    def _reduce_q_feas(self, score: Tensor) -> Tensor:
        if self.q_feas_mode is None:
            return self._reduce_q(score)
        if self.q_feas_mode == "prod":
            return score.prod(dim=-1)
        if self.q_feas_mode == "mean":
            return score.mean(dim=-1)
        if self.q_feas_mode == "min":
            return score.min(dim=-1).values
        if self.q_feas_mode == "max":
            return score.max(dim=-1).values
        raise ValueError(f"Unknown q_feas_mode: {self.q_feas_mode!r}.")

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._hetero_target_mean(raw_X)
        score = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = score - self._pending_penalty_per_point(Xt)
        score = score - self._observed_penalty_per_point(Xt)
        value = self._reduce_q_feas(score)
        value = value - self._same_batch_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassExpectedImprovement(_HeteroMulticlassBOBase):
    """Noise-aware expected improvement for target-class probability."""

    def __init__(self, model: Model, *, target_class: int | Sequence[int] | None, best_f: float | Tensor, **kwargs) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        hetero = self._hetero_target_samples(raw_X)
        best_q = hetero.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        value = (best_q - best_f).clamp_min(0.0)
        value = _mean_over_sample_dims(value, self.sampler)
        value = value - self._q_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassProbabilityOfImprovement(_HeteroMulticlassBOBase):
    """Noise-aware probability of improvement for target-class probability."""

    def __init__(
        self,
        model: Model,
        *,
        target_class: int | Sequence[int] | None,
        best_f: float | Tensor,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        hetero = self._hetero_target_samples(raw_X)
        best_q = hetero.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(self.eps)
        value = torch.sigmoid((best_q - best_f) / tau)
        value = _mean_over_sample_dims(value, self.sampler)
        value = value - self._q_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassUpperConfidenceBound(_HeteroMulticlassBOBase):
    """Noise-aware UCB-style target-class probability acquisition."""

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        hetero = self._hetero_target_samples(raw_X)
        mean = _mean_over_sample_dims(hetero, self.sampler)
        std = _std_over_sample_dims(hetero, self.sampler, eps=self.eps)
        score = mean + self.beta.to(mean).sqrt() * std
        return self._pointwise_score_to_value(score, raw_X, Xt)


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "_get_noise_posterior",
    "_get_noise_std",
    "hetero_adjust_multiclass_target_probability_samples",
    "compute_hetero_multiclass_target_probability_values",
    "compute_hetero_multiclass_target_probability_best_f",
    "_HeteroMulticlassBOBase",
    "qHeteroMulticlassProbabilityOfFeasibility",
    "qHeteroMulticlassExpectedImprovement",
    "qHeteroMulticlassProbabilityOfImprovement",
    "qHeteroMulticlassUpperConfidenceBound",
]
