from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ReductionType

from .single_output import (
    qMulticlassBALD,
    qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassMarginUncertainty,
    qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance,
)

NoiseWeightMode = Literal["none", "inverse_linear", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]


class _HeteroMulticlassMixin:
    """Noise-aware mixin for heteroscedastic multiclass active learning."""

    def __init__(
        self,
        *args,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_weight_fn: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        **kwargs,
    ) -> None:
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_weight_fn = noise_weight_fn
        super().__init__(*args, **kwargs)

    def _predict_noise_var_for_score(self, X: Tensor, score: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return torch.zeros_like(score)
        if hasattr(self.model, "predict_noise_var"):
            noise = self.model.predict_noise_var(X)
        elif hasattr(self.model, "noise_model"):
            noise = self.model.noise_model.posterior(X).mean.exp()
        else:
            return torch.zeros_like(score)
        while noise.ndim > score.ndim:
            noise = noise.mean(dim=-1)
        if noise.shape == score.shape:
            return noise.to(score)
        if noise.shape[-1:] and noise.shape[-1] != score.shape[-1]:
            noise = noise.mean(dim=-1)
        if noise.shape == score.shape:
            return noise.to(score)
        if noise.numel() == score.numel():
            return noise.reshape_as(score).to(score)
        return noise.mean().expand_as(score).to(score)

    def _noise_weight(self, noise: Tensor, X: Tensor, score: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return torch.ones_like(score)
        if self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            weight = self.noise_weight_fn(noise, X)
        elif self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "exp":
            weight = torch.exp(-self.noise_penalty_lambda * noise.clamp_min(0.0))
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")
        weight = self.noise_weight_scale * weight
        return weight.clamp_min(self.noise_min_weight).to(score)

    def _apply_noise_to_score(self, score: Tensor, X: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return score
        Xq = self._ensure_q_batch(X)
        noise = self._predict_noise_var_for_score(Xq, score)
        if self.noise_combine == "multiply":
            return score * self._noise_weight(noise, Xq, score)
        if self.noise_combine == "subtract":
            return score - self.noise_penalty_lambda * noise
        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")


class qHeteroMulticlassPredictiveEntropy(_HeteroMulticlassMixin, qMulticlassPredictiveEntropy):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._entropy(probs)
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassProbabilityVariance(_HeteroMulticlassMixin, qMulticlassProbabilityVariance):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._class_probability_variance(probs)
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassMarginUncertainty(_HeteroMulticlassMixin, qMulticlassMarginUncertainty):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._margin_uncertainty(probs)
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassBALD(_HeteroMulticlassMixin, qMulticlassBALD):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        samples = self._sample_probs(Xq, num_samples=self.num_samples)
        mean_probs = samples.mean(dim=0)
        predictive_entropy = self._entropy(mean_probs)
        expected_entropy = self._entropy(samples).mean(dim=0)
        score = predictive_entropy - expected_entropy
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassIntegratedPosteriorVarianceProxy(_HeteroMulticlassMixin, qMulticlassIntegratedPosteriorVarianceProxy):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._class_probability_variance(probs)
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "qHeteroMulticlassPredictiveEntropy",
    "qHeteroMulticlassProbabilityVariance",
    "qHeteroMulticlassMarginUncertainty",
    "qHeteroMulticlassBALD",
    "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
]
