from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    qMulticlassBoundaryVarianceAcquisition,
    qMulticlassClassEntropyAcquisition,
    qMulticlassICUAcquisition,
    qMulticlassJointLatentStraddleAcquisition,
    qMulticlassLatentStraddleAcquisition,
    qMulticlassLevelSetUncertainty,
    qMulticlassProbabilityOfExceedance,
)

NoiseWeightMode = Literal["none", "inverse_linear", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract"]


class _HeteroMulticlassLevelSetMixin:
    """Noise-aware mixin for heteroscedastic multiclass level-set acquisitions."""

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
        if noise.ndim > 0 and noise.shape[-1] != score.shape[-1]:
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


class qHeteroMulticlassLatentStraddleAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassLatentStraddleAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        std = (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        score = self.beta * std - (p - self.threshold).abs()
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassJointLatentStraddleAcquisition(qHeteroMulticlassLatentStraddleAcquisition):
    pass


class qHeteroMulticlassICUAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassICUAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        uncertainty = (p * (1.0 - p)).clamp_min(self.eps)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score = uncertainty * contour_weight
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassBoundaryVarianceAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassBoundaryVarianceAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        variance = p * (1.0 - p)
        boundary_weight = torch.exp(-((p - self.threshold).abs() / max(self.bandwidth, self.eps)))
        score = variance * boundary_weight
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassClassEntropyAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassClassEntropyAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._entropy(probs)
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassProbabilityOfExceedance(_HeteroMulticlassLevelSetMixin, qMulticlassProbabilityOfExceedance):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        score = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = self._apply_noise_to_score(score, Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qHeteroMulticlassLevelSetUncertainty(qHeteroMulticlassICUAcquisition):
    pass


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "qHeteroMulticlassLatentStraddleAcquisition",
    "qHeteroMulticlassJointLatentStraddleAcquisition",
    "qHeteroMulticlassICUAcquisition",
    "qHeteroMulticlassBoundaryVarianceAcquisition",
    "qHeteroMulticlassClassEntropyAcquisition",
    "qHeteroMulticlassProbabilityOfExceedance",
    "qHeteroMulticlassLevelSetUncertainty",
]
