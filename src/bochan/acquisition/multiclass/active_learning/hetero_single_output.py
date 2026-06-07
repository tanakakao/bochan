from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    _align_pointwise_to_reference,
    qMulticlassBALD,
    qMulticlassGreedyJointBALD,
    qMulticlassIntegratedPosteriorVarianceProxy,
    qMulticlassJointBALD,
    qMulticlassMarginUncertainty,
    qMulticlassPredictiveEntropy,
    qMulticlassProbabilityVariance,
)

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract", "add"]
NoiseQAggregateType = Literal["mean", "sum", "max", "min", "product"]


class _HeteroMulticlassMixin:
    """Noise-aware mixin for complete heteroscedastic multiclass active learning."""

    def __init__(
        self,
        *args,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "multiply",
        noise_penalty_lambda: float = 1.0,
        noise_min_weight: float = 0.0,
        noise_weight_scale: float = 1.0,
        noise_model_outputs_log_var: bool = True,
        noise_q_aggregate: NoiseQAggregateType = "mean",
        noise_weight_fn: Callable[[Tensor | None, Tensor], Tensor] | None = None,
        **kwargs,
    ) -> None:
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_penalty_lambda = float(noise_penalty_lambda)
        self.noise_min_weight = float(noise_min_weight)
        self.noise_weight_scale = float(noise_weight_scale)
        self.noise_model_outputs_log_var = bool(noise_model_outputs_log_var)
        self.noise_q_aggregate = noise_q_aggregate
        self.noise_weight_fn = noise_weight_fn
        super().__init__(*args, **kwargs)

    def _call_predict_noise_var(self, X: Tensor) -> Tensor | None:
        fn = getattr(self.model, "predict_noise_var", None)
        if callable(fn):
            return fn(X)
        return None

    def _get_noise_tensor(self, X: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        model_noise = self._call_predict_noise_var(X)
        if model_noise is not None:
            return torch.as_tensor(model_noise, device=X.device, dtype=X.dtype)

        for name in ("posterior_noise", "noise_posterior"):
            fn = getattr(self.model, name, None)
            if callable(fn):
                return torch.as_tensor(fn(X).mean, device=X.device, dtype=X.dtype)

        noise_model = getattr(self.model, "noise_model", None)
        if noise_model is None:
            inner = getattr(self.model, "model", None)
            if inner is not None:
                noise_model = getattr(inner, "noise_model", None)
        if noise_model is not None:
            return torch.as_tensor(noise_model.posterior(X).mean, device=X.device, dtype=X.dtype)

        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    def _to_point_noise(self, noise: Tensor, reference: Tensor) -> Tensor:
        noise = torch.as_tensor(noise, device=reference.device, dtype=reference.dtype)
        while noise.ndim > reference.ndim:
            if noise.shape[-1] == 1:
                noise = noise.squeeze(-1)
            else:
                noise = noise.mean(dim=-1)
        if noise.shape == reference.shape:
            out = noise
        elif noise.shape == reference.shape[:-1]:
            out = noise.unsqueeze(-1).expand_as(reference)
        elif noise.numel() == reference.numel():
            out = noise.reshape_as(reference)
        elif noise.numel() == 1:
            out = noise.reshape(()).expand_as(reference)
        else:
            out = noise.mean().expand_as(reference)
        if self.noise_model_outputs_log_var:
            out = torch.exp(out.clamp(min=-30.0, max=30.0)).clamp_min(self.eps)
        return out.clamp_min(self.eps)

    def _noise_weight(self, noise: Tensor, X: Tensor) -> Tensor:
        if self.noise_mode == "none":
            weight = torch.ones_like(noise)
        elif self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            weight = self.noise_weight_fn(noise, X)
        elif self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "inverse_sqrt":
            weight = 1.0 / torch.sqrt(1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "exp":
            weight = torch.exp(-self.noise_penalty_lambda * noise.clamp_min(0.0))
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")
        return (self.noise_weight_scale * weight).clamp_min(self.noise_min_weight).to(noise)

    def _combine_score_and_weight(self, score: Tensor, weight: Tensor) -> Tensor:
        weight = weight.to(score)
        if self.noise_combine == "multiply":
            return score * weight
        if self.noise_combine in {"subtract", "add"}:
            return score - (1.0 - weight)
        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_noise_to_score(self, score: Tensor, Xt: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return score
        Xt = self._ensure_q_batch(Xt)
        raw_noise = self._get_noise_tensor(Xt)
        noise = self._to_point_noise(raw_noise, score)
        weight = self._noise_weight(noise, Xt)
        return self._combine_score_and_weight(score, weight)

    def _aggregate_noise_weight_over_q(self, weight: Tensor) -> Tensor:
        if self.noise_q_aggregate == "mean":
            return weight.mean(dim=-1)
        if self.noise_q_aggregate == "sum":
            return weight.sum(dim=-1)
        if self.noise_q_aggregate == "max":
            return weight.max(dim=-1).values
        if self.noise_q_aggregate == "min":
            return weight.min(dim=-1).values
        if self.noise_q_aggregate == "product":
            return weight.prod(dim=-1)
        raise ValueError(f"Unknown noise_q_aggregate: {self.noise_q_aggregate!r}.")

    def _apply_noise_to_joint_value(self, value: Tensor, Xt: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return value
        Xt = self._ensure_q_batch(Xt)
        point_ref = Xt.new_zeros(Xt.shape[:-1])
        raw_noise = self._get_noise_tensor(Xt)
        noise = self._to_point_noise(raw_noise, point_ref)
        weight = self._noise_weight(noise, Xt)
        q_weight = self._aggregate_noise_weight_over_q(weight)
        q_weight = _align_pointwise_to_reference(q_weight, value, name=f"{self.__class__.__name__}.noise_weight")
        return self._combine_score_and_weight(value, q_weight)


class qHeteroMulticlassPredictiveEntropy(_HeteroMulticlassMixin, qMulticlassPredictiveEntropy):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._entropy(probs)
        score = self._apply_noise_to_score(score, Xt)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassProbabilityVariance(_HeteroMulticlassMixin, qMulticlassProbabilityVariance):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._class_probability_variance(probs)
        score = self._apply_noise_to_score(score, Xt)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassMarginUncertainty(_HeteroMulticlassMixin, qMulticlassMarginUncertainty):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._margin_uncertainty(probs)
        score = self._apply_noise_to_score(score, Xt)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassBALD(_HeteroMulticlassMixin, qMulticlassBALD):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        score = self._pointwise_bald_score(raw_X)
        score = self._apply_noise_to_score(score, Xt)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassJointBALD(_HeteroMulticlassMixin, qMulticlassJointBALD):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        value = self._joint_bald_value(raw_X)
        value = self._apply_noise_to_joint_value(value, Xt)
        value = value - self._joint_penalty(raw_X, Xt)
        value = self._apply_active_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassGreedyJointBALD(_HeteroMulticlassMixin, qMulticlassGreedyJointBALD):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or torch.as_tensor(X_pending).numel() == 0:
            value = self._joint_bald_value(raw_X)
        else:
            Xp = torch.as_tensor(X_pending, dtype=raw_X.dtype, device=raw_X.device).detach()
            Xp = self._expand_pending_to_batch(Xp, raw_X.shape[:-2])
            pending_value = self._joint_bald_value(Xp)
            all_value = self._joint_bald_value(torch.cat([Xp, raw_X], dim=-2))
            value = all_value - pending_value
        value = self._apply_noise_to_joint_value(value, Xt)
        value = value - self._observed_penalty_per_point(Xt).sum(dim=-1)
        value = value - self._same_batch_penalty(Xt)
        value = self._apply_active_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassIntegratedPosteriorVarianceProxy(
    _HeteroMulticlassMixin,
    qMulticlassIntegratedPosteriorVarianceProxy,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_score_per_point(raw_X, Xt)
        integrated_score = _align_pointwise_to_reference(
            integrated_score,
            local_score,
            name="HeteroIPV.integrated_score",
        )
        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        score = self._apply_noise_to_score(score, Xt)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "NoiseQAggregateType",
    "qHeteroMulticlassPredictiveEntropy",
    "qHeteroMulticlassProbabilityVariance",
    "qHeteroMulticlassMarginUncertainty",
    "qHeteroMulticlassBALD",
    "qHeteroMulticlassJointBALD",
    "qHeteroMulticlassGreedyJointBALD",
    "qHeteroMulticlassIntegratedPosteriorVarianceProxy",
]
