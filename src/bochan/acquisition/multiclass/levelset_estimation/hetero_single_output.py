from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    _align_pointwise_to_reference,
    _boundary_weight,
    _class_entropy,
    _finalize_multiclass_acq_output_to_batch,
    ensure_q_batch,
    qMulticlassBoundaryVarianceAcquisition,
    qMulticlassClassEntropyAcquisition,
    qMulticlassICUAcquisition,
    qMulticlassJointLatentStraddleAcquisition,
    qMulticlassLatentStraddleAcquisition,
    qMulticlassLevelSetUncertainty,
    qMulticlassProbabilityOfExceedance,
)

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract", "add"]
NoiseQAggregateType = Literal["mean", "sum", "max", "min", "product"]


class _HeteroMulticlassLevelSetMixin:
    """Noise-aware mixin for complete heteroscedastic multiclass level-set acquisitions."""

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

    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        """active-learning 側と同じ method 名で q-batch 化する互換 helper。"""
        return ensure_q_batch(X)

    def _finalize(self, value: Tensor, raw_X: Tensor, *, name: str) -> Tensor:
        """BoTorch optimizer が期待する t-batch shape に acquisition 出力を揃える。"""
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=name)

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

    def _apply_noise_to_score(self, score: Tensor, X: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return score
        Xq = self._apply_input_transform(self._ensure_q_batch(X))
        raw_noise = self._get_noise_tensor(Xq)
        noise = self._to_point_noise(raw_noise, score)
        weight = self._noise_weight(noise, Xq)
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

    def _apply_noise_to_joint_value(self, value: Tensor, X: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return value
        Xq = self._apply_input_transform(self._ensure_q_batch(X))
        point_ref = Xq.new_zeros(Xq.shape[:-1])
        raw_noise = self._get_noise_tensor(Xq)
        noise = self._to_point_noise(raw_noise, point_ref)
        weight = self._noise_weight(noise, Xq)
        q_weight = self._aggregate_noise_weight_over_q(weight)
        q_weight = _align_pointwise_to_reference(q_weight, value, name=f"{self.__class__.__name__}.noise_weight")
        return self._combine_score_and_weight(value, q_weight)


class qHeteroMulticlassLatentStraddleAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassLatentStraddleAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score = self.beta * uncertainty - (p - self.threshold).abs()
        score = self._apply_noise_to_score(score, raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassJointLatentStraddleAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassJointLatentStraddleAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach().to(device=raw_X.device, dtype=raw_X.dtype)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            value = self._joint_score(raw_X)
            value = self._apply_noise_to_joint_value(value, raw_X)
            value = value - self._repulsion_penalty(raw_X)
            value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
            return self._finalize(value, raw_X, name=self.__class__.__name__)

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        pending_score = self._joint_score(Xp_batch)
        all_score = self._joint_score(torch.cat([Xp_batch, raw_X], dim=-2))
        value = all_score - pending_score
        value = self._apply_noise_to_joint_value(value, raw_X)
        value = value - self._repulsion_penalty(raw_X)
        value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMulticlassICUAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassICUAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score = uncertainty.pow(2) * contour_weight
        score = self._apply_noise_to_score(score, raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassBoundaryVarianceAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassBoundaryVarianceAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score = uncertainty.pow(2) * _boundary_weight(p, self.threshold, bandwidth=self.bandwidth, eps=self.eps)
        score = self._apply_noise_to_score(score, raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassClassEntropyAcquisition(_HeteroMulticlassLevelSetMixin, qMulticlassClassEntropyAcquisition):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._posterior_mean_probs(raw_X)
        if self.target_class is None:
            score = _class_entropy(probs, eps=self.eps)
        else:
            selected = self._target_prob(raw_X)
            score = -(selected.clamp_min(self.eps) * selected.clamp_min(self.eps).log())
        score = self._apply_noise_to_score(score, raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassProbabilityOfExceedance(_HeteroMulticlassLevelSetMixin, qMulticlassProbabilityOfExceedance):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        p = self._target_prob(raw_X)
        score = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = self._apply_noise_to_score(score, raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMulticlassLevelSetUncertainty(qHeteroMulticlassICUAcquisition):
    pass


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "NoiseQAggregateType",
    "qHeteroMulticlassLatentStraddleAcquisition",
    "qHeteroMulticlassJointLatentStraddleAcquisition",
    "qHeteroMulticlassICUAcquisition",
    "qHeteroMulticlassBoundaryVarianceAcquisition",
    "qHeteroMulticlassClassEntropyAcquisition",
    "qHeteroMulticlassProbabilityOfExceedance",
    "qHeteroMulticlassLevelSetUncertainty",
]
