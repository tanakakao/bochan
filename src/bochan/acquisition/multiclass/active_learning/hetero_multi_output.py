from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .multi_output import (
    qMultiOutputMulticlassBALD,
    qMultiOutputMulticlassGreedyJointBALD,
    qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
    qMultiOutputMulticlassJointBALD,
    qMultiOutputMulticlassMarginUncertainty,
    qMultiOutputMulticlassPredictiveEntropy,
    qMultiOutputMulticlassProbabilityVariance,
)

NoiseWeightMode = Literal["none", "inverse_linear", "inverse_sqrt", "exp", "custom"]
NoiseCombineType = Literal["multiply", "subtract", "add"]
NoiseQAggregateType = Literal["mean", "sum", "max", "min", "product"]


class _HeteroMultiOutputMulticlassMixin:
    """Noise-aware mixin for complete heteroscedastic multi-output multiclass AL."""

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

    def _call_predict_noise_var(self, model, X: Tensor) -> Tensor | None:
        fn = getattr(model, "predict_noise_var", None)
        if callable(fn):
            return fn(X)
        return None

    def _get_single_model_noise_tensor(self, model, X: Tensor) -> Tensor:
        noise = self._call_predict_noise_var(model, X)
        if noise is not None:
            return noise
        for name in ("posterior_noise", "noise_posterior"):
            fn = getattr(model, name, None)
            if callable(fn):
                return fn(X).mean
        noise_model = getattr(model, "noise_model", None)
        if noise_model is None:
            inner = getattr(model, "model", None)
            if inner is not None:
                noise_model = getattr(inner, "noise_model", None)
        if noise_model is not None:
            return noise_model.posterior(X).mean
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    def _to_point_noise(self, noise: Tensor, X: Tensor) -> Tensor:
        point_shape = X.shape[:-1]
        noise = torch.as_tensor(noise, device=X.device, dtype=X.dtype)
        while noise.ndim > len(point_shape):
            if noise.shape[-1] == 1:
                noise = noise.squeeze(-1)
            else:
                noise = noise.mean(dim=-1)
        if noise.shape == point_shape:
            return noise
        expected_numel = 1
        for s in point_shape:
            expected_numel *= int(s)
        if noise.numel() == expected_numel:
            return noise.reshape(point_shape)
        if noise.ndim == len(point_shape) + 1 and noise.shape[-1] == 1:
            return noise.squeeze(-1)
        return noise.mean().expand(point_shape)

    def _to_multioutput_noise(self, noise: Tensor, X: Tensor, *, n_outputs: int) -> Tensor:
        point_shape = X.shape[:-1]
        target_shape = (*point_shape, int(n_outputs))
        noise = torch.as_tensor(noise, device=X.device, dtype=X.dtype)
        if noise.shape == target_shape:
            return noise
        if noise.ndim == len(target_shape) + 1 and noise.shape[-1] == 1:
            noise = noise.squeeze(-1)
            if noise.shape == target_shape:
                return noise
        if noise.shape == point_shape and n_outputs == 1:
            return noise.unsqueeze(-1)
        expected_numel = 1
        for s in target_shape:
            expected_numel *= int(s)
        if noise.numel() == expected_numel:
            return noise.reshape(target_shape)
        while noise.ndim > len(target_shape):
            noise = noise.mean(dim=-1)
        if noise.shape == target_shape:
            return noise
        if noise.shape[-1:] == (n_outputs,) and noise.numel() % n_outputs == 0:
            try:
                return noise.reshape(target_shape)
            except RuntimeError:
                pass
        return noise.mean().expand(target_shape)

    def _maybe_convert_log_var(self, noise: Tensor) -> Tensor:
        if self.noise_model_outputs_log_var:
            return torch.exp(noise.clamp(min=-30.0, max=30.0)).clamp_min(self.eps)
        return noise.clamp_min(self.eps)

    def _get_noise_values(self, X: Tensor, *, n_outputs: int) -> Tensor:
        """Return noise values with shape ``batch_shape x q_like x m``."""

        X = self._ensure_q_batch(X)
        if self.noise_mode == "none":
            return torch.zeros(*X.shape[:-1], int(n_outputs), device=X.device, dtype=X.dtype)

        if self.noise_weight_fn is not None:
            custom = self.noise_weight_fn(None, X)
            return self._to_multioutput_noise(custom, X, n_outputs=n_outputs)

        model_noise = self._call_predict_noise_var(self.model, X)
        if model_noise is not None:
            return self._maybe_convert_log_var(self._to_multioutput_noise(model_noise, X, n_outputs=n_outputs))

        submodels = self._submodels()
        if len(submodels) > 0:
            pieces = []
            for submodel in submodels:
                noise_i = self._get_single_model_noise_tensor(submodel, X)
                noise_i = self._maybe_convert_log_var(self._to_point_noise(noise_i, X))
                pieces.append(noise_i.unsqueeze(-1))
            noise = torch.cat(pieces, dim=-1)
            if noise.shape[-1] != n_outputs:
                return self._to_multioutput_noise(noise, X, n_outputs=n_outputs)
            return noise

        noise = self._get_single_model_noise_tensor(self.model, X)
        return self._maybe_convert_log_var(self._to_multioutput_noise(noise, X, n_outputs=n_outputs))

    def _noise_to_weight(self, noise: Tensor) -> Tensor:
        if self.noise_mode == "none":
            weight = torch.ones_like(noise)
        elif self.noise_mode == "custom":
            if self.noise_weight_fn is None:
                raise ValueError("noise_weight_fn must be provided when noise_mode='custom'.")
            weight = self.noise_weight_fn(noise, None)
        elif self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "inverse_sqrt":
            weight = 1.0 / torch.sqrt(1.0 + self.noise_penalty_lambda * noise.clamp_min(0.0))
        elif self.noise_mode == "exp":
            weight = torch.exp(-self.noise_penalty_lambda * noise.clamp_min(0.0))
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")
        return (self.noise_weight_scale * weight).clamp_min(self.noise_min_weight)

    def _combine_score_and_weight(self, score: Tensor, weight: Tensor) -> Tensor:
        if self.noise_combine == "multiply":
            return score * weight.to(score)
        if self.noise_combine in {"subtract", "add"}:
            return score - (1.0 - weight.to(score))
        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")

    def _apply_noise_to_score_per_output(self, score_per_output: Tensor, X: Tensor) -> Tensor:
        noise = self._get_noise_values(X, n_outputs=score_per_output.shape[-1])
        weight = self._noise_to_weight(noise).to(score_per_output)
        return self._combine_score_and_weight(score_per_output, weight)

    def _aggregate_noise_over_q(self, weight: Tensor) -> Tensor:
        if self.noise_q_aggregate == "mean":
            return weight.mean(dim=-2)
        if self.noise_q_aggregate == "sum":
            return weight.sum(dim=-2)
        if self.noise_q_aggregate == "max":
            return weight.max(dim=-2).values
        if self.noise_q_aggregate == "min":
            return weight.min(dim=-2).values
        if self.noise_q_aggregate == "product":
            return weight.prod(dim=-2)
        raise ValueError(f"Unknown noise_q_aggregate: {self.noise_q_aggregate!r}.")

    def _apply_noise_to_q_aggregated_output_score(self, score_per_output: Tensor, X: Tensor) -> Tensor:
        noise = self._get_noise_values(X, n_outputs=score_per_output.shape[-1])
        weight = self._aggregate_noise_over_q(self._noise_to_weight(noise)).to(score_per_output)
        return self._combine_score_and_weight(score_per_output, weight)


class qHeteroMultiOutputMulticlassPredictiveEntropy(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassPredictiveEntropy,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._entropy(self._mean_probs(raw_X))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassProbabilityVariance(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityVariance,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._class_probability_variance(self._mean_probs(raw_X))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassMarginUncertainty(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassMarginUncertainty,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._margin_uncertainty(self._mean_probs(raw_X))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassBALD(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassBALD,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        score_per_output = self._pointwise_bald_per_output(raw_X)
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassJointBALD(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassJointBALD,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        value_per_output = self._joint_bald_per_output(raw_X)
        value_per_output = self._apply_noise_to_q_aggregated_output_score(value_per_output, Xt)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._joint_penalty(raw_X, Xt)
        value = self._apply_objective(value, raw_X=raw_X, expanded_X=Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassGreedyJointBALD(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassGreedyJointBALD,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or X_pending.numel() == 0:
            value_per_output = self._joint_bald_per_output(raw_X)
        else:
            Xp = X_pending.to(device=raw_X.device, dtype=raw_X.dtype)
            Xp = self._expand_pending_to_batch(Xp, raw_X.shape[:-2])
            pending_value = self._joint_bald_per_output(Xp)
            all_value = self._joint_bald_per_output(torch.cat([Xp, raw_X], dim=-2))
            value_per_output = all_value - pending_value
        value_per_output = self._apply_noise_to_q_aggregated_output_score(value_per_output, Xt)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._observed_penalty_per_point(Xt).sum(dim=-1)
        value = value - self._same_batch_penalty(Xt)
        value = self._apply_objective(value, raw_X=raw_X, expanded_X=Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassIntegratedPosteriorVarianceProxy,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_per_output(raw_X, Xt)
        score_per_output = self.local_weight * local_score + self.integrated_weight * integrated_score
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "NoiseWeightMode",
    "NoiseCombineType",
    "NoiseQAggregateType",
    "qHeteroMultiOutputMulticlassPredictiveEntropy",
    "qHeteroMultiOutputMulticlassProbabilityVariance",
    "qHeteroMultiOutputMulticlassMarginUncertainty",
    "qHeteroMultiOutputMulticlassBALD",
    "qHeteroMultiOutputMulticlassJointBALD",
    "qHeteroMultiOutputMulticlassGreedyJointBALD",
    "qHeteroMultiOutputMulticlassIntegratedPosteriorVarianceProxy",
]
