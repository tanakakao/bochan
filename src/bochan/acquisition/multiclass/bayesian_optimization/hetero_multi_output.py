from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin

from .multi_output import (
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassExpectedImprovement,
    qMultiOutputMulticlassNParEGO,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qMultiOutputMulticlassProbabilityOfFeasibility,
    qMultiOutputMulticlassProbabilityOfImprovement,
    qMultiOutputMulticlassUpperConfidenceBound,
)


class qHeteroMultiOutputMulticlassProbabilityOfFeasibility(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfFeasibility,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        score_per_output = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._candidate_samples(raw_X)
        ref = self._ref(device=samples.device, dtype=samples.dtype)
        base_hv = self._baseline_hv(ref=ref, device=samples.device, dtype=samples.dtype)
        hv = self._hypervolume(samples, ref)
        value = (hv - base_hv).clamp_min(0.0).mean(dim=0)
        # Hypervolume is already output-aggregated. Penalize by average output noise.
        noise = self._get_noise_values(Xt, n_outputs=samples.shape[-1])
        weight = self._aggregate_noise_over_q(self._noise_to_weight(noise)).mean(dim=-1).to(value)
        value = self._combine_score_and_weight(value, weight)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._candidate_samples(raw_X)
        ref = self._ref(device=samples.device, dtype=samples.dtype)
        Yb = self._baseline_values(device=samples.device, dtype=samples.dtype)
        if Yb is None:
            base_hv = samples.new_zeros(())
            hv = self._hypervolume(samples, ref)
        else:
            batch_shape = samples.shape[1:-2]
            Yb_exp = Yb.view(*([1] * len(batch_shape)), *Yb.shape).expand(*batch_shape, *Yb.shape)
            Yb_exp = Yb_exp.unsqueeze(0).expand(samples.shape[0], *Yb_exp.shape)
            combined = torch.cat([Yb_exp, samples], dim=-2)
            base_hv = self._hypervolume(Yb_exp, ref)
            hv = self._hypervolume(combined, ref)
        value = (hv - base_hv).clamp_min(0.0).mean(dim=0)
        noise = self._get_noise_values(Xt, n_outputs=samples.shape[-1])
        weight = self._aggregate_noise_over_q(self._noise_to_weight(noise)).mean(dim=-1).to(value)
        value = self._combine_score_and_weight(value, weight)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassNParEGO(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassNParEGO,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        weights = self._weights(m=samples.shape[-1], device=samples.device, dtype=samples.dtype)
        scalar = self._scalarize(samples, weights)
        best_q = scalar.max(dim=-1).values
        best_f = self._baseline_best_f(weights=weights, device=samples.device, dtype=samples.dtype)
        value = (best_q - best_f).clamp_min(0.0).mean(dim=0)
        noise = self._get_noise_values(Xt, n_outputs=samples.shape[-1])
        weighted_noise = self._noise_to_weight(noise) * weights.view(*([1] * (noise.ndim - 1)), -1)
        weight = self._aggregate_noise_over_q(weighted_noise.sum(dim=-1, keepdim=True)).squeeze(-1).to(value)
        value = self._combine_score_and_weight(value, weight)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassExpectedImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassExpectedImprovement,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        value_per_output = (best_q_per_output - best_f).clamp_min(0.0).mean(dim=0)
        value_per_output = self._apply_noise_to_q_aggregated_output_score(value_per_output, Xt)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassProbabilityOfImprovement(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfImprovement,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        best_q_per_output = samples.max(dim=-2).values
        best_f = self._align_output_param(self.best_f, ref=best_q_per_output, name="best_f")
        tau = self.tau.to(best_q_per_output).clamp_min(self.eps)
        value_per_output = torch.sigmoid((best_q_per_output - best_f) / tau).mean(dim=0)
        value_per_output = self._apply_noise_to_q_aggregated_output_score(value_per_output, Xt)
        value = self._aggregate_outputs(value_per_output)
        value = value - self._pending_q_penalty(Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassUpperConfidenceBound(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassUpperConfidenceBound,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        samples = self._target_prob_samples_per_output(raw_X, num_samples=self.num_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        beta = self._align_output_param(self.beta, ref=mean, name="beta")
        score_per_output = mean + beta.sqrt() * std
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


__all__ = [
    "qHeteroMultiOutputMulticlassProbabilityOfFeasibility",
    "qHeteroMultiOutputMulticlassExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNoisyExpectedHypervolumeImprovement",
    "qHeteroMultiOutputMulticlassNParEGO",
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
]
