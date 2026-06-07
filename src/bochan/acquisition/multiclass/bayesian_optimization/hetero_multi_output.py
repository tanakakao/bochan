from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin

from .multi_output import (
    qMultiOutputMulticlassExpectedImprovement,
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
    "qHeteroMultiOutputMulticlassExpectedImprovement",
    "qHeteroMultiOutputMulticlassProbabilityOfImprovement",
    "qHeteroMultiOutputMulticlassUpperConfidenceBound",
]
