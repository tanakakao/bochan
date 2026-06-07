from __future__ import annotations

from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
import torch

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin

from .multi_output import (
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
    qMultiOutputMulticlassClassEntropyAcquisition,
    qMultiOutputMulticlassICUAcquisition,
    qMultiOutputMulticlassJointLatentStraddleAcquisition,
    qMultiOutputMulticlassLatentStraddleAcquisition,
    qMultiOutputMulticlassLevelSetUncertainty,
    qMultiOutputMulticlassProbabilityOfExceedance,
)


class qHeteroMultiOutputMulticlassLatentStraddleAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassLatentStraddleAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        std = (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        score_per_output = self.beta * std - (p - self.threshold).abs()
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition(qHeteroMultiOutputMulticlassLatentStraddleAcquisition):
    def __init__(self, model, **kwargs) -> None:
        kwargs.setdefault("reduction", "sum")
        super().__init__(model=model, **kwargs)


class qHeteroMultiOutputMulticlassICUAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassICUAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        uncertainty = (p * (1.0 - p)).clamp_min(self.eps)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score_per_output = uncertainty * contour_weight
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        variance = p * (1.0 - p)
        boundary_weight = torch.exp(-((p - self.threshold).abs() / max(self.bandwidth, self.eps)))
        score_per_output = variance * boundary_weight
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassClassEntropyAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassClassEntropyAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        score_per_output = self._entropy(probs)
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassProbabilityOfExceedance(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfExceedance,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        self._current_batch_shape = raw_X.shape[:-2]
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        p = self._target_prob_per_output(probs)
        score_per_output = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        value = self._pointwise_score_to_value(score_per_output, raw_X, Xt)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassLevelSetUncertainty(qHeteroMultiOutputMulticlassICUAcquisition):
    pass


__all__ = [
    "qHeteroMultiOutputMulticlassLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition",
    "qHeteroMultiOutputMulticlassICUAcquisition",
    "qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition",
    "qHeteroMultiOutputMulticlassClassEntropyAcquisition",
    "qHeteroMultiOutputMulticlassProbabilityOfExceedance",
    "qHeteroMultiOutputMulticlassLevelSetUncertainty",
]
