from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.active_learning.hetero_multi_output import _HeteroMultiOutputMulticlassMixin

from .multi_output import (
    _boundary_weight,
    _class_entropy,
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
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = self.beta * uncertainty - (p - self.threshold).abs()
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassJointLatentStraddleAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassJointLatentStraddleAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        batch_shape = raw_X.shape[:-2]
        Xp = getattr(self, "X_pending", None)
        if Xp is not None:
            Xp = Xp.detach().to(device=raw_X.device, dtype=raw_X.dtype)

        if Xp is None or Xp.numel() == 0 or not self.marginalize_pending:
            score_per_output = self._joint_score_per_output(raw_X)
            score_per_output = self._apply_noise_to_q_aggregated_output_score(score_per_output, self._apply_input_transform(raw_X))
            value = self._aggregate_outputs(score_per_output)
            value = value - self._repulsion_penalty(raw_X)
            value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
            return self._finalize(value, raw_X, name=self.__class__.__name__)

        Xp_batch = self._expand_pending_to_batch(Xp, batch_shape)
        pending_score = self._joint_score_per_output(Xp_batch)
        all_score = self._joint_score_per_output(torch.cat([Xp_batch, raw_X], dim=-2))
        score_per_output = all_score - pending_score
        score_per_output = self._apply_noise_to_q_aggregated_output_score(score_per_output, self._apply_input_transform(raw_X))
        value = self._aggregate_outputs(score_per_output)
        value = value - self._repulsion_penalty(raw_X)
        value = self._apply_levelset_objective(value, raw_X, name=self.__class__.__name__)
        return self._finalize(value, raw_X, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassICUAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassICUAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score_per_output = uncertainty.pow(2) * contour_weight
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassBoundaryVarianceAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassBoundaryVarianceAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        uncertainty = self._target_uncertainty(raw_X, p, mode=self.uncertainty_mode)
        score_per_output = uncertainty.pow(2) * _boundary_weight(p, self.threshold, bandwidth=self.bandwidth, eps=self.eps)
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassClassEntropyAcquisition(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassClassEntropyAcquisition,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        probs = self._mean_probs(raw_X)
        if self.target_class is None and self.output_target_classes is None:
            score_per_output = _class_entropy(probs, eps=self.eps)
        else:
            selected = self._target_prob_per_output(probs)
            score_per_output = -(selected.clamp_min(self.eps) * selected.clamp_min(self.eps).log())
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


class qHeteroMultiOutputMulticlassProbabilityOfExceedance(
    _HeteroMultiOutputMulticlassMixin,
    qMultiOutputMulticlassProbabilityOfExceedance,
):
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        raw_X = self._ensure_q_batch(X)
        Xt = self._ensure_q_batch(self._apply_input_transform(raw_X))
        p = self._target_prob_mean_per_output(raw_X)
        score_per_output = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score_per_output = self._apply_noise_to_score_per_output(score_per_output, Xt)
        return self._score_to_value(score_per_output, raw_X, Xt, name=self.__class__.__name__)


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
