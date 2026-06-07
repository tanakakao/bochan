from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .base import ClassReductionType, ReductionType, _MulticlassAcquisitionBase


class _MulticlassTargetProbabilityBase(_MulticlassAcquisitionBase):
    """Base class for target-class probability level-set acquisitions."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int] | None = None,
        threshold: float = 0.5,
        class_reduction: ClassReductionType = "mean",
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        eps: float = 1e-8,
        objective=None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            objective=objective,
        )
        self.target_class = target_class
        self.threshold = float(threshold)
        self.class_reduction = class_reduction

    def _target_prob(self, X: Tensor) -> Tensor:
        probs = self._mean_probs(X)
        return self._select_class_probs(
            probs,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
        )


class qMulticlassLatentStraddleAcquisition(_MulticlassTargetProbabilityBase):
    """Target-class probability straddle acquisition.

    Scores points close to ``p(target_class | x) = threshold`` and with high
    Bernoulli-style target probability variance.
    """

    def __init__(self, model, *, beta: float = 1.0, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.beta = float(beta)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        std = (p * (1.0 - p)).clamp_min(self.eps).sqrt()
        score = self.beta * std - (p - self.threshold).abs()
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassJointLatentStraddleAcquisition(qMulticlassLatentStraddleAcquisition):
    """q-batch straddle alias for multiclass models."""

    pass


class qMulticlassICUAcquisition(_MulticlassTargetProbabilityBase):
    """Integrated contour uncertainty style acquisition for target class probability."""

    def __init__(self, model, *, bandwidth: float = 0.10, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        uncertainty = (p * (1.0 - p)).clamp_min(self.eps)
        contour_weight = torch.exp(-0.5 * ((p - self.threshold) / max(self.bandwidth, self.eps)) ** 2)
        score = uncertainty * contour_weight
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassBoundaryVarianceAcquisition(_MulticlassTargetProbabilityBase):
    """Boundary-weighted target class variance acquisition."""

    def __init__(self, model, *, bandwidth: float = 0.15, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = float(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        variance = p * (1.0 - p)
        boundary_weight = torch.exp(-((p - self.threshold).abs() / max(self.bandwidth, self.eps)))
        score = variance * boundary_weight
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassClassEntropyAcquisition(_MulticlassAcquisitionBase):
    """Class entropy acquisition for multiclass level-set / boundary exploration."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._entropy(probs)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassProbabilityOfExceedance(_MulticlassTargetProbabilityBase):
    """Probability that target class probability exceeds a threshold.

    Since multiclass posterior already returns probabilities, this uses a smooth
    probability-space indicator around ``threshold``.
    """

    def __init__(self, model, *, tau: float = 0.02, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob(Xq)
        score = torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


__all__ = [
    "qMulticlassLatentStraddleAcquisition",
    "qMulticlassJointLatentStraddleAcquisition",
    "qMulticlassICUAcquisition",
    "qMulticlassBoundaryVarianceAcquisition",
    "qMulticlassClassEntropyAcquisition",
    "qMulticlassProbabilityOfExceedance",
]
