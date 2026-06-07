from __future__ import annotations

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .base import ReductionType, _MulticlassAcquisitionBase


class qMulticlassPredictiveEntropy(_MulticlassAcquisitionBase):
    """Multiclass predictive entropy acquisition.

    Selects points with high class-probability entropy:
    ``H[y | x] = -sum_c p_c log p_c``.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._entropy(probs)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassProbabilityVariance(_MulticlassAcquisitionBase):
    """Multiclass probability variance acquisition.

    Uses ``sum_c p_c(1 - p_c)`` as a lightweight uncertainty score.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._class_probability_variance(probs)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassMarginUncertainty(_MulticlassAcquisitionBase):
    """Multiclass margin uncertainty acquisition.

    Uses ``1 - (p_top1 - p_top2)``. Large values indicate ambiguous class boundaries.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._margin_uncertainty(probs)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassBALD(_MulticlassAcquisitionBase):
    """Multiclass BALD-style mutual information acquisition.

    Computes ``H[E_w p(y|x,w)] - E_w[H[p(y|x,w)]]`` using probability posterior samples.
    """

    def __init__(
        self,
        model,
        *,
        num_samples: int = 32,
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
        self.num_samples = int(num_samples)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        samples = self._sample_probs(Xq, num_samples=self.num_samples)
        mean_probs = samples.mean(dim=0)
        predictive_entropy = self._entropy(mean_probs)
        expected_entropy = self._entropy(samples).mean(dim=0)
        score = predictive_entropy - expected_entropy
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassJointBALD(qMulticlassBALD):
    """Practical q-batch BALD alias for multiclass models.

    The first implementation uses pointwise BALD with q aggregation, which is stable
    for arbitrary q. A true joint categorical qBALD can be added later if needed.
    """

    pass


class qMulticlassGreedyJointBALD(qMulticlassBALD):
    """Greedy joint BALD alias for multiclass models.

    This currently uses the same robust pointwise BALD score as ``qMulticlassBALD``.
    """

    pass


class qMulticlassIntegratedPosteriorVarianceProxy(_MulticlassAcquisitionBase):
    """Lightweight IPV-style proxy for multiclass active learning.

    This acquisition maximizes the local probability variance. It is intended as
    a cheap proxy until true fantasy-based NIPV is implemented for multiclass models.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        score = self._class_probability_variance(probs)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


__all__ = [
    "qMulticlassPredictiveEntropy",
    "qMulticlassProbabilityVariance",
    "qMulticlassMarginUncertainty",
    "qMulticlassBALD",
    "qMulticlassJointBALD",
    "qMulticlassGreedyJointBALD",
    "qMulticlassIntegratedPosteriorVarianceProxy",
]
