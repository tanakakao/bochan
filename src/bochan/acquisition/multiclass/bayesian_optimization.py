from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .base import ClassReductionType, ReductionType, _MulticlassAcquisitionBase


class _MulticlassTargetClassBOBase(_MulticlassAcquisitionBase):
    """Base for multiclass BO acquisitions over target-class probability."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int],
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
        self.class_reduction = class_reduction

    def _target_prob_samples(self, X: Tensor, *, num_samples: int) -> Tensor:
        samples = self._sample_probs(X, num_samples=num_samples)
        return self._select_class_probs(
            samples,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
        )

    def _target_prob_mean(self, X: Tensor) -> Tensor:
        probs = self._mean_probs(X)
        return self._select_class_probs(
            probs,
            target_class=self.target_class,
            class_reduction=self.class_reduction,
        )


class qMulticlassProbabilityOfFeasibility(_MulticlassTargetClassBOBase):
    """Probability of target class feasibility.

    Maximizes ``p(target_class | x)`` or a smooth indicator that this probability
    exceeds ``threshold``.
    """

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int],
        threshold: float | None = None,
        tau: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.threshold = None if threshold is None else float(threshold)
        self.tau = float(tau)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        p = self._target_prob_mean(Xq)
        score = p if self.threshold is None else torch.sigmoid((p - self.threshold) / max(self.tau, self.eps))
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassExpectedImprovement(_MulticlassTargetClassBOBase):
    """Expected improvement for target-class probability.

    The objective is ``p(target_class | x)``. ``best_f`` should be the current best
    observed / predicted target-class probability.
    """

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int],
        best_f: float | Tensor,
        num_samples: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        samples = self._target_prob_samples(Xq, num_samples=self.num_samples)
        best_q = samples.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        value = (best_q - best_f).clamp_min(0.0).mean(dim=0)
        return self._finalize(value, Xq, name=self.__class__.__name__)


class qMulticlassProbabilityOfImprovement(_MulticlassTargetClassBOBase):
    """Probability of improvement for target-class probability."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int],
        best_f: float | Tensor,
        num_samples: int = 128,
        tau: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        samples = self._target_prob_samples(Xq, num_samples=self.num_samples)
        best_q = samples.max(dim=-1).values
        best_f = self.best_f.to(best_q)
        tau = self.tau.to(best_q).clamp_min(self.eps)
        value = torch.sigmoid((best_q - best_f) / tau).mean(dim=0)
        return self._finalize(value, Xq, name=self.__class__.__name__)


class qMulticlassUpperConfidenceBound(_MulticlassTargetClassBOBase):
    """Upper confidence bound for target-class probability."""

    def __init__(
        self,
        model,
        *,
        target_class: int | Sequence[int],
        beta: float | Tensor = 2.0,
        num_samples: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(model=model, target_class=target_class, **kwargs)
        self.num_samples = int(num_samples)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        samples = self._target_prob_samples(Xq, num_samples=self.num_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=False).clamp_min(self.eps)
        beta = self.beta.to(mean)
        score = mean + beta.sqrt() * std
        score = self._apply_common_pointwise_adjustments(score, Xq)
        value = score.max(dim=-1).values
        return self._finalize(value, Xq, name=self.__class__.__name__)


__all__ = [
    "qMulticlassProbabilityOfFeasibility",
    "qMulticlassExpectedImprovement",
    "qMulticlassProbabilityOfImprovement",
    "qMulticlassUpperConfidenceBound",
]
