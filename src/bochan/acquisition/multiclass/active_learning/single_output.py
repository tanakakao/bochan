from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ReductionType
from bochan.acquisition.multiclass.bayesian_optimization.single_output import (
    _MulticlassProbabilityBOBase,
    _finalize_multiclass_acq_output_to_batch,
    ensure_q_batch,
)

LargeQStrategy = Literal["per_point", "truncate", "raise"]


def _class_entropy(probs: Tensor, *, eps: float) -> Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _class_probability_variance(probs: Tensor) -> Tensor:
    return (probs * (1.0 - probs)).sum(dim=-1)


def _margin_uncertainty(probs: Tensor) -> Tensor:
    top2 = probs.topk(k=2, dim=-1).values
    return 1.0 - (top2[..., 0] - top2[..., 1])


def _reduce_extra_leading_dims_to_raw_X(score: Tensor, raw_X: Tensor, *, name: str) -> Tensor:
    """Reduce sample-like leading dims so score aligns with raw_X t-batch/q dims.

    DeepGP posteriors can leave an additional sample-like leading dimension in
    posterior means / probability tensors. For example, with
    ``raw_X.shape == (32, 1, 2)``, entropy can have shape ``(10, 32)`` or
    ``(10, 32, 1)``. Acquisition optimizers, however, expect either
    ``batch_shape`` or ``batch_shape x q``. This helper averages only those
    extra leading dimensions while preserving the t-batch and q dimensions.
    """

    raw_X = ensure_q_batch(raw_X)
    batch_shape = tuple(raw_X.shape[:-2])
    q = int(raw_X.shape[-2])

    if score.shape == raw_X.shape[:-2]:
        return score

    if score.ndim >= 1 and tuple(score.shape[:-1]) == batch_shape and score.shape[-1] == q:
        return score

    if len(batch_shape) > 0:
        batch_ndim = len(batch_shape)
        if score.ndim >= batch_ndim and tuple(score.shape[-batch_ndim:]) == batch_shape:
            extra_ndim = score.ndim - batch_ndim
            if extra_ndim > 0:
                return score.mean(dim=tuple(range(extra_ndim)))
            return score

        target_with_q = batch_shape + (q,)
        target_ndim = len(target_with_q)
        if score.ndim >= target_ndim and tuple(score.shape[-target_ndim:]) == target_with_q:
            extra_ndim = score.ndim - target_ndim
            if extra_ndim > 0:
                return score.mean(dim=tuple(range(extra_ndim)))
            return score

    elif score.ndim >= 2 and score.shape[-1] == q:
        return score.mean(dim=tuple(range(score.ndim - 1)))

    return score


def _align_pointwise_to_reference(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Align a pointwise tensor to a reference score tensor."""

    if value.ndim >= 1 and value.shape[-1] == 1 and reference.ndim >= 1 and reference.shape[-1] != 1:
        value = value.squeeze(-1)
    while value.ndim > reference.ndim:
        if value.shape[-1] == 1:
            value = value.squeeze(-1)
        else:
            value = value.mean(dim=-1)
    if value.shape == reference.shape:
        return value.to(reference)
    if value.shape == reference.shape[:-1]:
        return value.unsqueeze(-1).expand_as(reference).to(reference)
    if value.ndim < reference.ndim and value.ndim > 0:
        tail_shape = tuple(reference.shape[-value.ndim:])
        if tuple(value.shape) == tail_shape:
            view_shape = (1,) * (reference.ndim - value.ndim) + tuple(value.shape)
            return value.reshape(view_shape).expand_as(reference).to(reference)
    if value.ndim == reference.ndim and value.shape[:-1] == reference.shape[:-1]:
        q_ref = reference.shape[-1]
        q_value = value.shape[-1]
        if q_ref % q_value == 0:
            return value.repeat_interleave(q_ref // q_value, dim=-1).to(reference)
        if q_value % q_ref == 0:
            return value.reshape(*reference.shape[:-1], q_ref, q_value // q_ref).mean(dim=-1).to(reference)
    if value.numel() == reference.numel():
        return value.reshape_as(reference).to(reference)
    if value.numel() == 1:
        return value.reshape(()).expand_as(reference).to(reference)
    raise RuntimeError(
        f"{name}: cannot align pointwise value. "
        f"value.shape={tuple(value.shape)}, reference.shape={tuple(reference.shape)}."
    )


class _MulticlassActiveLearningBase(_MulticlassProbabilityBOBase):
    """Complete base for multiclass single-output active learning acquisitions.

    This base mirrors the more complete binary / ordinal implementations by
    supporting latent logits via softmax, probability-posterior models,
    ``class_probs`` models, input-transform-aware pending / observed / same-batch
    penalties, and optional pointwise score objectives.
    """

    def __init__(
        self,
        model,
        *,
        num_samples: int = 128,
        sampler: Optional[MCSampler] = None,
        reduction: ReductionType = "mean",
        apply_softmax_if_needed: bool = True,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        eps: float = 1e-8,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(num_samples)]))
        super().__init__(
            model=model,
            sampler=sampler,
            target_class=None,
            class_reduction="mean",
            apply_softmax_if_needed=apply_softmax_if_needed,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_observed=X_observed,
            eps=eps,
            objective=None,
        )
        self.num_samples = int(num_samples)
        self.active_objective = objective

    # Backward-compatible aliases used by older hetero wrappers and notebooks.
    def _ensure_q_batch(self, X: Tensor) -> Tensor:
        return ensure_q_batch(X)

    def _mean_probs(self, X: Tensor) -> Tensor:
        return self._posterior_mean_probs(X)

    def _sample_probs(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        # ``num_samples`` is kept for API compatibility. The actual sample shape
        # comes from ``self.sampler``.
        return self._posterior_samples_as_probs(X)

    def _entropy(self, probs: Tensor) -> Tensor:
        return _class_entropy(probs, eps=self.eps)

    def _class_probability_variance(self, probs: Tensor) -> Tensor:
        return _class_probability_variance(probs)

    def _margin_uncertainty(self, probs: Tensor) -> Tensor:
        return _margin_uncertainty(probs)

    def _apply_active_objective(self, score: Tensor, raw_X: Tensor, *, name: str) -> Tensor:
        if self.active_objective is None:
            return score
        try:
            out = self.active_objective(score, X=raw_X)
        except TypeError:
            out = self.active_objective(score)
        if not torch.is_tensor(out):
            raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
        return out

    def _score_to_value(self, score: Tensor, raw_X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        score = _reduce_extra_leading_dims_to_raw_X(score, raw_X, name=name)
        pending = _align_pointwise_to_reference(self._pending_penalty_per_point(Xt), score, name=f"{name}.pending")
        observed = _align_pointwise_to_reference(self._observed_penalty_per_point(Xt), score, name=f"{name}.observed")
        score = score - pending - observed
        score = self._apply_active_objective(score, raw_X, name=name)
        score = _reduce_extra_leading_dims_to_raw_X(score, raw_X, name=f"{name}.objective")
        value = score if score.shape == raw_X.shape[:-2] else self._reduce_q(score)
        value = value - self._same_batch_penalty(Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=name)

    def _joint_penalty(self, raw_X: Tensor, Xt: Tensor) -> Tensor:
        penalty = self._pending_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._observed_penalty_per_point(Xt).sum(dim=-1)
        penalty = penalty + self._same_batch_penalty(Xt)
        return penalty


class qMulticlassPredictiveEntropy(_MulticlassActiveLearningBase):
    """Multiclass predictive entropy acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._entropy(probs)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassProbabilityVariance(_MulticlassActiveLearningBase):
    """Multiclass probability variance acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._class_probability_variance(probs)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassMarginUncertainty(_MulticlassActiveLearningBase):
    """Multiclass margin uncertainty acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._margin_uncertainty(probs)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassBALD(_MulticlassActiveLearningBase):
    """Multiclass BALD-style mutual information acquisition."""

    def _pointwise_bald_score(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        samples = self._sample_probs(Xq)
        mean_probs = samples.mean(dim=0)
        predictive_entropy = self._entropy(mean_probs)
        expected_entropy = self._entropy(samples).mean(dim=0)
        return predictive_entropy - expected_entropy

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        score = self._pointwise_bald_score(raw_X)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassIntegratedPosteriorVarianceProxy(_MulticlassActiveLearningBase):
    """Cheap pointwise IPV proxy based on class-probability variance."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        score = self._class_probability_variance(probs)
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


class qMulticlassJointBALD(qMulticlassBALD):
    """Greedy joint BALD approximation for multiclass q-batches.

    Exact JointBALD is expensive for large q. This implementation computes a
    pointwise BALD score and then uses a log-det diversity correction based on
    posterior mean class probabilities.
    """

    def __init__(
        self,
        model,
        *,
        diversity_weight: float = 0.05,
        diversity_jitter: float = 1e-6,
        max_q_for_exact: int = 16,
        large_q_strategy: LargeQStrategy = "per_point",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.diversity_weight = float(diversity_weight)
        self.diversity_jitter = float(diversity_jitter)
        self.max_q_for_exact = int(max_q_for_exact)
        self.large_q_strategy = large_q_strategy

    def _diversity_bonus(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        q = Xq.shape[-2]
        if q <= 1:
            return Xq.new_zeros(Xq.shape[:-2])
        if q > self.max_q_for_exact:
            if self.large_q_strategy == "per_point":
                return Xq.new_zeros(Xq.shape[:-2])
            if self.large_q_strategy == "truncate":
                Xq = Xq[..., : self.max_q_for_exact, :]
                q = Xq.shape[-2]
            elif self.large_q_strategy == "raise":
                raise RuntimeError(
                    f"{self.__class__.__name__}: q={q} is too large for exact diversity. "
                    f"Set large_q_strategy='per_point' or increase max_q_for_exact."
                )
            else:
                raise ValueError(f"Unknown large_q_strategy: {self.large_q_strategy!r}.")
        probs = self._mean_probs(Xq)
        probs = probs - probs.mean(dim=-1, keepdim=True)
        K = probs @ probs.transpose(-1, -2)
        K = K + self.diversity_jitter * torch.eye(q, device=K.device, dtype=K.dtype)
        return torch.linalg.slogdet(K).logabsdet

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        pointwise = self._pointwise_bald_score(raw_X)
        pointwise = _reduce_extra_leading_dims_to_raw_X(pointwise, raw_X, name=self.__class__.__name__)
        value = self._reduce_q(pointwise)
        value = value + self.diversity_weight * self._diversity_bonus(raw_X)
        value = value - self._joint_penalty(raw_X, Xt)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassGreedyJointBALD(qMulticlassJointBALD):
    """Alias for greedy JointBALD-style q-batch scoring."""


__all__ = [
    "qMulticlassPredictiveEntropy",
    "qMulticlassBALD",
    "qMulticlassJointBALD",
    "qMulticlassGreedyJointBALD",
    "qMulticlassIntegratedPosteriorVarianceProxy",
    "qMulticlassMarginUncertainty",
    "qMulticlassProbabilityVariance",
]
