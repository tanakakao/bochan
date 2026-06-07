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
        pending = _align_pointwise_to_reference(self._pending_penalty_per_point(Xt), score, name=f"{name}.pending")
        observed = _align_pointwise_to_reference(self._observed_penalty_per_point(Xt), score, name=f"{name}.observed")
        score = score - pending - observed
        score = self._apply_active_objective(score, raw_X, name=name)
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


class qMulticlassJointBALD(qMulticlassBALD):
    """Multiclass joint qBALD-style mutual information acquisition."""

    def __init__(
        self,
        model,
        *,
        num_samples: int = 32,
        max_joint_q: int = 5,
        max_joint_states: int = 4096,
        large_q_strategy: LargeQStrategy = "per_point",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        eps: float = 1e-8,
        objective=None,
        sampler: Optional[MCSampler] = None,
        apply_softmax_if_needed: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            num_samples=num_samples,
            sampler=sampler,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_observed=X_observed,
            eps=eps,
            objective=objective,
            apply_softmax_if_needed=apply_softmax_if_needed,
        )
        self.max_joint_q = int(max_joint_q)
        self.max_joint_states = int(max_joint_states)
        self.large_q_strategy = large_q_strategy

    def _joint_predictive_entropy_exact(self, samples: Tensor) -> Tensor:
        if len(samples.shape) < 4:
            raise RuntimeError(f"samples must have shape S x batch_shape x q x C. Got {tuple(samples.shape)}.")
        q = int(samples.shape[-2])
        num_classes = int(samples.shape[-1])
        batch_shape = samples.shape[1:-2]
        entropy = samples.new_zeros(batch_shape)
        for state in itertools.product(range(num_classes), repeat=q):
            p_state_per_sample = samples[..., 0, state[0]]
            for i in range(1, q):
                p_state_per_sample = p_state_per_sample * samples[..., i, state[i]]
            p_state = p_state_per_sample.mean(dim=0).clamp_min(self.eps)
            entropy = entropy - p_state * p_state.log()
        return entropy

    def _conditional_joint_entropy(self, samples: Tensor) -> Tensor:
        return self._entropy(samples).sum(dim=-1).mean(dim=0)

    def _pointwise_fallback_value(self, X: Tensor) -> Tensor:
        return self._pointwise_bald_score(X).sum(dim=-1)

    def _joint_bald_value(self, X: Tensor) -> Tensor:
        Xq = ensure_q_batch(X)
        q = int(Xq.shape[-2])
        samples = self._sample_probs(Xq)
        num_classes = int(samples.shape[-1])
        num_joint_states = int(num_classes**q)

        if q <= self.max_joint_q and num_joint_states <= self.max_joint_states:
            joint_entropy = self._joint_predictive_entropy_exact(samples)
            conditional_entropy = self._conditional_joint_entropy(samples)
            return joint_entropy - conditional_entropy

        if self.large_q_strategy == "raise":
            raise RuntimeError(
                f"Exact multiclass joint BALD is too large: q={q}, C={num_classes}, "
                f"C**q={num_joint_states}, max_joint_q={self.max_joint_q}, max_joint_states={self.max_joint_states}."
            )
        if self.large_q_strategy == "per_point":
            return self._pointwise_fallback_value(Xq)
        if self.large_q_strategy == "truncate":
            k = min(q, self.max_joint_q)
            first = Xq[..., :k, :]
            rest = Xq[..., k:, :]
            first_val = self._joint_bald_value(first)
            if rest.shape[-2] == 0:
                return first_val
            return first_val + self._pointwise_fallback_value(rest)
        raise ValueError(f"Unknown large_q_strategy: {self.large_q_strategy!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        value = self._joint_bald_value(raw_X)
        value = value - self._joint_penalty(raw_X, Xt)
        value = self._apply_active_objective(value, raw_X, name=self.__class__.__name__)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassGreedyJointBALD(qMulticlassJointBALD):
    """Greedy multiclass joint qBALD acquisition."""

    @staticmethod
    def _expand_pending_to_batch(X_pending: Tensor, batch_shape: torch.Size) -> Tensor:
        if X_pending.ndim == 1:
            X_pending = X_pending.view(1, -1)
        if X_pending.ndim == 2:
            m, d = X_pending.shape
            return X_pending.view(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        if X_pending.ndim >= 3:
            m, d = X_pending.shape[-2], X_pending.shape[-1]
            leading = X_pending.shape[:-2]
            if leading == batch_shape:
                return X_pending
            return X_pending.reshape(*([1] * len(batch_shape)), m, d).expand(*batch_shape, m, d)
        raise ValueError(f"Unexpected X_pending shape: {tuple(X_pending.shape)}")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or torch.as_tensor(X_pending).numel() == 0:
            value = self._joint_bald_value(raw_X)
            value = value - self._observed_penalty_per_point(Xt).sum(dim=-1)
            value = value - self._same_batch_penalty(Xt)
            value = self._apply_active_objective(value, raw_X, name=self.__class__.__name__)
            return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)

        Xp = torch.as_tensor(X_pending, dtype=raw_X.dtype, device=raw_X.device).detach()
        Xp = self._expand_pending_to_batch(Xp, raw_X.shape[:-2])
        pending_value = self._joint_bald_value(Xp)
        all_value = self._joint_bald_value(torch.cat([Xp, raw_X], dim=-2))
        value = all_value - pending_value
        value = value - self._observed_penalty_per_point(Xt).sum(dim=-1)
        value = value - self._same_batch_penalty(Xt)
        value = self._apply_active_objective(value, raw_X, name=self.__class__.__name__)
        return _finalize_multiclass_acq_output_to_batch(value, raw_X, name=self.__class__.__name__)


class qMulticlassIntegratedPosteriorVarianceProxy(_MulticlassActiveLearningBase):
    """Differentiable multiclass IPV-style active learning proxy."""

    def __init__(
        self,
        model,
        *,
        mc_points: Tensor | None = None,
        integration_beta: float = 25.0,
        local_weight: float | None = None,
        integrated_weight: float = 1.0,
        num_samples: int = 128,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
        eps: float = 1e-8,
        objective=None,
        sampler: Optional[MCSampler] = None,
        apply_softmax_if_needed: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            num_samples=num_samples,
            sampler=sampler,
            reduction=reduction,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_observed=X_observed,
            eps=eps,
            objective=objective,
            apply_softmax_if_needed=apply_softmax_if_needed,
        )
        if mc_points is not None and mc_points.ndim != 2:
            raise ValueError(f"mc_points must have shape n_mc x d. Got {tuple(mc_points.shape)}.")
        self.register_buffer("mc_points", mc_points.detach() if mc_points is not None else None)
        self.integration_beta = float(integration_beta)
        self.local_weight = 1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        self.integrated_weight = float(integrated_weight)

    def _integrated_variance_score_per_point(self, raw_X: Tensor, Xt: Tensor) -> Tensor:
        if self.mc_points is None:
            return raw_X.new_zeros(raw_X.shape[:-1])
        mc_points = self.mc_points.to(device=raw_X.device, dtype=raw_X.dtype)
        mc_probs = self._mean_probs(mc_points.unsqueeze(0))
        mc_var = self._class_probability_variance(mc_probs).reshape(-1)
        mc_points_t = self._apply_input_transform(mc_points.unsqueeze(0)).reshape(-1, Xt.shape[-1])
        d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), mc_points_t).pow(2)
        weights = torch.exp(-self.integration_beta * d2)
        score = (weights * mc_var.view(1, -1)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
        return score.reshape(Xt.shape[:-1])

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        raw_X = ensure_q_batch(X)
        Xt = self._apply_input_transform(raw_X)
        probs = self._mean_probs(raw_X)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_score_per_point(raw_X, Xt)
        integrated_score = _align_pointwise_to_reference(integrated_score, local_score, name="IPV.integrated_score")
        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        return self._score_to_value(score, raw_X, Xt, name=self.__class__.__name__)


__all__ = [
    "LargeQStrategy",
    "_MulticlassActiveLearningBase",
    "qMulticlassPredictiveEntropy",
    "qMulticlassProbabilityVariance",
    "qMulticlassMarginUncertainty",
    "qMulticlassBALD",
    "qMulticlassJointBALD",
    "qMulticlassGreedyJointBALD",
    "qMulticlassIntegratedPosteriorVarianceProxy",
]
