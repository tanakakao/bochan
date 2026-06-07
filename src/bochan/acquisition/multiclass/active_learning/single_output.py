from __future__ import annotations

import itertools
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.multiclass.base import ReductionType, _MulticlassAcquisitionBase

LargeQStrategy = Literal["per_point", "truncate", "raise"]


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

    def _pointwise_bald_score(self, X: Tensor, *, apply_adjustments: bool = True) -> Tensor:
        Xq = self._ensure_q_batch(X)
        samples = self._sample_probs(Xq, num_samples=self.num_samples)
        mean_probs = samples.mean(dim=0)
        predictive_entropy = self._entropy(mean_probs)
        expected_entropy = self._entropy(samples).mean(dim=0)
        score = predictive_entropy - expected_entropy
        if apply_adjustments:
            score = self._apply_common_pointwise_adjustments(score, Xq)
        return score

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        score = self._pointwise_bald_score(Xq, apply_adjustments=True)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


class qMulticlassJointBALD(qMulticlassBALD):
    """Multiclass joint qBALD-style mutual information acquisition.

    For small ``q`` and class count ``C``, this computes the exact categorical
    joint entropy by enumerating ``C ** q`` class assignments:

    ``I[y_1, ..., y_q; w | X, D] = H[p(y_1, ..., y_q | X, D)]
    - E_w[H[p(y_1, ..., y_q | X, w)]].``

    If the exact state space is too large, ``large_q_strategy`` controls the
    fallback:

    - ``"per_point"``: sum pointwise BALD scores.
    - ``"truncate"``: exact joint BALD for the first ``max_joint_q`` points plus
      pointwise BALD for the remaining points.
    - ``"raise"``: raise ``RuntimeError``.
    """

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
        eps: float = 1e-8,
        objective=None,
    ) -> None:
        super().__init__(
            model=model,
            num_samples=num_samples,
            reduction="sum",
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            eps=eps,
            objective=objective,
        )
        self.max_joint_q = int(max_joint_q)
        self.max_joint_states = int(max_joint_states)
        self.large_q_strategy = large_q_strategy

    def _joint_predictive_entropy_exact(self, samples: Tensor) -> Tensor:
        """Compute exact joint predictive entropy from categorical samples.

        Args:
            samples: Probability samples with shape ``S x batch_shape x q x C``.

        Returns:
            Tensor with shape ``batch_shape``.
        """

        sample_shape = samples.shape
        if len(sample_shape) < 4:
            raise RuntimeError(f"samples must have shape S x batch_shape x q x C. Got {tuple(sample_shape)}.")
        q = int(sample_shape[-2])
        num_classes = int(sample_shape[-1])
        batch_shape = sample_shape[1:-2]
        entropy = samples.new_zeros(batch_shape)

        for state in itertools.product(range(num_classes), repeat=q):
            # p(y_1=c_1, ..., y_q=c_q | w_s) = prod_i p_i(c_i | w_s)
            p_state_per_sample = samples[..., 0, state[0]]
            for i in range(1, q):
                p_state_per_sample = p_state_per_sample * samples[..., i, state[i]]
            p_state = p_state_per_sample.mean(dim=0).clamp_min(self.eps)
            entropy = entropy - p_state * p_state.log()
        return entropy

    def _conditional_joint_entropy(self, samples: Tensor) -> Tensor:
        # Conditional on sampled model parameters, labels are independent across q.
        return self._entropy(samples).sum(dim=-1).mean(dim=0)

    def _pointwise_fallback_value(self, X: Tensor) -> Tensor:
        score = self._pointwise_bald_score(X, apply_adjustments=False)
        return score.sum(dim=-1)

    def _joint_bald_value(self, X: Tensor) -> Tensor:
        Xq = self._ensure_q_batch(X)
        q = int(Xq.shape[-2])
        samples = self._sample_probs(Xq, num_samples=self.num_samples)
        num_classes = int(samples.shape[-1])
        num_joint_states = int(num_classes**q)

        if q <= self.max_joint_q and num_joint_states <= self.max_joint_states:
            joint_entropy = self._joint_predictive_entropy_exact(samples)
            conditional_entropy = self._conditional_joint_entropy(samples)
            return joint_entropy - conditional_entropy

        if self.large_q_strategy == "raise":
            raise RuntimeError(
                f"Exact multiclass joint BALD is too large: q={q}, C={num_classes}, "
                f"C**q={num_joint_states}, max_joint_q={self.max_joint_q}, "
                f"max_joint_states={self.max_joint_states}."
            )

        if self.large_q_strategy == "per_point":
            return self._pointwise_fallback_value(Xq)

        if self.large_q_strategy == "truncate":
            k = min(q, self.max_joint_q)
            if k <= 0:
                return self._pointwise_fallback_value(Xq)
            first = Xq[..., :k, :]
            rest = Xq[..., k:, :]
            first_val = self._joint_bald_value(first)
            if rest.shape[-2] == 0:
                return first_val
            return first_val + self._pointwise_fallback_value(rest)

        raise ValueError(f"Unknown large_q_strategy: {self.large_q_strategy!r}.")

    def _aggregated_pending_penalty(self, X: Tensor) -> Tensor:
        if self.pending_penalty_weight <= 0:
            return X.new_zeros(X.shape[:-2])
        penalty = self._pending_penalty_per_point(X)
        if penalty.ndim == 0:
            return penalty
        return penalty.sum(dim=-1)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        value = self._joint_bald_value(Xq)
        value = value - self._aggregated_pending_penalty(Xq)
        # Joint BALD is already q-aggregated, so do not apply q-point objective here.
        return self._finalize(value, Xq, name=self.__class__.__name__)


class qMulticlassGreedyJointBALD(qMulticlassJointBALD):
    """Greedy multiclass joint qBALD acquisition.

    If ``X_pending`` is set, this returns the marginal joint information gain:

    ``JointBALD(X_pending ∪ X) - JointBALD(X_pending)``.

    This is useful for sequential or greedy batch construction. If no pending
    points are set, it reduces to ``qMulticlassJointBALD``.
    """

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
        Xq = self._ensure_q_batch(X)
        X_pending = getattr(self, "X_pending", None)
        if X_pending is None or torch.as_tensor(X_pending).numel() == 0:
            value = self._joint_bald_value(Xq)
            return self._finalize(value, Xq, name=self.__class__.__name__)

        Xp = torch.as_tensor(X_pending, dtype=Xq.dtype, device=Xq.device).detach()
        Xp = self._expand_pending_to_batch(Xp, Xq.shape[:-2])
        pending_value = self._joint_bald_value(Xp)
        all_value = self._joint_bald_value(torch.cat([Xp, Xq], dim=-2))
        value = all_value - pending_value
        # X_pending is used as the greedy context, so duplicate avoidance is
        # handled by marginalization. Additional pending penalty is not applied.
        return self._finalize(value, Xq, name=self.__class__.__name__)


class qMulticlassIntegratedPosteriorVarianceProxy(_MulticlassAcquisitionBase):
    """Lightweight multiclass IPV-style active learning proxy.

    With ``mc_points=None`` this reduces to local probability variance at the
    candidate points, matching the previous lightweight behavior.

    With ``mc_points`` provided, this computes a differentiable proxy for
    integrated posterior variance by weighting uncertainty over the integration
    points according to candidate proximity:

    ``score(x) = weighted_mean_z[sum_c p(c|z)(1-p(c|z))]``.

    This does not condition on fantasy labels. It is intentionally cheaper and
    optimizer-friendly compared with fantasy NIPV.
    """

    def __init__(
        self,
        model,
        *,
        mc_points: Tensor | None = None,
        integration_beta: float = 25.0,
        local_weight: float | None = None,
        integrated_weight: float = 1.0,
        reduction: ReductionType = "mean",
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        X_observed: Tensor | None = None,
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
        if mc_points is not None and mc_points.ndim != 2:
            raise ValueError(f"mc_points must have shape n_mc x d. Got {tuple(mc_points.shape)}.")
        self.register_buffer("mc_points", mc_points.detach() if mc_points is not None else None)
        self.integration_beta = float(integration_beta)
        self.local_weight = 1.0 if local_weight is None and mc_points is None else float(local_weight or 0.0)
        self.integrated_weight = float(integrated_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.X_observed: Tensor | None = None
        self.set_X_observed(X_observed)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        if X_observed is None:
            X_observed = self._resolve_observed_X()
        if X_observed is None:
            self.X_observed = None
            return
        self.X_observed = torch.as_tensor(X_observed).detach()

    def _resolve_observed_X(self) -> Tensor | None:
        for attr in ("train_inputs_raw", "train_X_original", "train_X"):
            x = getattr(self.model, attr, None)
            if x is not None:
                return x[0] if isinstance(x, tuple) else x
        train_inputs = getattr(self.model, "train_inputs", None)
        if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
            return train_inputs[0]
        return None

    @staticmethod
    def _as_reference_points(X_ref: Tensor | None, *, ref: Tensor) -> Tensor | None:
        if X_ref is None:
            return None
        X_ref = torch.as_tensor(X_ref, dtype=ref.dtype, device=ref.device)
        if X_ref.numel() == 0:
            return None
        if X_ref.ndim == 1:
            X_ref = X_ref.view(1, -1)
        if X_ref.ndim > 2:
            X_ref = X_ref.reshape(-1, X_ref.shape[-1])
        return X_ref

    def _mean_probs_q_batch(self, X: Tensor) -> Tensor:
        if X.ndim == 1:
            X = X.view(1, 1, -1)
        elif X.ndim == 2:
            X = X.unsqueeze(0)
        posterior = self.model.posterior(X)
        return self._normalize_probs(posterior.mean, X, name="integrated_mean_probs")

    def _observed_penalty_per_point(self, X: Tensor) -> Tensor:
        if self.observed_penalty_weight <= 0:
            return X.new_zeros(X.shape[:-1])
        X_obs = self._as_reference_points(self.X_observed, ref=X)
        if X_obs is None:
            return X.new_zeros(X.shape[:-1])
        dist = torch.cdist(X.reshape(-1, 1, X.shape[-1]), X_obs.unsqueeze(0)).min(dim=-1).values
        dist = dist.reshape(X.shape[:-1])
        return self.observed_penalty_weight * torch.exp(-self.observed_penalty_beta * dist)

    def _integrated_variance_score_per_point(self, X: Tensor) -> Tensor:
        if self.mc_points is None:
            return X.new_zeros(X.shape[:-1])

        mc_points = self.mc_points.to(device=X.device, dtype=X.dtype)
        mc_probs = self._mean_probs_q_batch(mc_points)
        mc_var = self._class_probability_variance(mc_probs).reshape(-1)

        d2 = torch.cdist(X.reshape(-1, X.shape[-1]), mc_points).pow(2)
        weights = torch.exp(-self.integration_beta * d2)
        score = (weights * mc_var.view(1, -1)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
        return score.reshape(X.shape[:-1])

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._prepare_eval()
        Xq = self._ensure_q_batch(X)
        probs = self._mean_probs(Xq)
        local_score = self._class_probability_variance(probs)
        integrated_score = self._integrated_variance_score_per_point(Xq)
        score = self.local_weight * local_score + self.integrated_weight * integrated_score
        score = score - self._observed_penalty_per_point(Xq)
        score = self._apply_common_pointwise_adjustments(score, Xq)
        return self._finalize(self._reduce_q(score), Xq, name=self.__class__.__name__)


__all__ = [
    "LargeQStrategy",
    "qMulticlassPredictiveEntropy",
    "qMulticlassProbabilityVariance",
    "qMulticlassMarginUncertainty",
    "qMulticlassBALD",
    "qMulticlassJointBALD",
    "qMulticlassGreedyJointBALD",
    "qMulticlassIntegratedPosteriorVarianceProxy",
]
