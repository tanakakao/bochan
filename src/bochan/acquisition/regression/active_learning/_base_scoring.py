"""Posterior scoring and duplicate/reference penalties."""

from __future__ import annotations

import torch
from torch import Tensor

from ._base_common import _ensure_q_batch


class _RegressionScoringMixin:
    # ------------------------------------------------------------
    # Posterior scores
    # ------------------------------------------------------------
    def _posterior_mean_variance(
        self,
        X: Tensor,
        *,
        observation_noise: bool | Tensor = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean / variance as pointwise tensors and transformed X."""
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        post = self.model.posterior(Xq, observation_noise=observation_noise)
        Xt = self._apply_input_transform_for_distance(Xq)

        mean = self._reduce_outputs_if_needed(post.mean, Xt, name="posterior.mean")
        var = self._reduce_outputs_if_needed(post.variance, Xt, name="posterior.variance")
        var = var.clamp_min(self.eps)

        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")
        var = self._align_pointwise_score_to_X(var, Xt, name="posterior.variance")

        return mean, var, Xt

    def _posterior_variance_score(self, X: Tensor) -> tuple[Tensor, Tensor]:
        _, var, Xt = self._posterior_mean_variance(X, observation_noise=False)
        return var, Xt

    # ------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------
    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        q = int(Xt.shape[-2])
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if q <= 1:
            return zeros
        if (
            self.same_batch_penalty_weight <= 0.0
            and self.hard_duplicate_penalty <= 0.0
            and not self.exclude_same_batch_duplicates
        ):
            return zeros

        d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
        eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye

        penalty = zeros
        if self.same_batch_penalty_weight > 0.0:
            soft = torch.exp(-self.same_batch_penalty_beta * d2)
            soft = torch.where(valid, soft, torch.zeros_like(soft))
            penalty = self.same_batch_penalty_weight * soft.sum(dim=-1)

        duplicate_pairs = valid & (d2 <= self.hard_duplicate_tol)
        if self.hard_duplicate_penalty > 0.0:
            penalty = penalty + self.hard_duplicate_penalty * duplicate_pairs.to(
                dtype=Xt.dtype
            ).sum(dim=-1)

        if self.exclude_same_batch_duplicates:
            duplicate_batch = duplicate_pairs.any(dim=-1).any(dim=-1, keepdim=True)
            penalty = torch.where(
                duplicate_batch.expand_as(penalty),
                torch.full_like(penalty, torch.inf),
                penalty,
            )
        return penalty

    def _reference_penalty_per_point(
        self,
        Xt: Tensor,
        ref,
        *,
        weight: float,
        beta: float,
        exclude_duplicates: bool = False,
    ) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if weight <= 0.0 and not exclude_duplicates:
            return zeros

        ref_t = self._reference_to_distance_space(ref, like=Xt)
        if ref_t is None or ref_t.numel() == 0:
            return zeros

        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != Xt.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, ref_transformed.shape={tuple(ref_t.shape)}."
            )

        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), ref2d)
        min_dist = dist.min(dim=-1).values.reshape(*Xt.shape[:-1])
        penalty = (
            weight * torch.exp(-beta * min_dist)
            if weight > 0.0
            else zeros
        )
        if exclude_duplicates:
            duplicate_batch = (min_dist.square() <= self.hard_duplicate_tol).any(
                dim=-1,
                keepdim=True,
            )
            penalty = torch.where(
                duplicate_batch.expand_as(penalty),
                torch.full_like(penalty, torch.inf),
                penalty,
            )
        return penalty

    def _total_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return (
            self._same_batch_penalty_per_point(Xt)
            + self._reference_penalty_per_point(
                Xt,
                self.X_pending,
                weight=self.pending_penalty_weight,
                beta=self.pending_penalty_beta,
                exclude_duplicates=self.exclude_pending_duplicates,
            )
            + self._reference_penalty_per_point(
                Xt,
                self.X_observed,
                weight=self.observed_penalty_weight,
                beta=self.observed_penalty_beta,
                exclude_duplicates=self.exclude_observed_duplicates,
            )
        )

