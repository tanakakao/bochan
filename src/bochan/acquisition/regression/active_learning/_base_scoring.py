"""Posterior scoring and duplicate/reference penalties."""

from __future__ import annotations

import torch
from torch import Tensor

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)

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
    def _same_batch_penalty_per_point(
        self,
        Xt: Tensor,
        *,
        raw_X: Tensor | None = None,
    ) -> Tensor:
        """Return soft diversity and hard duplicate penalties.

        ``Xt`` may contain ``q * n_w`` InputPerturbation evaluation rows. Those
        rows are uncertainty replicas of the same nominal candidates and must
        not be treated as additional q-batch candidates for hard duplicate
        exclusion. When ``raw_X`` is supplied, hard duplicate decisions are
        therefore made in the nominal candidate space and then broadcast to the
        transformed evaluation rows.
        """

        Xt = _ensure_q_batch(Xt)
        q_like = int(Xt.shape[-2])
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if q_like <= 1:
            return zeros
        if (
            self.same_batch_penalty_weight <= 0.0
            and self.hard_duplicate_penalty <= 0.0
            and not self.exclude_same_batch_duplicates
        ):
            return zeros

        d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
        eye = torch.eye(q_like, dtype=torch.bool, device=Xt.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye

        penalty = zeros
        if self.same_batch_penalty_weight > 0.0:
            soft = torch.exp(-self.same_batch_penalty_beta * d2)
            soft = torch.where(valid, soft, torch.zeros_like(soft))
            penalty = self.same_batch_penalty_weight * soft.sum(dim=-1)

        if raw_X is None:
            duplicate_pairs = valid & (d2 <= self.hard_duplicate_tol**2)
            if self.hard_duplicate_penalty > 0.0:
                penalty = penalty + self.hard_duplicate_penalty * duplicate_pairs.to(
                    dtype=Xt.dtype
                ).sum(dim=-1)
            if self.exclude_same_batch_duplicates:
                duplicate_batch = duplicate_pairs.any(dim=-1).any(
                    dim=-1,
                    keepdim=True,
                )
                penalty = torch.where(
                    duplicate_batch.expand_as(penalty),
                    torch.full_like(penalty, torch.inf),
                    penalty,
                )
            return penalty

        raw_X = _ensure_q_batch(raw_X).to(device=Xt.device, dtype=Xt.dtype)
        if raw_X.shape[:-2] != Xt.shape[:-2]:
            raise RuntimeError(
                "Raw and transformed t-batch shapes must match for duplicate "
                "exclusion. "
                f"raw_X.shape={tuple(raw_X.shape)}, Xt.shape={tuple(Xt.shape)}."
            )

        raw_q = int(raw_X.shape[-2])
        if self.hard_duplicate_penalty > 0.0 and raw_q > 1:
            raw_d2 = (
                (raw_X.unsqueeze(-2) - raw_X.unsqueeze(-3))
                .pow(2)
                .sum(dim=-1)
            )
            raw_eye = torch.eye(raw_q, dtype=torch.bool, device=raw_X.device)
            while raw_eye.ndim < raw_d2.ndim:
                raw_eye = raw_eye.unsqueeze(0)
            raw_pairs = (~raw_eye) & (raw_d2 <= self.hard_duplicate_tol**2)
            raw_penalty = self.hard_duplicate_penalty * raw_pairs.to(
                dtype=Xt.dtype
            ).sum(dim=-1)
            if q_like == raw_q:
                penalty = penalty + raw_penalty
            elif self.n_w is not None and q_like == raw_q * int(self.n_w):
                penalty = penalty + raw_penalty.unsqueeze(-1).expand(
                    *raw_penalty.shape,
                    int(self.n_w),
                ).reshape(*raw_penalty.shape[:-1], q_like)

        if self.exclude_same_batch_duplicates:
            raw_hard = hard_same_batch_duplicate_penalty_per_point(
                raw_X,
                enabled=True,
                tolerance=self.hard_duplicate_tol,
            )
            invalid_batch = torch.isinf(raw_hard).any(dim=-1, keepdim=True)
            penalty = torch.where(
                invalid_batch.expand_as(penalty),
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
        raw_X: Tensor | None = None,
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
        if not exclude_duplicates:
            return penalty

        if raw_X is None:
            duplicate_batch = (min_dist.square() <= self.hard_duplicate_tol**2).any(
                dim=-1,
                keepdim=True,
            )
        else:
            raw_X = _ensure_q_batch(raw_X).to(device=Xt.device, dtype=Xt.dtype)
            raw_ref = self._coerce_reference_to_tensor(ref, like=raw_X)
            raw_hard = hard_reference_duplicate_penalty_per_point(
                raw_X,
                raw_ref,
                enabled=True,
                tolerance=self.hard_duplicate_tol,
            )
            duplicate_batch = torch.isinf(raw_hard).any(dim=-1, keepdim=True)

        return torch.where(
            duplicate_batch.expand_as(penalty),
            torch.full_like(penalty, torch.inf),
            penalty,
        )

    def _total_penalty_per_point(
        self,
        Xt: Tensor,
        *,
        raw_X: Tensor | None = None,
    ) -> Tensor:
        return (
            self._same_batch_penalty_per_point(Xt, raw_X=raw_X)
            + self._reference_penalty_per_point(
                Xt,
                self.X_pending,
                weight=self.pending_penalty_weight,
                beta=self.pending_penalty_beta,
                exclude_duplicates=self.exclude_pending_duplicates,
                raw_X=raw_X,
            )
            + self._reference_penalty_per_point(
                Xt,
                self.X_observed,
                weight=self.observed_penalty_weight,
                beta=self.observed_penalty_beta,
                exclude_duplicates=self.exclude_observed_duplicates,
                raw_X=raw_X,
            )
        )
