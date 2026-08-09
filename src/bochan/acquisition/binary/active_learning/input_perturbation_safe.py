"""InputPerturbation-safe binary active-learning acquisitions."""

from __future__ import annotations

import torch
from torch import Tensor

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
)

from .single_output import qBinaryProbabilityVariance as _qBinaryProbabilityVariance


class _NominalDuplicatePenaltyMixin:
    """Evaluate hard duplicate semantics on nominal candidates, not replicas.

    Binary active-learning acquisitions score InputPerturbation-expanded rows in
    transformed feature space. Those rows are uncertainty-evaluation replicas,
    not additional optimization candidates. Soft distance penalties still use
    the transformed rows, while hard duplicate exclusion is evaluated on the
    raw nominal q-batch and raw reference points.
    """

    _raw_X_for_duplicate_penalty: Tensor | None = None

    def _soft_reference_penalty(
        self,
        Xt: Tensor,
        X_ref,
        *,
        weight: float,
        beta: float,
    ) -> Tensor:
        Xt = self._ensure_q_batch(Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if weight <= 0.0:
            return zeros

        ref_t = self._get_reference_in_feature_space(X_ref)
        if ref_t is None or ref_t.numel() == 0:
            return zeros

        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != Xt.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, ref.shape={tuple(ref_t.shape)}."
            )
        min_dist = torch.cdist(
            Xt.reshape(-1, Xt.shape[-1]),
            ref2d,
        ).min(dim=-1).values.reshape(*Xt.shape[:-1])
        return weight * torch.exp(-beta * min_dist)

    def _raw_duplicate_invalid_batch(self, raw_X: Tensor) -> Tensor:
        raw_X = self._ensure_q_batch(raw_X)
        invalid = torch.zeros(
            raw_X.shape[:-2],
            dtype=torch.bool,
            device=raw_X.device,
        )

        same = hard_same_batch_duplicate_penalty_per_point(
            raw_X,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        invalid = invalid | torch.isinf(same).any(dim=-1)

        pending = self._coerce_reference_to_tensor(
            getattr(self, "X_pending", None),
            ref=raw_X,
        )
        pending_hard = hard_reference_duplicate_penalty_per_point(
            raw_X,
            pending,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        invalid = invalid | torch.isinf(pending_hard).any(dim=-1)

        observed = self._coerce_reference_to_tensor(
            getattr(self, "X_observed", None),
            ref=raw_X,
        )
        observed_hard = hard_reference_duplicate_penalty_per_point(
            raw_X,
            observed,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        invalid = invalid | torch.isinf(observed_hard).any(dim=-1)
        return invalid

    def _candidate_penalty_per_point(self, Xt: Tensor) -> Tensor:
        raw_X = self._raw_X_for_duplicate_penalty
        if raw_X is None:
            return super()._candidate_penalty_per_point(Xt)

        Xt = self._ensure_q_batch(Xt)
        raw_X = self._ensure_q_batch(raw_X).to(device=Xt.device, dtype=Xt.dtype)
        if raw_X.shape[:-2] != Xt.shape[:-2]:
            raise RuntimeError(
                "Raw and transformed t-batch shapes must match for binary "
                "duplicate exclusion. "
                f"raw_X.shape={tuple(raw_X.shape)}, Xt.shape={tuple(Xt.shape)}."
            )

        penalty = self._soft_reference_penalty(
            Xt,
            getattr(self, "X_pending", None),
            weight=self.pending_penalty_weight,
            beta=self.pending_penalty_beta,
        )
        penalty = penalty + self._soft_reference_penalty(
            Xt,
            getattr(self, "X_observed", None),
            weight=self.observed_penalty_weight,
            beta=self.observed_penalty_beta,
        )

        invalid = self._raw_duplicate_invalid_batch(raw_X)
        while invalid.ndim < penalty.ndim:
            invalid = invalid.unsqueeze(-1)
        return torch.where(
            invalid.expand_as(penalty),
            torch.full_like(penalty, torch.inf),
            penalty,
        )

    def forward(self, X: Tensor) -> Tensor:
        self._raw_X_for_duplicate_penalty = self._ensure_q_batch(X)
        try:
            return super().forward(X)
        finally:
            self._raw_X_for_duplicate_penalty = None


class qBinaryProbabilityVariance(
    _NominalDuplicatePenaltyMixin,
    _qBinaryProbabilityVariance,
):
    """Binary probability variance with nominal-candidate duplicate semantics."""


__all__ = ["qBinaryProbabilityVariance"]
