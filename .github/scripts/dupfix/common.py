from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_regex_once(text: str, pattern: str, new: str, *, label: str) -> str:
    updated, count = re.subn(pattern, new, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one regex match, found {count}")
    return updated


DUPLICATE_MODULE = '''"""Scale-independent hard duplicate exclusion for acquisition functions.

The helpers return additive penalties.  They only invalidate exact / tolerance
matches and do not alter scores for nearby distinct candidates.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("duplicate tolerance must be non-negative.")
    return tolerance


def hard_same_batch_duplicate_penalty_per_point(
    X: Tensor,
    *,
    enabled: bool = True,
    tolerance: float = 1e-8,
) -> Tensor:
    """Return ``inf`` for every point in a q-batch containing duplicates."""

    X = _ensure_q_batch(X)
    zeros = X.new_zeros(X.shape[:-1])
    q = int(X.shape[-2])
    if not enabled or q <= 1:
        return zeros

    tolerance = _validate_tolerance(tolerance)
    d2 = (X.unsqueeze(-2) - X.unsqueeze(-3)).pow(2).sum(dim=-1)
    eye = torch.eye(q, dtype=torch.bool, device=X.device)
    while eye.ndim < d2.ndim:
        eye = eye.unsqueeze(0)
    duplicate_pairs = (~eye) & (d2 <= tolerance)
    duplicate_batch = duplicate_pairs.any(dim=-1).any(dim=-1, keepdim=True)
    return torch.where(
        duplicate_batch.expand_as(zeros),
        torch.full_like(zeros, torch.inf),
        zeros,
    )


def hard_reference_duplicate_penalty_per_point(
    X: Tensor,
    X_ref: Tensor | None,
    *,
    enabled: bool = True,
    tolerance: float = 1e-8,
) -> Tensor:
    """Return ``inf`` when any candidate duplicates a reference point."""

    X = _ensure_q_batch(X)
    zeros = X.new_zeros(X.shape[:-1])
    if not enabled or X_ref is None or X_ref.numel() == 0:
        return zeros

    tolerance = _validate_tolerance(tolerance)
    X_ref = torch.as_tensor(X_ref, device=X.device, dtype=X.dtype)
    if X_ref.ndim == 1:
        X_ref = X_ref.unsqueeze(0)
    X_ref = X_ref.reshape(-1, X_ref.shape[-1])
    if X_ref.shape[-1] != X.shape[-1]:
        raise RuntimeError(
            "Reference feature dimension mismatch: "
            f"X.shape={tuple(X.shape)}, X_ref.shape={tuple(X_ref.shape)}."
        )

    d2 = torch.cdist(X.reshape(-1, X.shape[-1]), X_ref).pow(2)
    d2 = d2.reshape(*X.shape[:-1], X_ref.shape[-2])
    duplicate_batch = (d2 <= tolerance).any(dim=-1).any(dim=-1, keepdim=True)
    return torch.where(
        duplicate_batch.expand_as(zeros),
        torch.full_like(zeros, torch.inf),
        zeros,
    )


__all__ = [
    "hard_reference_duplicate_penalty_per_point",
    "hard_same_batch_duplicate_penalty_per_point",
]
'''


BINARY_PENDING_SECTION = '''    # =========================================================
    # pending / duplicate penalty
    # =========================================================
    def _get_pending_in_feature_space(self) -> Optional[Tensor]:
        """X_pending を現在の candidate と同じ feature space に写す。"""
        Xp = getattr(self, "X_pending", None)
        if Xp is None or Xp.numel() == 0:
            return None
        return self._apply_input_transform(Xp)

    def _same_batch_duplicate_penalty_per_point(self, X: Tensor) -> Tensor:
        return hard_same_batch_duplicate_penalty_per_point(
            X,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )

    def _pending_penalty_per_point(self, X: Tensor) -> Tensor:
        """Return soft pending repulsion plus scale-independent hard exclusion."""
        X = self._ensure_q_batch(X)
        zeros = X.new_zeros(X.shape[:-1])
        Xp = self._get_pending_in_feature_space()
        if Xp is None or Xp.numel() == 0:
            return zeros

        d = X.shape[-1]
        X2d = X.reshape(-1, d)
        Xp2d = Xp.reshape(-1, d)
        min_dist = torch.cdist(X2d, Xp2d).min(dim=-1).values.reshape(*X.shape[:-1])
        soft = (
            self.pending_penalty_weight
            * torch.exp(-self.pending_penalty_beta * min_dist)
            if self.pending_penalty_weight > 0.0
            else zeros
        )
        hard = hard_reference_duplicate_penalty_per_point(
            X,
            Xp,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return soft + hard

    def _candidate_penalty_per_point(self, X: Tensor) -> Tensor:
        return (
            self._pending_penalty_per_point(X)
            + self._same_batch_duplicate_penalty_per_point(X)
        )

    def _pending_penalty_aggregated(
        self,
        X: Tensor,
        reduction: Optional[ReductionType] = None,
    ) -> Tensor:
        return self._reduce_q(self._pending_penalty_per_point(X), reduction=reduction)

    def _candidate_penalty_aggregated(
        self,
        X: Tensor,
        reduction: Optional[ReductionType] = None,
    ) -> Tensor:
        return self._reduce_q(self._candidate_penalty_per_point(X), reduction=reduction)

'''
