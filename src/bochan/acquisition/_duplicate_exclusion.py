"""Scale-independent hard duplicate exclusion for acquisition functions.

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
            f"Reference feature dimension mismatch: X.shape={tuple(X.shape)}, X_ref.shape={tuple(X_ref.shape)}."
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
