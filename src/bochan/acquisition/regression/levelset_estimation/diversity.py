from __future__ import annotations

"""Diversity utilities for regression level-set q-batch acquisitions.

The level-set acquisitions are mostly pointwise scores.  When optimized with
``q > 1``, several candidates can collapse to the same high-score point unless
an explicit q-batch diversity term is strong enough.  This module patches the
common level-set base class to make the existing diversity knobs behave more
robustly:

- ``same_batch_penalty_weight`` controls a differentiable soft repulsion.
- ``hard_duplicate_penalty`` is applied independently from the soft weight.
- ``hard_duplicate_tol`` is interpreted as a distance tolerance, not squared
  distance.
"""

import torch
from torch import Tensor

from .single_output import _RegressionLevelSetBase, _ensure_q_batch


def _same_batch_penalty_per_point(self: _RegressionLevelSetBase, Xt: Tensor) -> Tensor:
    """Return per-point q-batch diversity penalties.

    The old implementation multiplied hard duplicate penalties by
    ``same_batch_penalty_weight`` and returned early when that weight was zero.
    That made ``hard_duplicate_penalty`` ineffective unless the soft penalty was
    also enabled.  It also compared squared distance to ``hard_duplicate_tol``.

    This implementation keeps the two penalties independent and uses both an
    RBF penalty and a local inverse-distance repulsion for near-duplicates.  The
    inverse-distance term makes the penalty much steeper when q candidates are
    almost identical, which is important for ICU / straddle-style pointwise
    acquisitions.
    """
    Xt = _ensure_q_batch(Xt)
    q = int(Xt.shape[-2])
    if q <= 1:
        return Xt.new_zeros(Xt.shape[:-1])

    d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
    eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
    while eye.ndim < d2.ndim:
        eye = eye.unsqueeze(0)
    valid = ~eye

    per_point = Xt.new_zeros(Xt.shape[:-1])

    if self.same_batch_penalty_weight > 0.0:
        beta = torch.as_tensor(
            self.same_batch_penalty_beta,
            dtype=Xt.dtype,
            device=Xt.device,
        ).clamp_min(torch.as_tensor(1e-12, dtype=Xt.dtype, device=Xt.device))
        rbf = torch.exp(-beta * d2)

        # Keep the same public parameters while making the near-duplicate
        # penalty steeper.  The sqrt term is stabilized by eps; the RBF factor
        # localizes the inverse-distance penalty so far-away points are not
        # punished materially.
        eps = torch.as_tensor(
            max(float(getattr(self, "hard_duplicate_tol", 1e-8)), 1e-12),
            dtype=Xt.dtype,
            device=Xt.device,
        )
        dist = torch.sqrt(d2 + eps.pow(2))
        inv_local = rbf / dist
        soft = rbf + inv_local
        soft = torch.where(valid, soft, torch.zeros_like(soft))
        per_point = per_point + float(self.same_batch_penalty_weight) * soft.sum(dim=-1)

    if self.hard_duplicate_penalty > 0.0:
        tol = torch.as_tensor(
            max(float(self.hard_duplicate_tol), 0.0),
            dtype=Xt.dtype,
            device=Xt.device,
        )
        dup = (d2 <= tol.pow(2)).to(dtype=Xt.dtype)
        dup = torch.where(valid, dup, torch.zeros_like(dup))
        per_point = per_point + float(self.hard_duplicate_penalty) * dup.sum(dim=-1)

    return per_point


# Patch the common base class used by qRegressionICU, qRegressionStraddle,
# qRegressionBoundaryVariance, and qRegressionProbabilityOfExceedance when they
# are imported from bochan.acquisition.regression.levelset_estimation.
_RegressionLevelSetBase._same_batch_penalty_per_point = _same_batch_penalty_per_point


__all__ = ["_same_batch_penalty_per_point"]
