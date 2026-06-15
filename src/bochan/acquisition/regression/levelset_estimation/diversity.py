from __future__ import annotations

"""Diversity utilities for regression level-set q-batch acquisitions.

The level-set acquisitions are mostly pointwise scores.  When optimized with
``q > 1``, several candidates can collapse to the same high-score point unless
an explicit q-batch diversity term is strong enough.  This module patches the
common level-set base class to make the existing diversity knobs behave more
robustly and patches ICU / BoundaryVariance to use joint covariance-aware scores
for q-batches.

- ``same_batch_penalty_weight`` controls a differentiable soft repulsion.
- ``hard_duplicate_penalty`` is applied independently from the soft weight.
- ``hard_duplicate_tol`` is interpreted as a distance tolerance, not squared
  distance.
- ``qRegressionICU`` and ``qRegressionBoundaryVariance`` use a weighted
  posterior-covariance log-det score when ``q > 1``.
"""

import torch
from torch import Tensor

from botorch.utils.transforms import t_batch_mode_transform

from .single_output import (
    _RegressionLevelSetBase,
    _ensure_q_batch,
    _safe_logdet,
    qRegressionBoundaryVariance,
    qRegressionICU,
)

_ORIGINAL_Q_REGRESSION_ICU_FORWARD = qRegressionICU.forward
_ORIGINAL_Q_REGRESSION_BOUNDARY_VARIANCE_FORWARD = qRegressionBoundaryVariance.forward


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


def _weighted_logdet_joint_score(
    owner: _RegressionLevelSetBase,
    *,
    mean: Tensor,
    covar: Tensor,
    weight: Tensor,
    X: Tensor,
    Xt: Tensor,
    name: str,
) -> Tensor:
    """Finalize a weighted covariance log-det score for q-batches.

    Pointwise ICU / BoundaryVariance are prone to selecting duplicate q points
    because they ignore posterior correlation between candidates.  A weighted
    log-det score approximates the joint information in the selected batch:
    duplicate points create almost duplicate covariance rows / columns and add
    little score compared with separated points.
    """
    Xt = _ensure_q_batch(Xt)
    q = int(Xt.shape[-2])
    if q <= 1:
        diag = covar.diagonal(dim1=-2, dim2=-1).clamp_min(owner.eps)
        point_score = diag * owner._align_pointwise_score_to_X(
            weight,
            Xt,
            name=f"{name} weight",
        )
        return owner._finalize_pointwise_score(point_score, X, Xt, name=name)

    weight = owner._align_pointwise_score_to_X(weight, Xt, name=f"{name} weight").clamp_min(owner.eps)
    sqrt_w = weight.sqrt()
    weighted_covar = covar * sqrt_w.unsqueeze(-1) * sqrt_w.unsqueeze(-2)

    q_eye = torch.eye(q, device=weighted_covar.device, dtype=weighted_covar.dtype)
    while q_eye.ndim < weighted_covar.ndim:
        q_eye = q_eye.unsqueeze(0)

    score = _safe_logdet(q_eye + weighted_covar, jitter=owner.eps)
    return owner._finalize_joint_score(score, X, Xt, name=name)


@t_batch_mode_transform()
def _q_regression_icu_forward(self: qRegressionICU, X: Tensor) -> Tensor:
    Xq = _ensure_q_batch(X)
    q = int(Xq.shape[-2])
    if q <= 1:
        return _ORIGINAL_Q_REGRESSION_ICU_FORWARD(self, X)

    mean, covar, Xt = self._posterior_covariance(Xq)
    var = covar.diagonal(dim1=-2, dim2=-1).clamp_min(self.eps)
    std = var.sqrt().clamp_min(self.eps)
    threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)

    if self.bandwidth is None:
        bw = std
    else:
        bw = self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)

    z = (mean - threshold) / bw
    contour_weight = torch.exp(-0.5 * z.pow(2))
    return _weighted_logdet_joint_score(
        self,
        mean=mean,
        covar=covar,
        weight=contour_weight,
        X=Xq,
        Xt=Xt,
        name="qRegressionICU",
    )


@t_batch_mode_transform()
def _q_regression_boundary_variance_forward(self: qRegressionBoundaryVariance, X: Tensor) -> Tensor:
    Xq = _ensure_q_batch(X)
    q = int(Xq.shape[-2])
    if q <= 1:
        return _ORIGINAL_Q_REGRESSION_BOUNDARY_VARIANCE_FORWARD(self, X)

    mean, covar, Xt = self._posterior_covariance(Xq)
    threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
    tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
    boundary_weight = torch.exp(-0.5 * ((mean - threshold) / tau).pow(2))
    return _weighted_logdet_joint_score(
        self,
        mean=mean,
        covar=covar,
        weight=boundary_weight,
        X=Xq,
        Xt=Xt,
        name="qRegressionBoundaryVariance",
    )


# Patch the common base class used by qRegressionICU, qRegressionStraddle,
# qRegressionBoundaryVariance, and qRegressionProbabilityOfExceedance when they
# are imported from bochan.acquisition.regression.levelset_estimation.
_RegressionLevelSetBase._same_batch_penalty_per_point = _same_batch_penalty_per_point

# Patch the two most duplicate-prone pointwise acquisitions to use joint
# covariance-aware scores when q > 1.
qRegressionICU.forward = _q_regression_icu_forward
qRegressionBoundaryVariance.forward = _q_regression_boundary_variance_forward


__all__ = [
    "_same_batch_penalty_per_point",
    "_weighted_logdet_joint_score",
]
