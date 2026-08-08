"""Deprecated compatibility helpers for regression level-set q-batch diversity.

The q-batch covariance extraction, soft diversity penalty, and covariance-aware
ICU / BoundaryVariance implementations now live directly in
``regression.levelset_estimation.single_output``.  This module keeps the old
private helper imports usable without mutating acquisition classes at import
time.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .single_output import (
    _RegressionLevelSetBase,
    _extract_covariance_matrix,
)


def _posterior_covariance(
    self: _RegressionLevelSetBase,
    X: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Delegate to the native regression LSE covariance implementation."""
    return self._posterior_covariance(X)


def _same_batch_penalty_per_point(
    self: _RegressionLevelSetBase,
    Xt: Tensor,
) -> Tensor:
    """Delegate to the native regression LSE same-batch penalty."""
    return self._same_batch_penalty_per_point(Xt)


def _weighted_logdet_joint_score(
    owner: _RegressionLevelSetBase,
    *,
    mean: Tensor | None = None,
    covar: Tensor,
    weight: Tensor,
    X: Tensor,
    Xt: Tensor,
    name: str,
) -> Tensor:
    """Delegate to the native covariance-aware q-batch score.

    ``mean`` is accepted for compatibility with the old private helper signature
    but is no longer required by the implementation.
    """
    del mean
    return owner._weighted_logdet_joint_score(
        covar=covar,
        weight=weight,
        X=X,
        Xt=Xt,
        name=name,
    )


__all__ = [
    "_extract_covariance_matrix",
    "_posterior_covariance",
    "_same_batch_penalty_per_point",
    "_weighted_logdet_joint_score",
]
