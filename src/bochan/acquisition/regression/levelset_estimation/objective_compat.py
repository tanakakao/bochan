from __future__ import annotations

"""Input-perturbation objective compatibility for regression level-set scores.

Regression level-set acquisitions use two score shapes:

- pointwise scores with a final ``q * n_w`` dimension;
- joint scores that have already reduced the candidate dimension and therefore
  match the raw candidate tensor's t-batch shape.

The automatically created ``RegressionScalarObjective`` needs the raw candidate
``X`` in both cases.  For pointwise scores it recognizes and aggregates the
expanded perturbation dimension.  For joint scores it recognizes that the score
is already aggregated and avoids applying ``n_w`` a second time.
"""

import torch
from torch import Tensor

from .single_output import (
    _RegressionLevelSetBase,
    _ensure_q_batch,
    _objective_call,
    _objective_X_for_score,
)


def _objective_X_for_perturbed_score(
    score: Tensor,
    X: Tensor | None,
    objective: object,
) -> Tensor | None:
    """Return raw ``X`` for expanded pointwise or aggregated joint scores."""
    X_for_objective = _objective_X_for_score(score, X)
    if X_for_objective is not None or X is None or score.ndim == 0:
        return X_for_objective

    raw_X = _ensure_q_batch(X)

    # Joint acquisitions such as qRegressionICU and
    # qRegressionBoundaryVariance already reduce the transformed q dimension to
    # one scalar per t-batch.  Passing raw X lets RegressionScalarObjective keep
    # this batch-shaped value unchanged instead of interpreting the t-batch size
    # as another perturbation-expanded q dimension.
    if tuple(score.shape) == tuple(raw_X.shape[:-2]):
        return raw_X

    n_w = getattr(objective, "n_w", None)
    try:
        n_w = None if n_w is None else int(n_w)
    except (TypeError, ValueError):
        n_w = None

    if (
        n_w is not None
        and n_w > 0
        and int(score.shape[-1]) == int(raw_X.shape[-2]) * n_w
    ):
        return raw_X
    return None


def _apply_objective_to_score(
    self: _RegressionLevelSetBase,
    score: Tensor,
    X: Tensor,
    name: str,
) -> Tensor:
    """Apply score objectives without losing or re-aggregating the q axis."""
    objective = self.objective
    if objective is None:
        return score

    X_for_objective = _objective_X_for_perturbed_score(score, X, objective)
    out = _objective_call(objective, score, X_for_objective)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
    return out


# Keep this compatibility behavior scoped to regression level-set acquisitions.
_RegressionLevelSetBase._apply_objective_to_score = _apply_objective_to_score


__all__ = [
    "_apply_objective_to_score",
    "_objective_X_for_perturbed_score",
]
