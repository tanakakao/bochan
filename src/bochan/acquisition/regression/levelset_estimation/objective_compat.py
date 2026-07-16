from __future__ import annotations

"""Input-perturbation objective compatibility for regression level-set scores.

Regression level-set acquisitions compute pointwise scores before applying an
optional objective.  With ``InputPerturbation``, those scores have a final
``q * n_w`` dimension while the raw candidate tensor still has ``q`` points.
The automatically created ``RegressionScalarObjective`` needs the raw candidate
tensor to recognize and aggregate that expanded pointwise-score dimension.
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
    """Return raw ``X`` when an objective can aggregate ``q * n_w`` scores."""
    X_for_objective = _objective_X_for_score(score, X)
    if X_for_objective is not None or X is None or score.ndim == 0:
        return X_for_objective

    raw_X = _ensure_q_batch(X)
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
    """Apply score objectives without losing the perturbation-expanded q axis."""
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
