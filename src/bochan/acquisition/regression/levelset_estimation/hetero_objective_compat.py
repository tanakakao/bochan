"""Objective compatibility for hetero regression level-set scores.

The high-level API automatically creates ``RegressionScalarObjective`` when a
single-output model uses ``InputPerturbation``. Hetero level-set acquisitions
produce either pointwise ``q * n_w`` scores or joint scores that already match
the t-batch shape. Both shapes need special handling so the objective neither
mistakes a candidate axis for an output axis nor aggregates ``n_w`` twice.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .hetero_single_output import (
    _ensure_q_batch,
    _HeteroRegressionLevelSetBase,
    _objective_call,
    _objective_X_for_score,
)


def _is_joint_score(score: Tensor, X: Tensor | None) -> bool:
    """Return whether ``score`` already contains one value per t-batch."""
    if X is None:
        return False
    raw_X = _ensure_q_batch(X)
    return tuple(score.shape) == tuple(raw_X.shape[:-2])


def _objective_X_for_perturbed_score(
    score: Tensor,
    X: Tensor | None,
    objective: object,
) -> Tensor | None:
    """Return raw ``X`` when the objective can aggregate ``q * n_w`` scores."""
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


def _objective_forward_call(
    objective: Any,
    score: Tensor,
    X: Tensor,
) -> Any:
    """Apply a joint-score objective without BoTorch's q-shape verification."""
    forward = getattr(objective, "forward", None)
    if callable(forward):
        try:
            return forward(score, X=X)
        except TypeError:
            return forward(score)
    return _objective_call(objective, score, None)


def _apply_objective_to_score(
    self: _HeteroRegressionLevelSetBase,
    score: Tensor,
    X: Tensor,
    name: str,
) -> Tensor:
    """Apply objectives while preserving perturbed pointwise and joint shapes."""
    objective = self.objective
    if objective is None:
        return score

    raw_X = _ensure_q_batch(X)
    if _is_joint_score(score, raw_X):
        out = _objective_forward_call(objective, score, raw_X)
    else:
        X_for_objective = _objective_X_for_perturbed_score(
            score,
            raw_X,
            objective,
        )
        out = _objective_call(objective, score, X_for_objective)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
    return out


_HeteroRegressionLevelSetBase._apply_objective_to_score = _apply_objective_to_score


__all__ = [
    "_apply_objective_to_score",
    "_is_joint_score",
    "_objective_X_for_perturbed_score",
]
