"""Input-perturbation objective compatibility for regression level-set scores.

Regression level-set acquisitions use two score shapes:

- pointwise scores with a final ``q * n_w`` dimension;
- joint scores that have already reduced the candidate dimension and therefore
  match the raw candidate tensor's t-batch shape.

Pointwise scores must receive raw candidate ``X`` so that the automatically
created ``RegressionScalarObjective`` can aggregate the perturbation-expanded
axis. Joint scores must also retain ``X`` while bypassing BoTorch's generic
q-batch output verification because they no longer have a q dimension.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .single_output import (
    _RegressionLevelSetBase,
    _ensure_q_batch,
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


def _objective_forward_call(
    objective: object,
    score: Tensor,
    X: Tensor,
) -> Tensor:
    """Apply an objective without MCAcquisitionObjective q-shape verification.

    A joint score has already reduced the q dimension, so BoTorch's generic
    ``MCAcquisitionObjective.__call__`` check cannot compare it with
    ``X.shape[-2]``. Calling ``forward`` directly still applies scalar direction,
    weight, and equality-target handling while avoiding both that check and a
    second ``n_w`` aggregation.
    """
    forward = getattr(objective, "forward", None)
    if callable(forward):
        try:
            return forward(score, X=X)
        except TypeError:
            return forward(score)
    return _objective_call(objective, score, None)


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

    raw_X = _ensure_q_batch(X)
    if _is_joint_score(score, raw_X):
        out = _objective_forward_call(objective, score, raw_X)
    else:
        X_for_objective = _objective_X_for_perturbed_score(score, raw_X, objective)
        out = _objective_call(objective, score, X_for_objective)

    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
    return out


# Keep this compatibility behavior scoped to regression level-set acquisitions.
_RegressionLevelSetBase._apply_objective_to_score = _apply_objective_to_score


__all__ = [
    "_apply_objective_to_score",
    "_is_joint_score",
    "_objective_X_for_perturbed_score",
]