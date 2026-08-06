"""Common types and helpers for regression active-learning acquisitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor

try:
    from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
except Exception:  # pragma: no cover - depends on BoTorch version
    MCMultiOutputObjective = None  # type: ignore


ReductionType = Literal["mean", "sum", "max", "min"]
OutputReductionType = Literal["mean", "sum", "max", "min"]


# ============================================================
# Generic helpers
# ============================================================


def _reduce(t: Tensor, dim: int, mode: str) -> Tensor:
    if mode == "mean":
        return t.mean(dim=dim)
    if mode == "sum":
        return t.sum(dim=dim)
    if mode == "max":
        return t.max(dim=dim).values
    if mode == "min":
        return t.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode!r}.")


def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _safe_prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _objective_call(objective: Callable, score: Tensor, X: Tensor | None):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _is_mc_multi_output_objective(objective: Any) -> bool:
    return MCMultiOutputObjective is not None and isinstance(objective, MCMultiOutputObjective)


def _looks_like_score_objective(objective: Any) -> bool:
    """Detect score objectives used in classification / ordinal implementations."""
    if objective is None:
        return False
    if _is_mc_multi_output_objective(objective):
        return False
    return (
        hasattr(objective, "n_w")
        or hasattr(objective, "risk_type")
        or hasattr(objective, "alpha")
        or objective.__class__.__name__.endswith("ScoreObjective")
    )


# ============================================================
