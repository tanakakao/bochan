"""Shared public acquisition API types.

The public constructor surface follows BoTorch conventions:

- Monte Carlo sample counts are configured through ``MCSampler``.
- ``objective`` means a BoTorch ``MCAcquisitionObjective``.
- ``constraints`` use BoTorch raw posterior-sample semantics.
- ``X_pending`` is the only candidate-state argument owned by the acquisition.
- q-batch reduction is defined by the acquisition itself; public q-acquisitions
  do not expose pointwise/joint mode switches.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.sampling.base import MCSampler
from torch import Tensor

ConstraintCallable: TypeAlias = Callable[[Tensor], Tensor]
Constraints: TypeAlias = Sequence[ConstraintCallable] | None
MCObjective: TypeAlias = MCAcquisitionObjective | None
Sampler: TypeAlias = MCSampler | None
PosteriorTransformArg: TypeAlias = PosteriorTransform | None
ClassReduction = Literal["mean", "sum", "max", "min", "prod"]


def reduce_class_values(values: Tensor, reduction: ClassReduction) -> Tensor:
    """Reduce the final class axis using one canonical reduction vocabulary."""

    if reduction == "mean":
        return values.mean(dim=-1)
    if reduction == "sum":
        return values.sum(dim=-1)
    if reduction == "max":
        return values.max(dim=-1).values
    if reduction == "min":
        return values.min(dim=-1).values
    if reduction == "prod":
        return values.prod(dim=-1)
    raise ValueError(f"Unknown class reduction: {reduction!r}.")


__all__ = [
    "ClassReduction",
    "ConstraintCallable",
    "Constraints",
    "MCObjective",
    "PosteriorTransformArg",
    "Sampler",
    "reduce_class_values",
]
