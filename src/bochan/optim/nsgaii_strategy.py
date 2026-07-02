"""Acquisition-side strategy object for NSGA-II candidate generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor


class NSGAIIStrategy:
    """Describe posterior-mean Pareto search for the NSGA-II backend.

    NSGA-II is not a scalar acquisition function. This lightweight object keeps
    the model and optional objective-space metadata together until the optimizer
    adapter converts it to ``MultiOutputPosteriorMean``.
    """

    def __init__(
        self,
        *,
        model: Any,
        objective: Any | None = None,
        constraints: Sequence[Any] | None = None,
        ref_point: Tensor | Sequence[float] | None = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.outcome_constraints = (
            None if constraints is None else list(constraints)
        )
        self.ref_point = ref_point


def build_nsgaii_strategy(
    *,
    bundle: Any,
    config: Any,
    data_context: Any,
) -> NSGAIIStrategy:
    """Build an NSGA-II strategy from the high-level acquisition config."""

    from bochan.api.factory import build_objective

    objective = build_objective(
        bundle=bundle,
        config=config,
        data_context=data_context,
    )
    constraints = getattr(data_context, "constraints", None)
    if constraints is None:
        constraints = config.acqf_kwargs.get("constraints")
    ref_point = getattr(data_context, "ref_point", None)
    if ref_point is None:
        ref_point = config.acqf_kwargs.get("ref_point")

    return NSGAIIStrategy(
        model=bundle.model,
        objective=objective,
        constraints=constraints,
        ref_point=ref_point,
    )


__all__ = ["NSGAIIStrategy", "build_nsgaii_strategy"]
