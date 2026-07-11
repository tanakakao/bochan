"""Objective-space outcome constraints for the NSGA-II adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor

from bochan.acquisition.objective.outcome_constraints import (
    OutcomeConstraint,
    wrap_objective_space_constraints,
)

from .nsgaii_adapter import LinearConstraint
from .nsgaii_adapter import optimize_acqf_nsgaii as _base_optimize_acqf_nsgaii


def optimize_acqf_nsgaii(
    acq_function: Any,
    bounds: Tensor,
    *,
    q: int | None = 10,
    num_objectives: int | None = None,
    ref_point: Tensor | Sequence[float] | None = None,
    objective: Any | None = None,
    constraints: Sequence[OutcomeConstraint] | None = None,
    inequality_constraints: Sequence[LinearConstraint] | None = None,
    equality_constraints: Sequence[LinearConstraint] | None = None,
    equality_tol: float = 1e-6,
    fixed_features: dict[int, float] | None = None,
    discrete_choices: dict[int, Sequence[float] | Tensor] | None = None,
    post_processing_func: Callable[[Tensor], Tensor] | None = None,
    population_size: int = 250,
    max_gen: int | None = 200,
    seed: int | None = None,
    max_attempts: int = 2,
    validate_output: bool = True,
    validate_discrete: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Run NSGA-II with generated constraints in transformed objective space.

    BoTorch applies ``objective`` to determine the Pareto objectives, but calls
    ``constraints`` on the raw multi-output acquisition values. Constraints built
    from ``OutcomeConstraintConfig`` refer to public objective indices, so adapt
    only those marked callables through the same objective. Explicit advanced
    constraints retain their raw-output semantics.
    """

    adapted_constraints = wrap_objective_space_constraints(
        constraints,
        objective_getter=lambda: objective,
    )
    return _base_optimize_acqf_nsgaii(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_objectives=num_objectives,
        ref_point=ref_point,
        objective=objective,
        constraints=adapted_constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        equality_tol=equality_tol,
        fixed_features=fixed_features,
        discrete_choices=discrete_choices,
        post_processing_func=post_processing_func,
        population_size=population_size,
        max_gen=max_gen,
        seed=seed,
        max_attempts=max_attempts,
        validate_output=validate_output,
        validate_discrete=validate_discrete,
        sequential=sequential,
        **kwargs,
    )


__all__ = ["optimize_acqf_nsgaii"]
