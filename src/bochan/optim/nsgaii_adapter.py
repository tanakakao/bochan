"""NSGA-II adapter for high-level optimizer dispatch.

The adapter targets Bochan's current BoTorch requirement directly and adapts
scalar multi-objective acquisitions such as EHVI to the multi-output posterior
mean acquisition expected by NSGA-II.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor

from . import nsgaii as _base
from .nsgaii_diversity import select_diverse_nsgaii_candidates
from .nsgaii_outputs import adapt_nsgaii_outputs

LinearConstraint = _base.LinearConstraint
OutcomeConstraint = _base.OutcomeConstraint


def _callable_sequence(value: Any) -> list[Callable[[Tensor], Tensor]] | None:
    """Return a callable sequence, ignoring methods and framework registries."""

    if value is None:
        return None
    if callable(value) or isinstance(value, (str, bytes, dict)):
        return None
    if not isinstance(value, Sequence):
        return None
    values = list(value)
    if not all(callable(item) for item in values):
        return None
    return values


def _resolve_acquisition_constraints(
    acq_function: Any,
) -> list[OutcomeConstraint] | None:
    """Extract explicitly stored outcome constraints from an acquisition."""

    namespace = getattr(acq_function, "__dict__", {})
    for name in ("constraints", "outcome_constraints"):
        if name in namespace:
            constraints = _callable_sequence(namespace[name])
            if constraints is not None:
                return constraints

    for name in ("constraints", "outcome_constraints"):
        constraints = _callable_sequence(getattr(acq_function, name, None))
        if constraints is not None:
            return constraints
    return None


def _model_num_outputs(model: Any) -> int | None:
    """Resolve a model's output count without assuming a concrete model class."""

    for name in ("num_outputs", "_num_outputs", "num_objectives"):
        value = getattr(model, name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if value is not None:
            return int(value)
    return None


def _resolve_nsgaii_target(acq_function: Any) -> Any:
    """Return a true multi-output acquisition suitable for NSGA-II.

    EHVI and NEHVI return a scalar hypervolume-improvement value. NSGA-II should
    instead optimize the underlying model's vector posterior mean while reusing
    the EHVI objective transform, constraints, and reference point.
    """

    from botorch.acquisition.multioutput_acquisition import (
        MultiOutputAcquisitionFunction,
        MultiOutputPosteriorMean,
    )

    if isinstance(acq_function, MultiOutputAcquisitionFunction):
        return acq_function

    model = getattr(acq_function, "model", None)
    if model is None:
        return acq_function
    num_outputs = _model_num_outputs(model)
    if num_outputs is None or num_outputs < 2:
        return acq_function
    return MultiOutputPosteriorMean(model=model)


def _resolve_diversity_pool_size(
    *,
    q: int,
    population_size: int,
    diversity_pool_size: int | None,
) -> int:
    """Return a larger final pool from which the requested batch is selected."""

    if diversity_pool_size is not None:
        pool_size = int(diversity_pool_size)
        if pool_size < q:
            raise ValueError(
                "diversity_pool_size must be greater than or equal to q. "
                f"Got diversity_pool_size={pool_size}, q={q}."
            )
        return pool_size

    preferred = max(50, q * 20)
    return max(q, min(max(int(population_size), q), preferred))


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
    diversify: bool = True,
    diversity_input_weight: float = 0.7,
    diversity_pool_size: int | None = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Optimize a model's vector objective with NSGA-II.

    Args:
        acq_function: Acquisition function or model-backed multi-objective acquisition.
        bounds: Tensor bounds with shape ``2 x d``.
        q: Number of candidates to return.
        num_objectives: Number of objective dimensions.
        ref_point: Optional reference point for hypervolume validation.
        objective: Optional objective transform.
        constraints: Optional outcome constraints.
        inequality_constraints: Optional linear inequality constraints.
        equality_constraints: Optional linear equality constraints.
        equality_tol: Equality-constraint conversion tolerance.
        fixed_features: Fixed feature assignments.
        discrete_choices: Discrete choices for dimensions.
        post_processing_func: Candidate post-processing function.
        population_size: NSGA-II population size.
        max_gen: Maximum generations.
        seed: Optional random seed.
        max_attempts: Number of retry attempts.
        validate_output: Whether to validate output shape.
        validate_discrete: Whether to validate discrete choices.
        sequential: Accepted for API alignment and ignored. NSGA-II always runs
            as a non-sequential population-based optimizer.
        diversify: Whether to select the final q-batch from a larger Pareto-oriented
            pool using input-space and objective-space maximin distances.
        diversity_input_weight: Weight of normalized input-space distance in the
            diversity selector. Objective-space distance receives the remaining
            weight.
        diversity_pool_size: Optional number of NSGA-II results retained before
            selecting the final q-batch. The automatic value is at least 50 or
            ``20 * q``, capped by ``population_size`` when possible.
        **kwargs: Additional current BoTorch optimizer options.

    Returns:
        A pair ``(candidates, values)`` returned by the optimizer.
    """

    target = _resolve_nsgaii_target(acq_function)
    if target is not acq_function:
        if objective is None:
            candidate_objective = getattr(acq_function, "objective", None)
            if callable(candidate_objective):
                objective = candidate_objective
        if constraints is None:
            constraints = _resolve_acquisition_constraints(acq_function)
        if ref_point is None:
            candidate_ref_point = getattr(acq_function, "ref_point", None)
            if candidate_ref_point is not None and not callable(candidate_ref_point):
                ref_point = candidate_ref_point

    # BoTorch's Pymoo bridge converts objective values directly to NumPy. DeepGP
    # model lists may retain a leading likelihood-sample axis, so pair the target
    # and objective with a shared X context and reduce only those leading axes.
    target, objective = adapt_nsgaii_outputs(target, objective)

    backend_q = q
    if diversify and q is not None and q > 1:
        backend_q = _resolve_diversity_pool_size(
            q=q,
            population_size=population_size,
            diversity_pool_size=diversity_pool_size,
        )

    candidates, values = _base.optimize_acqf_nsgaii(
        acq_function=target,
        bounds=bounds,
        q=backend_q,
        num_objectives=num_objectives,
        ref_point=ref_point,
        objective=objective,
        constraints=constraints,
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
        sequential=False,
        **kwargs,
    )

    if diversify and q is not None and q > 1 and candidates.shape[-2] > q:
        candidates, values = select_diverse_nsgaii_candidates(
            candidates,
            values,
            q=q,
            bounds=bounds,
            input_weight=diversity_input_weight,
        )
    return candidates, values


equality_constraints_to_inequality_constraints = (
    _base.equality_constraints_to_inequality_constraints
)
validate_discrete_choices = _base.validate_discrete_choices


__all__ = [
    "equality_constraints_to_inequality_constraints",
    "optimize_acqf_nsgaii",
    "validate_discrete_choices",
]
