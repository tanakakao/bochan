"""Factories for BoTorch-compatible outcome constraints.

BoTorch outcome constraints operate on posterior samples and are satisfied when
returned values are less than or equal to zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

from torch import Tensor

ConstraintOperator = Literal["ge", "gt", "le", "lt"]
OutcomeConstraint = Callable[[Tensor], Tensor]
_OBJECTIVE_SPACE_ATTR = "_bochan_objective_space_constraint"


def _normalize_operator(operator: str) -> ConstraintOperator:
    op = str(operator).lower()
    if op not in {"ge", "gt", "le", "lt"}:
        raise ValueError("operator must be one of 'ge', 'gt', 'le', or 'lt'.")
    return cast(ConstraintOperator, op)


def is_objective_space_constraint(constraint: OutcomeConstraint) -> bool:
    """Return whether a constraint targets transformed objective outputs.

    Constraints created from :class:`OutcomeConstraintConfig` refer to public
    output indices. Classification acquisitions may sample latent values or class
    probabilities with an additional class axis, so those generated constraints
    must be evaluated after the acquisition objective converts samples to
    ``[..., q, m]``. Explicit advanced callables remain raw-sample constraints.
    """

    return bool(getattr(constraint, _OBJECTIVE_SPACE_ATTR, False))


def split_outcome_constraints(
    constraints: Sequence[OutcomeConstraint] | None,
) -> tuple[list[OutcomeConstraint], list[OutcomeConstraint]]:
    """Split constraints into raw-sample and transformed-objective groups."""

    raw: list[OutcomeConstraint] = []
    objective_space: list[OutcomeConstraint] = []
    for constraint in constraints or []:
        if is_objective_space_constraint(constraint):
            objective_space.append(constraint)
        else:
            raw.append(constraint)
    return raw, objective_space


def wrap_objective_space_constraints(
    constraints: Sequence[OutcomeConstraint] | None,
    *,
    objective_getter: Callable[[], Callable[..., Tensor] | None],
) -> list[OutcomeConstraint] | None:
    """Adapt generated constraints for BoTorch acquisitions.

    BoTorch calls outcome constraints on raw posterior samples. Generated bochan
    constraints instead target the public objective-output axis. This helper
    leaves explicit raw constraints untouched and lazily applies the acquisition
    objective for generated constraints. The lazy getter is important because
    some BoTorch acquisitions install ``self.objective`` during ``__init__``.
    """

    if constraints is None:
        return None

    wrapped: list[OutcomeConstraint] = []
    for constraint in constraints:
        if not is_objective_space_constraint(constraint):
            wrapped.append(constraint)
            continue

        def objective_constraint(
            samples: Tensor,
            *,
            _constraint: OutcomeConstraint = constraint,
        ) -> Tensor:
            objective = objective_getter()
            if objective is None:
                values = samples
            else:
                try:
                    values = objective(samples, X=None)
                except TypeError:
                    values = objective(samples)
            return _constraint(values)

        wrapped.append(objective_constraint)

    return wrapped


def make_outcome_constraint(
    output_index: int,
    operator: ConstraintOperator,
    threshold: float,
) -> OutcomeConstraint:
    """Create one BoTorch-compatible scalar outcome constraint.

    The returned callable accepts objective values with shape
    ``sample_shape x batch_shape x q x m`` and returns a tensor with shape
    ``sample_shape x batch_shape x q``. A value ``<= 0`` is feasible.

    The callable is marked as an objective-space constraint so classification
    acquisitions can apply probability / utility objectives before indexing the
    public output dimension.

    Examples:
        ``y[1] >= 0.5``::

            constraint = make_outcome_constraint(
                output_index=1,
                operator="ge",
                threshold=0.5,
            )
    """

    idx = int(output_index)
    if idx < 0:
        raise ValueError("output_index must be non-negative.")

    op = _normalize_operator(operator)
    threshold_f = float(threshold)

    def constraint(samples: Tensor) -> Tensor:
        if samples.ndim < 1:
            raise ValueError("samples must have at least one dimension.")
        if idx >= samples.shape[-1]:
            raise IndexError(
                f"output_index={idx} is out of bounds for "
                f"samples.shape[-1]={samples.shape[-1]}."
            )

        output = samples[..., idx]
        if op in {"ge", "gt"}:
            return threshold_f - output
        return output - threshold_f

    setattr(constraint, _OBJECTIVE_SPACE_ATTR, True)
    return constraint


def make_outcome_constraints(
    output_indices: Sequence[int],
    operators: Sequence[ConstraintOperator],
    thresholds: Sequence[float],
) -> list[OutcomeConstraint]:
    """Create multiple outcome constraints from parallel sequences.

    The values at the same position define one constraint. For example,
    ``output_indices=[1, 2]``, ``operators=["ge", "le"]``, and
    ``thresholds=[0.5, 1.2]`` create ``y[1] >= 0.5`` and ``y[2] <= 1.2``.

    Args:
        output_indices: Constrained public objective-output indices.
        operators: Comparison operators for the corresponding outputs.
        thresholds: Thresholds for the corresponding outputs.

    Returns:
        list[OutcomeConstraint]: BoTorch-compatible outcome constraints.
    """

    lengths = {
        "output_indices": len(output_indices),
        "operators": len(operators),
        "thresholds": len(thresholds),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "output_indices, operators, and thresholds must have the same length. "
            f"Got: {lengths}"
        )

    return [
        make_outcome_constraint(
            output_index=output_index,
            operator=operator,
            threshold=threshold,
        )
        for output_index, operator, threshold in zip(
            output_indices,
            operators,
            thresholds,
            strict=True,
        )
    ]


def make_interval_outcome_constraints(
    output_index: int,
    lower: float,
    upper: float,
) -> list[OutcomeConstraint]:
    """Create lower and upper outcome constraints for a closed interval."""

    lower_f = float(lower)
    upper_f = float(upper)
    if lower_f > upper_f:
        raise ValueError("lower must be less than or equal to upper.")

    return [
        make_outcome_constraint(output_index, "ge", lower_f),
        make_outcome_constraint(output_index, "le", upper_f),
    ]


__all__ = [
    "ConstraintOperator",
    "OutcomeConstraint",
    "is_objective_space_constraint",
    "make_interval_outcome_constraints",
    "make_outcome_constraint",
    "make_outcome_constraints",
    "split_outcome_constraints",
    "wrap_objective_space_constraints",
]
