"""Factories for BoTorch-compatible outcome constraints.

BoTorch outcome constraints operate on posterior samples and are satisfied when
returned values are less than or equal to zero.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence, cast

from torch import Tensor


ConstraintOperator = Literal["ge", "gt", "le", "lt"]
OutcomeConstraint = Callable[[Tensor], Tensor]


def _normalize_operator(operator: str) -> ConstraintOperator:
    op = str(operator).lower()
    if op not in {"ge", "gt", "le", "lt"}:
        raise ValueError("operator must be one of 'ge', 'gt', 'le', or 'lt'.")
    return cast(ConstraintOperator, op)


def make_outcome_constraint(
    output_index: int,
    operator: ConstraintOperator,
    threshold: float,
) -> OutcomeConstraint:
    """Create one BoTorch-compatible scalar outcome constraint.

    The returned callable accepts posterior samples with shape
    ``sample_shape x batch_shape x q x m`` and returns a tensor with shape
    ``sample_shape x batch_shape x q``. A value ``<= 0`` is feasible.

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
        output_indices: Constrained model-output indices.
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
    "make_interval_outcome_constraints",
    "make_outcome_constraint",
    "make_outcome_constraints",
]
