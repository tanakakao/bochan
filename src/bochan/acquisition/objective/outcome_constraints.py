"""Factories for BoTorch-compatible outcome constraints.

BoTorch outcome constraints operate on posterior samples and are satisfied when
returned values are less than or equal to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence, cast

from torch import Tensor


ConstraintOperator = Literal["ge", "gt", "le", "lt"]
OutcomeConstraint = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class OutcomeConstraintSpec:
    """Declarative specification for a scalar model-output constraint.

    Args:
        output_index: Index of the constrained model output.
        operator: Comparison operator. ``"ge"`` / ``"gt"`` mean that the
            output must be greater than the threshold. ``"le"`` / ``"lt"``
            mean that it must be smaller than the threshold.
        threshold: Constraint threshold in the model output space.

    Notes:
        BoTorch uses a smooth feasibility indicator, so strict and non-strict
        operators have the same numerical representation at the acquisition
        level. Both are retained for a natural public API.
    """

    output_index: int
    operator: ConstraintOperator
    threshold: float


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

            make_outcome_constraint(1, "ge", 0.5)

        ``y[2] <= 1.2``::

            make_outcome_constraint(2, "le", 1.2)
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
    specs: Sequence[OutcomeConstraintSpec | Mapping[str, object] | tuple[int, str, float]],
) -> list[OutcomeConstraint]:
    """Create multiple outcome constraints from declarative specifications.

    Supported specification forms are:

    - ``OutcomeConstraintSpec(output_index=1, operator="ge", threshold=0.5)``
    - ``{"output_index": 1, "operator": "ge", "threshold": 0.5}``
    - ``(1, "ge", 0.5)``
    """

    constraints: list[OutcomeConstraint] = []
    for spec in specs:
        if isinstance(spec, OutcomeConstraintSpec):
            parsed = spec
        elif isinstance(spec, Mapping):
            parsed = OutcomeConstraintSpec(
                output_index=int(spec["output_index"]),
                operator=_normalize_operator(str(spec["operator"])),
                threshold=float(spec["threshold"]),
            )
        else:
            output_index, operator, threshold = spec
            parsed = OutcomeConstraintSpec(
                output_index=int(output_index),
                operator=_normalize_operator(operator),
                threshold=float(threshold),
            )

        constraints.append(
            make_outcome_constraint(
                output_index=parsed.output_index,
                operator=parsed.operator,
                threshold=parsed.threshold,
            )
        )

    return constraints


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
    "OutcomeConstraintSpec",
    "make_interval_outcome_constraints",
    "make_outcome_constraint",
    "make_outcome_constraints",
]
