from __future__ import annotations

import pytest
import torch

from bochan.acquisition.objective import (
    make_interval_outcome_constraints,
    make_outcome_constraint,
    make_outcome_constraints,
)


def _samples() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.2, 0.7, 1.0], [0.4, 0.3, 1.4]],
            ]
        ],
        dtype=torch.double,
    )


def test_make_greater_than_outcome_constraint() -> None:
    constraint = make_outcome_constraint(1, "ge", 0.5)

    result = constraint(_samples())

    expected = torch.tensor([[[-0.2, 0.2]]], dtype=torch.double)
    torch.testing.assert_close(result, expected)


def test_make_less_than_outcome_constraint() -> None:
    constraint = make_outcome_constraint(2, "le", 1.2)

    result = constraint(_samples())

    expected = torch.tensor([[[-0.2, 0.2]]], dtype=torch.double)
    torch.testing.assert_close(result, expected)


def test_make_outcome_constraints_from_parallel_sequences() -> None:
    constraints = make_outcome_constraints(
        output_indices=[0, 1, 2],
        operators=["ge", "le", "lt"],
        thresholds=[0.1, 0.8, 1.5],
    )

    samples = _samples()
    assert len(constraints) == 3
    assert all(constraint(samples).shape == samples.shape[:-1] for constraint in constraints)

    torch.testing.assert_close(
        constraints[0](samples),
        torch.tensor([[[-0.1, -0.3]]], dtype=torch.double),
    )
    torch.testing.assert_close(
        constraints[1](samples),
        torch.tensor([[[-0.1, -0.5]]], dtype=torch.double),
    )
    torch.testing.assert_close(
        constraints[2](samples),
        torch.tensor([[[-0.5, -0.1]]], dtype=torch.double),
    )


def test_make_outcome_constraints_requires_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        make_outcome_constraints(
            output_indices=[0, 1],
            operators=["ge"],
            thresholds=[0.1, 0.2],
        )


def test_make_interval_outcome_constraints() -> None:
    lower, upper = make_interval_outcome_constraints(1, 0.4, 0.8)
    samples = _samples()

    torch.testing.assert_close(
        lower(samples),
        torch.tensor([[[-0.3, 0.1]]], dtype=torch.double),
    )
    torch.testing.assert_close(
        upper(samples),
        torch.tensor([[[-0.1, -0.5]]], dtype=torch.double),
    )


def test_invalid_operator_is_rejected() -> None:
    with pytest.raises(ValueError, match="operator"):
        make_outcome_constraint(0, "eq", 0.5)  # type: ignore[arg-type]


def test_out_of_bounds_output_index_is_reported_at_call_time() -> None:
    constraint = make_outcome_constraint(3, "ge", 0.5)

    with pytest.raises(IndexError, match="output_index=3"):
        constraint(_samples())
