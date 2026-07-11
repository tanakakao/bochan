from __future__ import annotations

import pytest
import torch

from bochan.optim.nsgaii_legacy_linear_constraint_compat import (
    _make_linear_constraint_compatible_optimizer,
)


class _VectorAcquisition:
    def __call__(self, X):
        return X


def _equality_as_inequalities():
    indices = torch.tensor([0, 1], dtype=torch.long)
    coefficients = torch.tensor([1.0, 1.0], dtype=torch.double)
    return [
        (indices, coefficients, 1.0),
        (indices, -coefficients, -1.0),
    ]


def test_legacy_nsgaii_repairs_and_validates_linear_constraints() -> None:
    captured: dict[str, object] = {}

    def legacy_optimizer(
        acq_function,
        bounds,
        q=None,
        objective=None,
        constraints=None,
        seed=None,
    ):
        captured["kwargs"] = {
            "q": q,
            "objective": objective,
            "constraints": constraints,
            "seed": seed,
        }
        X = torch.tensor([[0.8, 0.8], [0.9, 0.9]], dtype=torch.double)
        Y = X.clone()
        return X, Y

    def repair(X):
        repaired = X.clone()
        repaired[..., 0] = 0.4
        repaired[..., 1] = 0.6
        return repaired

    optimizer = _make_linear_constraint_compatible_optimizer(legacy_optimizer)
    X, Y = optimizer(
        acq_function=_VectorAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=2,
        inequality_constraints=_equality_as_inequalities(),
        post_processing_func=repair,
        seed=3,
    )

    torch.testing.assert_close(
        X,
        torch.tensor([[0.4, 0.6], [0.4, 0.6]], dtype=torch.double),
    )
    torch.testing.assert_close(Y, X.unsqueeze(-2).squeeze(-2))
    assert captured["kwargs"]["q"] == 2
    assert captured["kwargs"]["seed"] == 3


def test_legacy_nsgaii_raises_when_candidates_remain_infeasible() -> None:
    def legacy_optimizer(acq_function, bounds, q=None):
        X = torch.tensor([[0.8, 0.8]], dtype=torch.double)
        return X, X.clone()

    optimizer = _make_linear_constraint_compatible_optimizer(legacy_optimizer)

    with pytest.raises(RuntimeError, match="remain infeasible"):
        optimizer(
            acq_function=_VectorAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            q=1,
            inequality_constraints=_equality_as_inequalities(),
        )


def test_native_nsgaii_linear_constraint_support_is_preserved() -> None:
    captured: dict[str, object] = {}
    constraints = _equality_as_inequalities()

    def native_optimizer(
        acq_function,
        bounds,
        q=None,
        inequality_constraints=None,
    ):
        captured["constraints"] = inequality_constraints
        X = torch.tensor([[0.4, 0.6]], dtype=torch.double)
        return X, X.clone()

    optimizer = _make_linear_constraint_compatible_optimizer(native_optimizer)
    X, _ = optimizer(
        acq_function=_VectorAcquisition(),
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        q=1,
        inequality_constraints=constraints,
    )

    torch.testing.assert_close(X, torch.tensor([[0.4, 0.6]], dtype=torch.double))
    assert captured["constraints"] is constraints
