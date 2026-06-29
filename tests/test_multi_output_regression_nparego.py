from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.acquisition.objective import RegressionLinearMCObjective
from bochan.acquisition.regression.bayesian_optimization import (
    qMultiOutputRegressionNParEGO,
)


def _make_model() -> tuple[SingleTaskGP, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.cat(
        [
            torch.sin(train_X * torch.pi),
            (train_X - 0.5).square(),
        ],
        dim=-1,
    )
    return SingleTaskGP(train_X, train_Y), train_X


def test_regression_nparego_accepts_linear_multi_output_objective() -> None:
    model, train_X = _make_model()
    objective = RegressionLinearMCObjective(
        output_indices=[0, 1],
        weights=[1.0, 1.0],
        signs=[1.0, -1.0],
    )
    acqf = qMultiOutputRegressionNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=torch.tensor([-0.1, -0.4], dtype=torch.double),
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        objective=objective,
    )

    value = acqf(torch.tensor([[[0.2], [0.8]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert acqf.base_objective is objective
    torch.testing.assert_close(
        acqf.weights,
        torch.tensor([0.4, 0.6], dtype=torch.double),
    )


def test_regression_nparego_supports_outcome_constraints() -> None:
    model, train_X = _make_model()
    acqf = qMultiOutputRegressionNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
        constraints=[lambda samples: 0.2 - samples[..., 0]],
    )

    value = acqf(torch.tensor([[[0.25]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_regression_nparego_validates_transformed_objective_dimension() -> None:
    model, train_X = _make_model()
    objective = RegressionLinearMCObjective(
        output_indices=[0],
        weights=[1.0],
        signs=[1.0],
    )

    with pytest.raises(ValueError, match="ref_point length"):
        qMultiOutputRegressionNParEGO(
            model=model,
            X_baseline=train_X,
            ref_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
            objective=objective,
        )


def test_regression_nparego_validates_parallel_scalarization_weights() -> None:
    model, train_X = _make_model()

    with pytest.raises(ValueError, match="weights length"):
        qMultiOutputRegressionNParEGO(
            model=model,
            X_baseline=train_X,
            ref_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
            weights=torch.tensor([1.0], dtype=torch.double),
        )
