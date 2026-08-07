from __future__ import annotations

import torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from bochan.acquisition.regression.active_learning import (
    qMultiOutputRegressionNegIntegratedPosteriorVariance,
    qRegressionNegIntegratedPosteriorVariance,
)
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.serving.webapp.workflows_tabular import (
    _set_active_learning_reference_kwargs,
)


def test_web_nipv_uses_mc_points_without_x_observed() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {}

    _set_active_learning_reference_kwargs(
        kwargs,
        acq_key="nipv",
        train_x=train_x,
    )

    assert kwargs["mc_points"] is train_x
    assert "X_observed" not in kwargs


def test_web_pointwise_active_learning_keeps_x_observed() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {}

    _set_active_learning_reference_kwargs(
        kwargs,
        acq_key="variance",
        train_x=train_x,
    )

    assert kwargs["X_observed"] is train_x
    assert "mc_points" not in kwargs


def _single_regression_hybrid() -> tuple[torch.Tensor, HybridMultiOutputModel]:
    train_x = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    train_y = (train_x - 0.35).square()
    submodel = SingleTaskGP(train_x, train_y)
    model = HybridMultiOutputModel(
        [OutputSpec(name="y", task_type="regression", model=submodel)]
    )
    return train_x, model


def _assert_nipv_optimizes(acquisition) -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=4,
        raw_samples=32,
        options={"maxiter": 50},
    )

    assert candidates.shape == (1, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(torch.as_tensor(value)).all()


def test_single_objective_hybrid_wrapper_nipv_runs_optimize_acqf() -> None:
    torch.manual_seed(0)
    train_x, model = _single_regression_hybrid()
    acquisition = qMultiOutputRegressionNegIntegratedPosteriorVariance(
        model=model,
        mc_points=train_x,
        output_weights=[1.0],
        output_reduction="weighted_mean",
    )

    _assert_nipv_optimizes(acquisition)


def test_single_output_nipv_runs_on_single_objective_hybrid_wrapper() -> None:
    torch.manual_seed(0)
    train_x, model = _single_regression_hybrid()
    acquisition = qRegressionNegIntegratedPosteriorVariance(
        model=model,
        mc_points=train_x,
    )

    assert not acquisition.uses_proxy
    _assert_nipv_optimizes(acquisition)
