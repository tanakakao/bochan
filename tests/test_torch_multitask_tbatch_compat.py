from __future__ import annotations

import torch
from botorch.models.multitask import KroneckerMultiTaskGP

from bochan.acquisition.regression.levelset_estimation.multi_output import (
    qMultiOutputRegressionStraddle,
)
from bochan.optim import optimize_acqf_torch
from bochan.optim import torch_opt


def _make_model() -> KroneckerMultiTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    base = torch.sin(2.0 * torch.pi * train_X.squeeze(-1))
    train_Y = torch.stack(
        [base, 0.7 * base + 0.2 * train_X.squeeze(-1)],
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)
    model.eval()
    model.likelihood.eval()
    return model


def test_multitask_straddle_restart_batch_backward_is_finite() -> None:
    model = _make_model()
    acquisition = qMultiOutputRegressionStraddle(
        model=model,
        thresholds=[0.0, 0.0],
        reduction="mean",
        output_reduction="mean",
    )
    X = torch.rand(2, 2, 1, dtype=torch.double, requires_grad=True)

    values = torch_opt._evaluate_acq_values(acquisition, X)

    assert values.shape == torch.Size([2])
    assert torch.isfinite(values).all()
    values.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_multitask_straddle_works_with_torch_optimizer() -> None:
    model = _make_model()
    acquisition = qMultiOutputRegressionStraddle(
        model=model,
        thresholds=[0.0, 0.0],
        reduction="mean",
        output_reduction="mean",
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, value = optimize_acqf_torch(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=4,
        options={"num_steps": 2, "lr": 0.01},
    )

    assert candidates.shape == torch.Size([2, 1])
    assert value.shape == torch.Size([1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
