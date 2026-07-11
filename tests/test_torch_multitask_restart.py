from __future__ import annotations

import torch
from botorch.models.multitask import KroneckerMultiTaskGP

from bochan.acquisition.regression.levelset_estimation.multi_output import (
    qMultiOutputRegressionStraddle,
)
from bochan.optim import optimize_acqf_torch


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


def test_multitask_straddle_accepts_explicit_restart_initial_conditions() -> None:
    model = _make_model()
    acquisition = qMultiOutputRegressionStraddle(
        model=model,
        thresholds=[0.0, 0.0],
        reduction="mean",
        output_reduction="mean",
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    initial_conditions = torch.tensor(
        [[[0.1], [0.3]], [[0.7], [0.9]]],
        dtype=torch.double,
    )

    candidates, value = optimize_acqf_torch(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=4,
        batch_initial_conditions=initial_conditions,
        options={"num_steps": 2, "lr": 0.01},
    )

    assert candidates.shape == torch.Size([2, 1])
    assert value.shape == torch.Size([1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()


def test_multitask_straddle_trims_sequential_pending_scores_with_perturbation_n_w() -> None:
    model = _make_model()
    acquisition = qMultiOutputRegressionStraddle(
        model=model,
        thresholds=[0.0, 0.0],
        reduction="mean",
        output_reduction="mean",
        n_w=4,
    )
    score = torch.tensor([[1.0, 2.0]], dtype=torch.double)

    aggregated = acquisition._aggregate_n_w_if_needed(
        score,
        q=1,
        context="qMultiOutputRegressionStraddle",
    )

    assert aggregated.shape == torch.Size([1, 1])
    assert torch.equal(aggregated, score[..., :1])
