from __future__ import annotations

import torch
from botorch.models import SingleTaskGP

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig


def test_tell_without_refit_updates_training_view_but_keeps_model() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    bo = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_cls=SingleTaskGP,
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    bo.fit(train_X, train_Y)
    fitted_model = bo.model

    bo.tell(
        torch.tensor([[0.75]], dtype=torch.double),
        torch.tensor([[0.8]], dtype=torch.double),
        refit=False,
    )

    assert bo.model is fitted_model
    assert bo.observations is not None
    assert bo.observations.X.shape[0] == 4
    assert bo.train_X.shape[0] == 4
    assert bo.train_Y.shape[0] == 4
    torch.testing.assert_close(bo.train_X[-1], torch.tensor([0.75], dtype=torch.double))
    torch.testing.assert_close(bo.train_Y[-1], torch.tensor([0.8], dtype=torch.double))


def test_tell_failed_row_updates_state_without_adding_objective_training_row() -> None:
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    bo = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_cls=SingleTaskGP,
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )
    bo.fit(train_X, train_Y)

    bo.tell(
        torch.tensor([[0.5]], dtype=torch.double),
        torch.tensor([[float("nan")]], dtype=torch.double),
        status="failed",
        refit=False,
    )

    assert bo.observations is not None
    assert bo.observations.X.shape[0] == 3
    assert bo.observations.failed_mask.tolist() == [False, False, True]
    assert bo.train_X.shape[0] == 2
    assert bo.train_Y.shape[0] == 2
