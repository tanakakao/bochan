from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
)


def _optimizer() -> BayesianOptimizer:
    return BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_cls=SingleTaskGP,
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )


def _training_data():
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.0], [0.4], [1.0]], dtype=torch.double)
    cost = torch.tensor([1.0, 1.5, 3.0], dtype=torch.double)
    return X, Y, cost


def test_fit_initializes_positive_cost_observation_state():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)

    assert bo.train_cost is not None
    assert bo.train_cost.shape == (3, 1)
    torch.testing.assert_close(bo.train_cost.squeeze(-1), cost)


def test_refit_preserves_cost_state():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)
    before = bo.train_cost.clone()

    bo.refit()

    torch.testing.assert_close(bo.train_cost, before)


def test_tell_appends_cost_in_row_sync_without_refit():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)

    bo.tell(
        torch.tensor([[0.75]], dtype=torch.double),
        torch.tensor([[0.7]], dtype=torch.double),
        new_cost=torch.tensor([2.25], dtype=torch.double),
        refit=False,
    )

    assert bo.train_X.shape[0] == 4
    assert bo.train_Y.shape[0] == 4
    assert bo.train_cost.shape == (4, 1)
    torch.testing.assert_close(
        bo.train_cost[-1],
        torch.tensor([2.25], dtype=torch.double),
    )


def test_cost_tracking_requires_every_followup_cost():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)

    with pytest.raises(ValueError, match="new_cost is required"):
        bo.tell(
            torch.tensor([[0.75]], dtype=torch.double),
            torch.tensor([[0.7]], dtype=torch.double),
            refit=False,
        )


def test_cost_tracking_cannot_start_after_historical_fit():
    X, Y, _ = _training_data()
    bo = _optimizer().fit(X, Y)

    with pytest.raises(ValueError, match="historical train_cost"):
        bo.tell(
            torch.tensor([[0.75]], dtype=torch.double),
            torch.tensor([[0.7]], dtype=torch.double),
            new_cost=[2.0],
            refit=False,
        )


def test_cost_observations_validate_rows_and_positive_values():
    X, Y, cost = _training_data()
    with pytest.raises(ValueError, match="one cost value per input row"):
        _optimizer().fit(X, Y, train_cost=cost[:-1])
    with pytest.raises(ValueError, match="strictly positive"):
        _optimizer().fit(X, Y, train_cost=[1.0, 0.0, 2.0])


def test_cost_tracking_rejects_failed_or_masked_observations():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)

    with pytest.raises(ValueError, match="successful observations"):
        bo.tell(
            torch.tensor([[0.75]], dtype=torch.double),
            torch.tensor([[float("nan")]], dtype=torch.double),
            new_cost=[2.0],
            status="failed",
            refit=False,
        )


def test_deferred_learned_gp_cost_config_uses_optimizer_cost_state():
    X, Y, cost = _training_data()
    bo = _optimizer().fit(X, Y, train_cost=cost)
    config = AcquisitionConfig(
        name="mfkg",
        acqf_kwargs={
            "cost_config": {
                "kind": "learned_gp",
                "fit_model": False,
            }
        },
    )

    resolved = bo._cost_state_acquisition_config(config)
    cost_config = resolved.acqf_kwargs["cost_config"]

    assert cost_config["train_X"] is bo.train_X
    assert cost_config["train_cost"] is bo.train_cost


def test_deferred_learned_gp_requires_optimizer_cost_state():
    X, Y, _ = _training_data()
    bo = _optimizer().fit(X, Y)
    config = AcquisitionConfig(
        name="mfkg",
        acqf_kwargs={"cost_config": {"kind": "learned_gp", "fit_model": False}},
    )

    with pytest.raises(ValueError, match="train_cost"):
        bo._cost_state_acquisition_config(config)
