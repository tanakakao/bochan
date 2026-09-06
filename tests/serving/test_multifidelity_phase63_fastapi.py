from __future__ import annotations

import torch
from botorch.models import SingleTaskGP

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig
from bochan.serving.fastapi.routers.models import tell_model
from bochan.serving.fastapi.schemas.requests import FitModelRequest, TellRequest


class _Store:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get(self, model_id: str):
        assert model_id == "mf-cost"
        return self.optimizer


def _optimizer() -> BayesianOptimizer:
    return BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_cls=SingleTaskGP,
            outcome_transform=False,
        ),
        FitConfig(skip_fit=True),
    )


def test_fit_request_accepts_initial_train_cost():
    request = FitModelRequest.model_validate(
        {
            "model_config": {"task_type": "regression"},
            "train_X": [[0.0], [1.0]],
            "train_Y": [[0.0], [1.0]],
            "train_cost": [1.0, 4.0],
        }
    )

    assert request.train_cost == [1.0, 4.0]


def test_tell_request_accepts_new_cost():
    request = TellRequest.model_validate(
        {
            "new_X": [[0.5]],
            "new_Y": [[0.6]],
            "new_cost": [2.5],
            "refit": False,
        }
    )

    assert request.new_cost == [2.5]


def test_tell_router_appends_new_cost_to_optimizer_state():
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    bo = _optimizer().fit(X, Y, train_cost=[1.0, 4.0])
    request = TellRequest.model_validate(
        {
            "new_X": [[0.5]],
            "new_Y": [[0.6]],
            "new_cost": [2.5],
            "refit": False,
            "tensor_options": {"dtype": "float64"},
        }
    )

    response = tell_model("mf-cost", request, _Store(bo))

    assert response.n_train == 3
    assert bo.train_cost is not None
    torch.testing.assert_close(
        bo.train_cost.squeeze(-1),
        torch.tensor([1.0, 4.0, 2.5], dtype=torch.double),
    )
