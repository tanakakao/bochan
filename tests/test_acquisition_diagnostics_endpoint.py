from types import SimpleNamespace

import torch

from bochan.api import ObservationData
from bochan.serving.fastapi.routers.candidates import acquisition_diagnostics


class _Store:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get(self, model_id):
        if model_id != "model-1":
            raise KeyError(model_id)
        return self.optimizer


def test_acquisition_diagnostics_returns_latest_report_and_observation_state():
    observations = ObservationData.from_status(
        X=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double),
        Y=torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.double),
        status=["success", "failed", "pending"],
    )
    optimizer = SimpleNamespace(
        observations=observations,
        last_acquisition_diagnostics={
            "training_rows": 1,
            "baseline_rows": 1,
            "pending_rows": 1,
        },
    )

    response = acquisition_diagnostics("model-1", _Store(optimizer))

    assert response.model_id == "model-1"
    assert response.diagnostics == {
        "training_rows": 1,
        "baseline_rows": 1,
        "pending_rows": 1,
    }
    assert response.observation_report is not None
    assert response.observation_report["n_rows"] == 3
    assert response.observation_report["n_success"] == 1
    assert response.observation_report["n_failed"] == 1
    assert response.observation_report["n_pending"] == 1


def test_acquisition_diagnostics_is_available_before_candidate_generation():
    optimizer = SimpleNamespace(
        observations=None,
        last_acquisition_diagnostics=None,
    )

    response = acquisition_diagnostics("model-1", _Store(optimizer))

    assert response.model_id == "model-1"
    assert response.diagnostics is None
    assert response.observation_report is None
