from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from bochan.api import MaterialAPIModelSpec, OptimizeConfig
from bochan.serving.fastapi.app import create_app
from bochan.serving.fastapi.routers import material_model_axes
from bochan.serving.fastapi.services.material_models import apply_material_target_task


class _FakeOptimizer:
    def __init__(self, *, model_config, fit_config, bounds=None, data_context=None):
        self.model_config = model_config
        self.fit_config = fit_config
        self.bounds = bounds
        self.data_context = data_context
        self.bundle = None

    def fit(self, train_X, train_Y, train_Yvar=None):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.bundle = SimpleNamespace(
            task_type="regression",
            model_type=self.model_config.model_type,
            metadata={},
        )
        return self

    def predict(self, X, *, return_type, return_result, posterior_kwargs):
        mean = X.sum(dim=-1, keepdim=True)
        variance = torch.ones_like(mean) * 0.25
        return SimpleNamespace(
            task_type="regression",
            prediction_space="outcome",
            variance_kind="latent",
            posterior=None,
            mean=mean,
            variance=variance,
        )


def test_material_fit_registers_model_for_generic_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(material_model_axes, "BayesianOptimizer", _FakeOptimizer)
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/materials/models/fit",
        json={
            "model": {"family": "crabnet"},
            "train_X": [[0.1, 0.2], [0.3, 0.4]],
            "train_Y": [[1.0], [2.0]],
            "bounds": [[0.0, 0.0], [1.0, 1.0]],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "material:crabnet:wide_output"
    assert payload["metadata"]["material_model_axes"]["family"] == "crabnet"

    prediction = client.post(
        f"/api/v1/models/{payload['model_id']}/predict",
        json={"X": [[0.2, 0.5]], "return_type": "mean_variance"},
    )
    assert prediction.status_code == 200
    assert prediction.json()["mean"] == [[0.7]]


def test_material_fit_rejects_reserved_fidelity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(material_model_axes, "BayesianOptimizer", _FakeOptimizer)
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/materials/models/fit",
        json={
            "model": {"family": "chgnet", "fidelity_mode": "continuous"},
            "train_X": [[0.1], [0.2]],
            "train_Y": [[1.0], [2.0]],
        },
    )

    assert response.status_code == 422
    assert "fidelity remains reserved but unimplemented" in response.json()["detail"]


def test_explicit_task_is_injected_into_candidate_fixed_features() -> None:
    optimizer = SimpleNamespace(
        material_api_model_spec=MaterialAPIModelSpec(
            family="crabnet",
            task_mode="explicit",
            task_feature=-1,
            all_tasks=(0, 2),
        ),
        material_input_dim=4,
    )

    updated = apply_material_target_task(
        optimizer,
        OptimizeConfig(fixed_features={0: 0.5}),
        2,
    )

    assert updated.fixed_features == {0: 0.5, 3: 2.0}


def test_explicit_task_requires_target_task() -> None:
    optimizer = SimpleNamespace(
        material_api_model_spec=MaterialAPIModelSpec(
            family="crabnet",
            task_mode="explicit",
            task_feature=-1,
            all_tasks=(0, 1),
        ),
        material_input_dim=3,
    )

    with pytest.raises(ValueError, match="target_task is required"):
        apply_material_target_task(optimizer, OptimizeConfig(), None)


def test_explicit_task_rejects_conflicting_manual_fixed_feature() -> None:
    optimizer = SimpleNamespace(
        material_api_model_spec=MaterialAPIModelSpec(
            family="crabnet",
            task_mode="explicit",
            task_feature=-1,
            all_tasks=(0, 1),
        ),
        material_input_dim=3,
    )

    with pytest.raises(ValueError, match="conflicts with required material task value"):
        apply_material_target_task(
            optimizer,
            OptimizeConfig(fixed_features={2: 0.0}),
            1,
        )
