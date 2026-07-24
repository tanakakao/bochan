"""Tests for portable Web-workbench model artifacts."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from bochan.serving.webapp.app import create_app
from bochan.serving.webapp.model_artifacts import (
    deserialize_web_model_artifact,
    restore_web_model_artifact,
    serialize_web_model_artifact,
)
from bochan.serving.webapp.model_reuse import register_model_signature
from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    get_visualization_session,
    register_visualization_session,
)
from bochan.tabular import TabularBayesianOptimizer


def _fake_tabular_optimizer() -> TabularBayesianOptimizer:
    optimizer = object.__new__(TabularBayesianOptimizer)
    optimizer.dataset = SimpleNamespace(
        X=None,
        Y=None,
        bounds=None,
        cat_dims=[],
        feature_names=["x"],
        target_names=["y"],
        category_maps={},
        target_category_maps={},
    )
    optimizer.bo = SimpleNamespace(bundle=SimpleNamespace(task_type="regression", model_type="base"))
    return optimizer


def _register_exportable_session() -> tuple[str, str]:
    run_id = uuid4().hex
    signature = f"signature-{run_id}"
    data = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})
    encoded_targets = pd.DataFrame({"y": [1.0, 2.0]})
    optimizer = _fake_tabular_optimizer()
    session = VisualizationSession(
        optimizer=optimizer.bo,
        tabular_optimizer=optimizer,
        data=data,
        encoded_targets=encoded_targets,
        feature_columns=["x"],
        target_columns=["y"],
        target_metadata={
            "y": {
                "target": "y",
                "task_type": "regression",
                "goal": "none",
                "internal_task": "regression",
            }
        },
        hybrid_model=False,
        rows=[
            {
                "rank": 1,
                "values": {"x": 0.5},
                "predictions": {"y": {"mean": 1.5, "std": 0.1}},
                "acq_value": 0.2,
                "constraints_ok": True,
            }
        ],
        request_details={
            "request_payload": {
                "dataset_id": "old-dataset",
                "feature_columns": ["x"],
                "target_column": "y",
                "target_columns": ["y"],
                "directions": {"y": "maximize"},
                "model_type": "base",
                "model_kwargs": {
                    "web_target_settings": [
                        {
                            "target": "y",
                            "task_type": "regression",
                            "optimize": True,
                            "direction": "maximize",
                            "goal": "none",
                            "value": None,
                        }
                    ]
                },
                "fit_maxiter": 128,
                "normalize": True,
                "outcome_transform": True,
                "input_perturbation": False,
                "n_w": 16,
                "perturbation_std": 0.1,
                "search_space": [
                    {
                        "name": "x",
                        "type": "numeric",
                        "lower": 0.0,
                        "upper": 1.0,
                        "fixed": False,
                    }
                ],
                "constraints": [],
                "k_sparse": None,
                "acquisition": {"name": "EI", "beta": 2.0, "acqf_kwargs": {}},
                "optimizer": {"name": "normal", "q": 1, "num_restarts": 10, "raw_samples": 64},
                "drop_missing": True,
            }
        },
    )
    session.result = {
        "dataset_id": "old-dataset",
        "dataset_name": "training.csv",
        "task_type": "regression",
        "model_type": "base",
        "n_train": 2,
        "n_features": 1,
        "feature_columns": ["x"],
        "target_columns": ["y"],
        "target_column": "y",
        "target_settings": [
            {
                "target": "y",
                "task_type": "regression",
                "optimize": True,
                "direction": "maximize",
                "goal": "none",
                "value": None,
            }
        ],
        "directions": {"y": "maximize"},
        "direction": "maximize",
        "best_observed": 2.0,
        "candidates": session.rows,
        "visualizations": [],
        "visualization_warnings": [],
        "visualization_run_id": run_id,
        "metadata": {},
    }
    register_visualization_session(run_id, session)
    register_model_signature(run_id, signature)
    return run_id, signature


def test_web_model_artifact_roundtrip() -> None:
    run_id, signature = _register_exportable_session()

    content, filename = serialize_web_model_artifact(run_id)
    assert content
    assert filename == "training.bochan.pt"

    with pytest.raises(ValueError, match="pickle"):
        deserialize_web_model_artifact(content, trust_pickle=False)

    payload = deserialize_web_model_artifact(content, trust_pickle=True)
    assert payload["model_signature"] == signature
    restored_run_id, result, request = restore_web_model_artifact(
        payload,
        dataset_id="new-dataset",
        dataset_name="training.csv",
    )

    assert restored_run_id != run_id
    assert result["dataset_id"] == "new-dataset"
    assert result["metadata"]["model_artifact_loaded"] is True
    assert request["dataset_id"] == "new-dataset"
    restored = get_visualization_session(restored_run_id)
    assert restored.feature_columns == ["x"]
    assert restored.target_columns == ["y"]
    assert restored.rows[0]["values"] == {"x": 0.5}


def test_model_artifact_import_requires_explicit_trust() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/model-artifacts/import",
        content=b"not-a-model",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 400
    assert "pickle" in response.json()["detail"]
