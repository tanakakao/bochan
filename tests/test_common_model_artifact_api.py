"""Common ``.bochan.pt`` persistence coverage for tensor and tabular APIs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from bochan.model_artifact import (  # noqa: E402
    MODEL_ARTIFACT_FORMAT,
    MODEL_ARTIFACT_VERSION,
    deserialize_model_artifact,
)
from bochan.serving.fastapi import create_app  # noqa: E402
from bochan.serving.fastapi.dependencies import (  # noqa: E402
    get_file_optimizer_store,
    get_optimizer_store,
    get_tabular_optimizer_store,
)
from bochan.serving.fastapi.stores import (  # noqa: E402
    FileOptimizerStore,
    InMemoryOptimizerStore,
    InMemoryTabularOptimizerStore,
)


@pytest.fixture
def client_and_stores(tmp_path: Path):
    tensor_store = InMemoryOptimizerStore()
    tabular_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app = create_app(title="common artifact test")
    app.dependency_overrides[get_optimizer_store] = lambda: tensor_store
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tensor_store, tabular_store, file_store


def test_tensor_api_saves_common_model_artifact(client_and_stores) -> None:
    client, _, _, file_store = client_and_stores
    fit_response = client.post(
        "/api/v1/models",
        json={
            "model_config": {"task_type": "regression", "model_type": "base"},
            "train_X": [[0.0], [0.5], [1.0]],
            "train_Y": [[0.0], [0.25], [1.0]],
            "fit_config": {"skip_fit": True},
        },
    )
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]

    save_response = client.post(
        f"/api/v1/models/{model_id}/save",
        json={"filename": "tensor_shared", "overwrite": False},
    )
    assert save_response.status_code == 200, save_response.text
    filename = save_response.json()["filename"]
    assert filename == "tensor_shared.bochan.pt"

    artifact = deserialize_model_artifact(
        file_store.root_dir / filename,
        trust_pickle=True,
    )
    assert artifact["format"] == MODEL_ARTIFACT_FORMAT
    assert artifact["artifact_version"] == MODEL_ARTIFACT_VERSION
    assert artifact["backend"] == "tensor"

    load_response = client.post(
        "/api/v1/models/load",
        json={"filename": filename, "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    assert load_response.json()["n_train"] == 3

    wrong_backend = client.post(
        "/api/v1/tabular/models/load",
        json={"filename": filename, "trust_pickle": True},
    )
    assert wrong_backend.status_code == 400
    assert "tensor" in wrong_backend.json()["detail"]


def test_tabular_tell_save_and_load_use_common_artifact(client_and_stores) -> None:
    client, _, _, file_store = client_and_stores
    fit_response = client.post(
        "/api/v1/tabular/models",
        json={
            "data": [
                {"material": "A", "temperature": 100.0, "property": 0.1},
                {"material": "B", "temperature": 120.0, "property": 0.7},
                {"material": "A", "temperature": 140.0, "property": 0.4},
            ],
            "input_cols": ["material", "temperature"],
            "target_cols": ["property"],
            "categorical_cols": ["material"],
            "model_config": {"task_type": "regression", "model_type": "base"},
            "fit_config": {"skip_fit": True},
        },
    )
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]

    tell_response = client.post(
        f"/api/v1/tabular/models/{model_id}/tell",
        json={
            "data": [{"material": "B", "temperature": 160.0, "property": 0.9}],
            "refit": False,
        },
    )
    assert tell_response.status_code == 200, tell_response.text
    assert tell_response.json()["n_train"] == 4

    save_response = client.post(
        f"/api/v1/tabular/models/{model_id}/save",
        json={"filename": "tabular_shared", "overwrite": False},
    )
    assert save_response.status_code == 200, save_response.text
    filename = save_response.json()["filename"]
    assert filename == "tabular_shared.bochan.pt"

    artifact = deserialize_model_artifact(
        file_store.root_dir / filename,
        trust_pickle=True,
    )
    assert artifact["format"] == MODEL_ARTIFACT_FORMAT
    assert artifact["artifact_version"] == MODEL_ARTIFACT_VERSION
    assert artifact["backend"] == "tabular"
    assert artifact["optimizer"].dataset.X.shape[-2] == 4

    untrusted = client.post(
        "/api/v1/tabular/models/load",
        json={"filename": filename, "trust_pickle": False},
    )
    assert untrusted.status_code == 400
    assert "pickle" in untrusted.json()["detail"]

    load_response = client.post(
        "/api/v1/tabular/models/load",
        json={"filename": filename, "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    body = load_response.json()
    assert body["n_train"] == 4
    assert body["feature_names"] == ["material", "temperature"]
    assert body["target_names"] == ["property"]

    wrong_backend = client.post(
        "/api/v1/models/load",
        json={"filename": filename, "trust_pickle": True},
    )
    assert wrong_backend.status_code == 400
    assert "tabular" in wrong_backend.json()["detail"]
