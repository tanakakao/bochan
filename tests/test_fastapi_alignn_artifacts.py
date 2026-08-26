"""Persistence and observation-update coverage for ALIGNN FastAPI models."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("alignn")
pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

from fastapi.testclient import TestClient  # noqa: E402

from bochan.model_artifact import deserialize_model_artifact  # noqa: E402
from bochan.serving.fastapi import create_app  # noqa: E402
from bochan.serving.fastapi.dependencies import (  # noqa: E402
    get_file_optimizer_store,
    get_tabular_optimizer_store,
)
from bochan.serving.fastapi.stores import (  # noqa: E402
    FileOptimizerStore,
    InMemoryTabularOptimizerStore,
)


def _structure(scale: float) -> dict[str, object]:
    return {
        "format": "mapping",
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def _small_encoder_config() -> dict[str, object]:
    return {
        "name": "alignn_atomwise_pure",
        "alignn_layers": 1,
        "gcn_layers": 1,
        "atom_input_features": 92,
        "edge_input_features": 16,
        "triplet_input_features": 8,
        "embedding_features": 8,
        "hidden_features": 16,
        "output_features": 1,
        "calculate_gradient": False,
        "gradwise_weight": 0.0,
        "energy_mult_natoms": False,
        "use_penalty": False,
    }


def _mixed_fit_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "phase": "alpha",
                "temperature": 900.0,
                "pressure": 0.8,
                "furnace": "A",
                "atmosphere": "air",
                "property": 0.40,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "pressure": 1.0,
                "furnace": "B",
                "atmosphere": "N2",
                "property": 0.80,
            },
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "pressure": 1.2,
                "furnace": "A",
                "atmosphere": "Ar",
                "property": 0.70,
            },
            {
                "phase": "beta",
                "temperature": 1050.0,
                "pressure": 1.4,
                "furnace": "B",
                "atmosphere": "N2",
                "property": 1.10,
            },
        ],
        "input_cols": [
            "phase",
            "temperature",
            "pressure",
            "furnace",
            "atmosphere",
        ],
        "categorical_cols": ["furnace", "atmosphere"],
        "target_cols": "property",
        "bounds": {
            "temperature": [850.0, 1150.0],
            "pressure": [0.5, 2.0],
        },
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": _structure(5.43),
            "beta": _structure(5.55),
        },
        "structure_graph_config": {
            "neighbor_strategy": "pure_torch",
            "cutoff": 5.0,
            "max_neighbors": 8,
            "three_body_cutoff": 3.5,
        },
        "model_config": {
            "task_type": "regression",
            "model_type": "alignn_gp",
            "model_kwargs": {
                "encoder_config": _small_encoder_config(),
                "latent_dim": 4,
            },
        },
        "fit_config": {"skip_fit": True},
    }


@pytest.fixture
def client_and_stores(tmp_path: Path):
    tabular_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app = create_app(title="ALIGNN artifact test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tabular_store, file_store


def test_alignn_artifact_routes_are_registered() -> None:
    app = create_app(title="ALIGNN artifact route test")
    paths = set(app.openapi()["paths"])

    assert "/api/v1/tabular/alignn/models/{model_id}/tell" in paths
    assert "/api/v1/tabular/alignn/models/{model_id}/save" in paths
    assert "/api/v1/tabular/alignn/models/load" in paths


def test_alignn_mixed_tell_save_load_predict_roundtrip(client_and_stores) -> None:
    client, _, file_store = client_and_stores

    fit_response = client.post(
        "/api/v1/tabular/alignn/models",
        json=_mixed_fit_payload(),
    )
    assert fit_response.status_code == 200, fit_response.text
    fitted = fit_response.json()
    original_id = fitted["model_id"]
    assert fitted["metadata"]["alignn"]["input_type"] == "mixed"

    tell_response = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/tell",
        json={
            "data": [
                {
                    "phase": "beta",
                    "temperature": 1080.0,
                    "pressure": 1.6,
                    "furnace": "B",
                    "atmosphere": "Ar",
                    "property": 1.25,
                }
            ],
            "refit": False,
        },
    )
    assert tell_response.status_code == 200, tell_response.text
    assert tell_response.json()["n_train"] == 5
    assert tell_response.json()["metadata"]["alignn"]["category_maps"] == {
        "furnace": {"A": 0, "B": 1},
        "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
    }

    prediction_payload = {
        "data": [
            {
                "phase": "beta",
                "temperature": 1010.0,
                "pressure": 1.3,
                "furnace": "B",
                "atmosphere": "Ar",
            }
        ],
        "include_input": True,
    }
    before_save = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/predict",
        json=prediction_payload,
    )
    assert before_save.status_code == 200, before_save.text

    save_response = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/save",
        json={"filename": "alignn_mixed_roundtrip"},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()
    assert saved["filename"] == "alignn_mixed_roundtrip.bochan.pt"
    assert saved["metadata"]["artifact_backend"] == "tabular"
    assert saved["metadata"]["alignn"]["input_type"] == "mixed"

    artifact = deserialize_model_artifact(
        file_store.root_dir / saved["filename"],
        trust_pickle=True,
    )
    restored = artifact["optimizer"]
    assert tuple(restored.structure.structure_ids) == ("alpha", "beta")
    assert restored.dataset.category_maps["furnace"] == {"A": 0, "B": 1}
    assert restored.dataset.category_maps["atmosphere"] == {
        "air": 0,
        "N2": 1,
        "Ar": 2,
    }
    assert restored.dataset.X.shape[-2] == 5
    assert restored.bo.train_X.shape[-2] == 5

    load_response = client.post(
        "/api/v1/tabular/alignn/models/load",
        json={"filename": saved["filename"], "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    loaded = load_response.json()
    loaded_id = loaded["model_id"]
    assert loaded_id != original_id
    assert loaded["n_train"] == 5
    assert loaded["metadata"]["alignn"]["input_type"] == "mixed"
    assert loaded["metadata"]["alignn"]["structure_ids"] == ["alpha", "beta"]
    assert loaded["metadata"]["alignn"]["categorical_process_cols"] == [
        "furnace",
        "atmosphere",
    ]

    after_load = client.post(
        f"/api/v1/tabular/alignn/models/{loaded_id}/predict",
        json=prediction_payload,
    )
    assert after_load.status_code == 200, after_load.text
    assert after_load.json()["records"] == before_save.json()["records"]


def test_alignn_load_requires_trusted_pickle(client_and_stores) -> None:
    client, _, _ = client_and_stores
    fit_response = client.post(
        "/api/v1/tabular/alignn/models",
        json=_mixed_fit_payload(),
    )
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]
    save_response = client.post(
        f"/api/v1/tabular/alignn/models/{model_id}/save",
        json={"filename": "alignn_untrusted"},
    )
    assert save_response.status_code == 200, save_response.text

    load_response = client.post(
        "/api/v1/tabular/alignn/models/load",
        json={
            "filename": save_response.json()["filename"],
            "trust_pickle": False,
        },
    )
    assert load_response.status_code in {400, 422}
    assert "pickle" in load_response.json()["detail"]
