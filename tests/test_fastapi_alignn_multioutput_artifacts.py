"""Artifact and observation coverage for multi-output ALIGNN FastAPI models."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("alignn")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from botorch.models.model_list_gp_regression import ModelListGP  # noqa: E402

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


def _fit_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "phase": "alpha",
                "temperature": 900.0,
                "pressure": 0.8,
                "furnace": "A",
                "strength": 100.0,
                "conductivity": 2.10,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "pressure": 1.0,
                "furnace": "B",
                "strength": 115.0,
                "conductivity": 2.40,
            },
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "pressure": 1.2,
                "furnace": "A",
                "strength": 125.0,
                "conductivity": 2.55,
            },
            {
                "phase": "beta",
                "temperature": 1050.0,
                "pressure": 1.4,
                "furnace": "B",
                "strength": 138.0,
                "conductivity": 2.80,
            },
        ],
        "input_cols": ["phase", "temperature", "pressure", "furnace"],
        "categorical_cols": ["furnace"],
        "target_cols": ["strength", "conductivity"],
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
    app = create_app(title="ALIGNN multi-output artifact test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tabular_store, file_store


def test_alignn_multi_output_tell_save_load_predict_roundtrip(client_and_stores) -> None:
    client, _, file_store = client_and_stores

    fit_response = client.post(
        "/api/v1/tabular/alignn/models",
        json=_fit_payload(),
    )
    assert fit_response.status_code == 200, fit_response.text
    fitted = fit_response.json()
    original_id = fitted["model_id"]
    alignn = fitted["metadata"]["alignn"]
    assert fitted["task_type"] == "multi_objective"
    assert fitted["target_names"] == ["strength", "conductivity"]
    assert alignn["multi_output"] is True
    assert alignn["num_outputs"] == 2
    assert alignn["output_names"] == ["strength", "conductivity"]
    assert alignn["output_dependency"] == "independent"

    tell_response = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/tell",
        json={
            "data": [
                {
                    "phase": "alpha",
                    "temperature": 1080.0,
                    "pressure": 1.6,
                    "furnace": "B",
                    "strength": 145.0,
                    "conductivity": 2.95,
                }
            ],
            "refit": False,
        },
    )
    assert tell_response.status_code == 200, tell_response.text
    assert tell_response.json()["n_train"] == 5
    assert tell_response.json()["metadata"]["alignn"]["num_outputs"] == 2

    prediction_payload = {
        "data": [
            {
                "phase": "beta",
                "temperature": 1020.0,
                "pressure": 1.3,
                "furnace": "B",
            }
        ],
        "include_input": True,
    }
    before_save = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/predict",
        json=prediction_payload,
    )
    assert before_save.status_code == 200, before_save.text
    records = before_save.json()["records"]
    assert len(records) == 1
    assert "strength_mean" in records[0]
    assert "conductivity_mean" in records[0]

    save_response = client.post(
        f"/api/v1/tabular/alignn/models/{original_id}/save",
        json={"filename": "alignn_multioutput_roundtrip"},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()

    artifact = deserialize_model_artifact(
        file_store.root_dir / saved["filename"],
        trust_pickle=True,
    )
    restored = artifact["optimizer"]
    assert isinstance(restored.bo.bundle.model, ModelListGP)
    assert len(restored.bo.bundle.model.models) == 2
    assert restored.dataset.target_names == ["strength", "conductivity"]
    assert restored.dataset.Y.shape == (5, 2)
    assert restored.bo.train_Y.shape == (5, 2)

    load_response = client.post(
        "/api/v1/tabular/alignn/models/load",
        json={"filename": saved["filename"], "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    loaded = load_response.json()
    loaded_id = loaded["model_id"]
    assert loaded_id != original_id
    assert loaded["task_type"] == "multi_objective"
    assert loaded["target_names"] == ["strength", "conductivity"]
    assert loaded["metadata"]["alignn"]["num_outputs"] == 2

    after_load = client.post(
        f"/api/v1/tabular/alignn/models/{loaded_id}/predict",
        json=prediction_payload,
    )
    assert after_load.status_code == 200, after_load.text
    assert after_load.json()["records"] == before_save.json()["records"]


def test_alignn_multi_output_ask_registers_all_targets_as_pending(client_and_stores) -> None:
    client, tabular_store, _ = client_and_stores
    fit_response = client.post(
        "/api/v1/tabular/alignn/models",
        json=_fit_payload(),
    )
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]

    ask_response = client.post(
        f"/api/v1/tabular/alignn/models/{model_id}/ask",
        json={
            "acquisition_config": {"name": "logei"},
            "objective_mode": "scalar",
            "objective_output": "strength",
            "objective_direction": "maximize",
            "optimize_config": {
                "q": 1,
                "num_restarts": 2,
                "raw_samples": 8,
            },
            "structure_ids": ["alpha", "beta"],
        },
    )
    assert ask_response.status_code == 200, ask_response.text

    optimizer = tabular_store.get(model_id)
    observations = optimizer.bo.observations
    assert observations is not None
    assert int(observations.pending_mask.sum().item()) == 1
    pending_y = observations.Y[observations.pending_mask]
    assert pending_y.shape == (1, 2)
    assert pending_y.isnan().all()
