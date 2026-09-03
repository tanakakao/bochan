from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.fastapi.app import create_app
from bochan.serving.fastapi.dependencies import (
    get_file_optimizer_store,
    get_tabular_optimizer_store,
)
from bochan.serving.fastapi.stores import (
    FileOptimizerStore,
    InMemoryTabularOptimizerStore,
)
from tests._material_residual_hardening_utils import resolve_toy_material_model

_STRUCTURE = {
    "format": "mapping",
    "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    "coords": [[0.0, 0.0, 0.0]],
    "elements": ["Si"],
}


def _fit_payload() -> dict:
    return {
        "model_config": {
            "task_type": "regression",
            "model_type": "chgnet_residual_gp",
            "model_kwargs": {},
        },
        "fit_config": {"skip_fit": True},
        "data": [
            {"structure": "s0", "temperature": 350.0, "energy": -1.0},
            {"structure": "s1", "temperature": 500.0, "energy": -0.8},
            {"structure": "s0", "temperature": 650.0, "energy": -0.7},
            {"structure": "s1", "temperature": 800.0, "energy": -0.5},
        ],
        "input_cols": ["structure", "temperature"],
        "target_cols": ["energy"],
        "bounds": {"temperature": [300.0, 900.0]},
        "structure_col": "structure",
        "structure_catalog": {"s0": _STRUCTURE, "s1": _STRUCTURE},
    }


def _candidate_payload() -> dict:
    return {
        "acquisition_config": {"name": "qlogei"},
        "optimize_config": {
            "q": 1,
            "num_restarts": 1,
            "raw_samples": 8,
            "sequential": False,
        },
        "structure_ids": ["s0", "s1"],
    }


def test_material_residual_fastapi_full_lifecycle(monkeypatch, tmp_path) -> None:
    import bochan.tabular.structure.material_residual as routing

    monkeypatch.setattr(routing, "_resolve_model_class", resolve_toy_material_model)
    app = create_app()
    memory_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: memory_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store

    with TestClient(app) as client:
        fit = client.post("/api/v1/tabular/material-residual/models", json=_fit_payload())
        assert fit.status_code == 200, fit.text
        fit_body = fit.json()
        model_id = fit_body["model_id"]
        assert fit_body["metadata"]["material_residual"]["residual_gp"] is True

        predict_payload = {
            "data": [
                {"structure": "s0", "temperature": 425.0},
                {"structure": "s1", "temperature": 725.0},
            ],
            "return_type": "dataframe",
        }
        prediction = client.post(
            f"/api/v1/tabular/material-residual/models/{model_id}/predict",
            json=predict_payload,
        )
        assert prediction.status_code == 200, prediction.text
        assert len(prediction.json()["records"]) == 2

        candidates = client.post(
            f"/api/v1/tabular/material-residual/models/{model_id}/candidates",
            json=_candidate_payload(),
        )
        assert candidates.status_code == 200, candidates.text
        assert len(candidates.json()["candidates"]) == 1
        assert candidates.json()["candidates"][0]["structure"] in {"s0", "s1"}

        asked = client.post(
            f"/api/v1/tabular/material-residual/models/{model_id}/ask",
            json=_candidate_payload(),
        )
        assert asked.status_code == 200, asked.text
        assert len(asked.json()["candidates"]) == 1

        told = client.post(
            f"/api/v1/tabular/material-residual/models/{model_id}/tell",
            json={
                "data": [{"structure": "s0", "temperature": 575.0, "energy": -0.75}],
                "refit": False,
            },
        )
        assert told.status_code == 200, told.text
        assert told.json()["n_train"] >= fit_body["n_train"] + 1

        saved = client.post(
            f"/api/v1/tabular/material-residual/models/{model_id}/save",
            json={"filename": "residual-e2e.bochan.pt"},
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["filename"] == "residual-e2e.bochan.pt"

        loaded = client.post(
            "/api/v1/tabular/material-residual/models/load",
            json={"filename": "residual-e2e.bochan.pt", "trust_pickle": True},
        )
        assert loaded.status_code == 200, loaded.text
        loaded_id = loaded.json()["model_id"]
        assert loaded_id != model_id

        restored_prediction = client.post(
            f"/api/v1/tabular/material-residual/models/{loaded_id}/predict",
            json=predict_payload,
        )
        assert restored_prediction.status_code == 200, restored_prediction.text
        assert restored_prediction.json()["records"] == prediction.json()["records"]
