"""Persistence and observation-update coverage for CHGNet FastAPI models."""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("chgnet")
pytest.importorskip("fastapi")
pytest.importorskip("pymatgen")

from fastapi.testclient import TestClient

from bochan.composition.encoders import chgnet as chgnet_encoder_module
from bochan.model_artifact import deserialize_model_artifact
from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.dependencies import (
    get_file_optimizer_store,
    get_tabular_optimizer_store,
)
from bochan.serving.fastapi.stores import (
    FileOptimizerStore,
    InMemoryTabularOptimizerStore,
)


class FakeCrystalGraph:
    def __init__(self, structure: Any) -> None:
        self.lattice = torch.tensor(
            [
                float(structure.lattice.a) / 10.0,
                float(len(structure)) / 10.0,
                float(structure.frac_coords.sum()) / max(len(structure), 1),
            ],
            dtype=torch.float32,
        )

    def to(self, device: str = "cpu") -> FakeCrystalGraph:
        self.lattice = self.lattice.to(device)
        return self


class FakeGraphConverter:
    def __call__(self, structure: Any) -> FakeCrystalGraph:
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    atom_fea_dim = 4
    mlp_first = True

    def __init__(self) -> None:
        super().__init__()
        self.atom_embedding = nn.Linear(3, 4)
        self.atom_conv_layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.mlp = nn.Linear(4, 1)
        self.graph_converter = FakeGraphConverter()

    @classmethod
    def load(cls, **kwargs: Any) -> FakeCHGNet:
        return cls()

    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        features = torch.stack([graph.lattice for graph in graphs])
        features = torch.tanh(self.atom_embedding(features))
        for layer in self.atom_conv_layers:
            features = features + torch.tanh(layer(features))
        result = {"e": self.mlp(features).squeeze(-1)}
        if return_crystal_feas:
            result["crystal_fea"] = features
        return result


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
        "input_cols": ["phase", "temperature", "pressure", "furnace", "atmosphere"],
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
        "model_config": {
            "task_type": "regression",
            "model_type": "chgnet_gp",
            "model_kwargs": {"latent_dim": 3, "model_name": "0.3.0"},
        },
        "fit_config": {"skip_fit": True},
    }


@pytest.fixture
def client_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chgnet_encoder_module, "_upstream_chgnet_class", lambda: FakeCHGNet)
    tabular_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app = create_app(title="CHGNet artifact test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tabular_store, file_store


def test_chgnet_mixed_tell_save_load_predict_roundtrip(client_and_stores) -> None:
    client, _, file_store = client_and_stores

    fit_response = client.post("/api/v1/tabular/chgnet/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    fitted = fit_response.json()
    original_id = fitted["model_id"]
    assert fitted["metadata"]["chgnet"]["input_type"] == "mixed"
    assert fitted["metadata"]["chgnet"]["model_name"] == "0.3.0"

    tell_response = client.post(
        f"/api/v1/tabular/chgnet/models/{original_id}/tell",
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
        f"/api/v1/tabular/chgnet/models/{original_id}/predict",
        json=prediction_payload,
    )
    assert before_save.status_code == 200, before_save.text

    save_response = client.post(
        f"/api/v1/tabular/chgnet/models/{original_id}/save",
        json={"filename": "chgnet_mixed_roundtrip"},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()
    assert saved["filename"] == "chgnet_mixed_roundtrip.bochan.pt"
    assert saved["metadata"]["artifact_backend"] == "tabular"
    assert saved["metadata"]["chgnet"]["input_type"] == "mixed"

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
    assert restored.bo.train_X.shape[-2] == 5

    load_response = client.post(
        "/api/v1/tabular/chgnet/models/load",
        json={"filename": saved["filename"], "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    loaded = load_response.json()
    loaded_id = loaded["model_id"]
    assert loaded_id != original_id
    assert loaded["n_train"] == 5
    assert loaded["metadata"]["chgnet"]["structure_ids"] == ["alpha", "beta"]
    assert loaded["metadata"]["chgnet"]["categorical_process_cols"] == [
        "furnace",
        "atmosphere",
    ]

    after_load = client.post(
        f"/api/v1/tabular/chgnet/models/{loaded_id}/predict",
        json=prediction_payload,
    )
    assert after_load.status_code == 200, after_load.text
    assert after_load.json()["records"] == before_save.json()["records"]


def test_chgnet_load_requires_trusted_pickle(client_and_stores) -> None:
    client, _, _ = client_and_stores
    fit_response = client.post("/api/v1/tabular/chgnet/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]
    save_response = client.post(
        f"/api/v1/tabular/chgnet/models/{model_id}/save",
        json={"filename": "chgnet_untrusted"},
    )
    assert save_response.status_code == 200, save_response.text

    load_response = client.post(
        "/api/v1/tabular/chgnet/models/load",
        json={"filename": save_response.json()["filename"], "trust_pickle": False},
    )
    assert load_response.status_code in {400, 422}
    assert "pickle" in load_response.json()["detail"]
