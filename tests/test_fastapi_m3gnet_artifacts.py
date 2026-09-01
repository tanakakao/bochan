"""Persistence and observation-update coverage for M3GNet FastAPI models."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("fastapi")
pytest.importorskip("pymatgen")

from fastapi.testclient import TestClient

from bochan.composition.encoders import m3gnet as m3gnet_encoder_module
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

_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


class FakeM3GNetGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((1, 3), dtype=torch.float32)
        self.pos = torch.empty_like(self.frac_coords)
        self.pbc_offshift = torch.empty_like(self.pbc_offset)

    def to(self, device: torch.device | str) -> FakeM3GNetGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pos = self.pos.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        return self


class FakeM3GNetConverter:
    def __init__(self, *, element_types: tuple[str, ...], cutoff: float) -> None:
        self.element_types = element_types
        self.cutoff = cutoff

    def get_graph(self, structure: Any) -> tuple[FakeM3GNetGraph, Tensor, list[float]]:
        graph = FakeM3GNetGraph(structure)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        return graph, lattice, [0.0, 0.0]


class FakeM3GNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 4
        self.is_intensive = False
        self.include_state = False
        self.element_types = ("Si",)
        self.cutoff = 5.0
        self.n_blocks = 1
        self.embedding = nn.Linear(3, 4, bias=False)
        self.graph_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.final_layer = nn.Linear(4, 1)
        self.feature_dict: dict[str, Any] = {}

    def forward(self, g: FakeM3GNetGraph, state_attr: Tensor | None = None) -> Tensor:
        del state_attr
        node_features = self.embedding(g.pos)
        node_features = node_features + torch.tanh(self.graph_layers[0](node_features))
        atomic_output = self.final_layer(node_features)
        self.feature_dict = {
            "gc_1": {"node_feat": node_features},
            "readout": atomic_output,
            "final": atomic_output.sum(),
        }
        return self.feature_dict["final"]


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
            "model_type": "m3gnet_gp",
            "model_kwargs": {"latent_dim": 3, "model_name": _MODEL_NAME},
        },
        "fit_config": {"skip_fit": True},
    }


@pytest.fixture
def client_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        m3gnet_encoder_module,
        "_matgl_api",
        lambda: (lambda model_name: FakeM3GNet(), FakeM3GNetConverter),
    )
    monkeypatch.setattr(m3gnet_encoder_module, "_unwrap_pretrained_model", lambda loaded: loaded)
    tabular_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app = create_app(title="M3GNet artifact test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tabular_store, file_store


def test_m3gnet_mixed_tell_save_load_predict_roundtrip(client_and_stores) -> None:
    client, _, file_store = client_and_stores

    fit_response = client.post("/api/v1/tabular/m3gnet/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    fitted = fit_response.json()
    original_id = fitted["model_id"]
    assert fitted["metadata"]["m3gnet"]["input_type"] == "mixed"
    assert fitted["metadata"]["m3gnet"]["representation_mode"] == "mean_node"

    tell_response = client.post(
        f"/api/v1/tabular/m3gnet/models/{original_id}/tell",
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
        f"/api/v1/tabular/m3gnet/models/{original_id}/predict",
        json=prediction_payload,
    )
    assert before_save.status_code == 200, before_save.text

    save_response = client.post(
        f"/api/v1/tabular/m3gnet/models/{original_id}/save",
        json={"filename": "m3gnet_mixed_roundtrip"},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()
    assert saved["filename"] == "m3gnet_mixed_roundtrip.bochan.pt"
    assert saved["metadata"]["artifact_backend"] == "tabular"
    assert saved["metadata"]["m3gnet"]["input_type"] == "mixed"

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
        "/api/v1/tabular/m3gnet/models/load",
        json={"filename": saved["filename"], "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    loaded = load_response.json()
    loaded_id = loaded["model_id"]
    assert loaded_id != original_id
    assert loaded["n_train"] == 5
    assert loaded["metadata"]["m3gnet"]["structure_ids"] == ["alpha", "beta"]
    assert loaded["metadata"]["m3gnet"]["categorical_process_cols"] == [
        "furnace",
        "atmosphere",
    ]

    after_load = client.post(
        f"/api/v1/tabular/m3gnet/models/{loaded_id}/predict",
        json=prediction_payload,
    )
    assert after_load.status_code == 200, after_load.text
    assert after_load.json()["records"] == before_save.json()["records"]


def test_m3gnet_load_requires_trusted_pickle(client_and_stores) -> None:
    client, _, _ = client_and_stores
    fit_response = client.post("/api/v1/tabular/m3gnet/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]
    save_response = client.post(
        f"/api/v1/tabular/m3gnet/models/{model_id}/save",
        json={"filename": "m3gnet_untrusted"},
    )
    assert save_response.status_code == 200, save_response.text

    load_response = client.post(
        "/api/v1/tabular/m3gnet/models/load",
        json={"filename": save_response.json()["filename"], "trust_pickle": False},
    )
    assert load_response.status_code in {400, 422}
    assert "pickle" in load_response.json()["detail"]
