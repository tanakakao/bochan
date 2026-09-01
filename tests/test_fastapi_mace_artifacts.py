"""Persistence and observation-update coverage for MACE FastAPI models."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("fastapi")
pytest.importorskip("mace")

from fastapi.testclient import TestClient

from bochan.composition.encoders import mace as mace_encoder_module
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

_MODEL_NAME = "medium-mpa-0"


class FakeDescriptorLinear(nn.Linear):
    def __init__(self, width: int) -> None:
        super().__init__(width, width, bias=False)
        self.irreps_out = f"{width}x0e + {width}x1o"


class FakeProduct(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = FakeDescriptorLinear(width)
        self.scale = nn.Parameter(torch.ones(()))


class FakeMACE(nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([14], dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("num_interactions", torch.tensor(2, dtype=torch.int64))
        self.heads = ["Default"]
        self.node_embedding = nn.Linear(3, width, bias=False)
        self.radial_embedding = nn.Linear(1, width, bias=False)
        self.spherical_harmonics = nn.Identity()
        self.interactions = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(2)]
        )
        self.products = nn.ModuleList([FakeProduct(width) for _ in range(2)])
        self.readouts = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])

    def forward(
        self,
        data: dict[str, Tensor],
        *,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> dict[str, Tensor]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        positions = data["positions"]
        first = self.products[0].scale * torch.tanh(self.node_embedding(positions))
        equivariant = torch.cat([positions, positions], dim=-1)
        final = self.products[1].scale * torch.tanh(self.interactions[-1](first))
        return {
            "node_feats": torch.cat([first, equivariant, final], dim=-1),
            "energy": self.readouts[-1](final).sum(),
        }


def _fake_default_batch(self: Any, structure: dict[str, object]) -> dict[str, Tensor]:
    reference = next(self.encoder.parameters())
    lattice = torch.tensor(
        structure["lattice_mat"],
        dtype=reference.dtype,
        device=reference.device,
    )
    coords = torch.tensor(
        structure["coords"],
        dtype=reference.dtype,
        device=reference.device,
    )
    positions = coords if bool(structure.get("cartesian", False)) else coords @ lattice
    return {"positions": positions}


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
                "property": 0.40,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "pressure": 1.0,
                "furnace": "B",
                "property": 0.80,
            },
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "pressure": 1.2,
                "furnace": "A",
                "property": 0.70,
            },
            {
                "phase": "beta",
                "temperature": 1050.0,
                "pressure": 1.4,
                "furnace": "B",
                "property": 1.10,
            },
        ],
        "input_cols": ["phase", "temperature", "pressure", "furnace"],
        "categorical_cols": ["furnace"],
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
            "model_type": "mace_gp",
            "model_kwargs": {"latent_dim": 3, "model_name": _MODEL_NAME},
        },
        "fit_config": {"skip_fit": True},
    }


@pytest.fixture
def client_and_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mace_encoder_module,
        "_load_pretrained_model",
        lambda model_name: FakeMACE(),
    )
    monkeypatch.setattr(
        mace_encoder_module.MACEEncoder,
        "_default_batch",
        _fake_default_batch,
    )
    tabular_store = InMemoryTabularOptimizerStore()
    file_store = FileOptimizerStore(tmp_path)
    app = create_app(title="MACE artifact test")
    app.dependency_overrides[get_tabular_optimizer_store] = lambda: tabular_store
    app.dependency_overrides[get_file_optimizer_store] = lambda: file_store
    return TestClient(app), tabular_store, file_store


def test_mace_mixed_tell_save_load_predict_roundtrip(client_and_stores) -> None:
    client, _, file_store = client_and_stores

    fit_response = client.post("/api/v1/tabular/mace/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    fitted = fit_response.json()
    original_id = fitted["model_id"]
    assert fitted["metadata"]["mace"]["input_type"] == "mixed"
    assert fitted["metadata"]["mace"]["representation_mode"] == "invariant_l0"

    tell_response = client.post(
        f"/api/v1/tabular/mace/models/{original_id}/tell",
        json={
            "data": [
                {
                    "phase": "beta",
                    "temperature": 1080.0,
                    "pressure": 1.6,
                    "furnace": "B",
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
            }
        ],
        "include_input": True,
    }
    before_save = client.post(
        f"/api/v1/tabular/mace/models/{original_id}/predict",
        json=prediction_payload,
    )
    assert before_save.status_code == 200, before_save.text

    save_response = client.post(
        f"/api/v1/tabular/mace/models/{original_id}/save",
        json={"filename": "mace_mixed_roundtrip"},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()
    assert saved["filename"] == "mace_mixed_roundtrip.bochan.pt"
    assert saved["metadata"]["artifact_backend"] == "tabular"
    assert saved["metadata"]["mace"]["model_name"] == _MODEL_NAME

    artifact = deserialize_model_artifact(
        file_store.root_dir / saved["filename"],
        trust_pickle=True,
    )
    restored = artifact["optimizer"]
    assert tuple(restored.structure.structure_ids) == ("alpha", "beta")
    assert restored.dataset.category_maps["furnace"] == {"A": 0, "B": 1}
    assert restored.bo.train_X.shape[-2] == 5

    load_response = client.post(
        "/api/v1/tabular/mace/models/load",
        json={"filename": saved["filename"], "trust_pickle": True},
    )
    assert load_response.status_code == 200, load_response.text
    loaded = load_response.json()
    loaded_id = loaded["model_id"]
    assert loaded_id != original_id
    assert loaded["n_train"] == 5
    assert loaded["metadata"]["mace"]["structure_ids"] == ["alpha", "beta"]
    assert loaded["metadata"]["mace"]["categorical_process_cols"] == ["furnace"]

    after_load = client.post(
        f"/api/v1/tabular/mace/models/{loaded_id}/predict",
        json=prediction_payload,
    )
    assert after_load.status_code == 200, after_load.text
    assert after_load.json()["records"] == before_save.json()["records"]


def test_mace_load_requires_trusted_pickle(client_and_stores) -> None:
    client, _, _ = client_and_stores
    fit_response = client.post("/api/v1/tabular/mace/models", json=_mixed_fit_payload())
    assert fit_response.status_code == 200, fit_response.text
    model_id = fit_response.json()["model_id"]
    save_response = client.post(
        f"/api/v1/tabular/mace/models/{model_id}/save",
        json={"filename": "mace_untrusted"},
    )
    assert save_response.status_code == 200, save_response.text

    load_response = client.post(
        "/api/v1/tabular/mace/models/load",
        json={"filename": save_response.json()["filename"], "trust_pickle": False},
    )
    assert load_response.status_code in {400, 422}
    assert "pickle" in load_response.json()["detail"]
