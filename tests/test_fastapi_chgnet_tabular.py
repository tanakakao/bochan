"""FastAPI coverage for structure-aware CHGNet tabular models."""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("fastapi")
pytest.importorskip("pymatgen")
pd = pytest.importorskip("pandas")

from bochan.composition.encoders import chgnet as chgnet_encoder_module
from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.schemas.chgnet_tabular import (
    CHGNetTabularCandidateRequest,
    CHGNetTabularFitModelRequest,
)
from bochan.serving.fastapi.services import chgnet_tabular as service


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

    @classmethod
    def from_file(cls, path: str) -> FakeCHGNet:
        return cls()

    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        assert task == "e"
        features = torch.stack([graph.lattice for graph in graphs])
        features = torch.tanh(self.atom_embedding(features))
        for layer in self.atom_conv_layers:
            features = features + torch.tanh(layer(features))
        result = {"e": self.mlp(features).squeeze(-1)}
        if return_crystal_feas:
            result["crystal_fea"] = features
        return result


def _structure(scale: float = 5.43) -> dict[str, object]:
    return {
        "format": "mapping",
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0]],
        "elements": ["Si"],
        "cartesian": False,
    }


def _fit_payload(model_type: str = "chgnet_gp") -> dict[str, object]:
    return {
        "data": [
            {"phase": "alpha", "temperature": 900.0, "property": 0.4},
            {"phase": "beta", "temperature": 950.0, "property": 0.8},
            {"phase": "alpha", "temperature": 1000.0, "property": 0.7},
            {"phase": "beta", "temperature": 1050.0, "property": 1.1},
        ],
        "input_cols": ["temperature", "phase"],
        "target_cols": "property",
        "bounds": {"temperature": [850.0, 1100.0]},
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": _structure(5.43),
            "beta": _structure(5.50),
        },
        "model_config": {
            "task_type": "regression",
            "model_type": model_type,
            "model_kwargs": {"latent_dim": 3, "model_name": "0.3.0"},
        },
        "fit_config": {"skip_fit": True},
    }


def test_chgnet_fastapi_routes_are_registered() -> None:
    app = create_app(title="CHGNet API test")
    paths = set(app.openapi()["paths"])

    assert "/api/v1/tabular/chgnet/models" in paths
    assert "/api/v1/tabular/chgnet/models/{model_id}/predict" in paths
    assert "/api/v1/tabular/chgnet/models/{model_id}/candidates" in paths
    assert "/api/v1/tabular/chgnet/models/{model_id}/ask" in paths
    assert "/api/v1/tabular/chgnet/models/{model_id}/tell" in paths
    assert "/api/v1/tabular/chgnet/models/{model_id}/save" in paths
    assert "/api/v1/tabular/chgnet/models/load" in paths


def test_chgnet_fit_schema_rejects_structure_ids_missing_from_catalog() -> None:
    payload = _fit_payload()
    payload["data"] = [{"phase": "gamma", "temperature": 900.0, "property": 0.4}]

    with pytest.raises(ValueError, match="unknown IDs"):
        CHGNetTabularFitModelRequest.model_validate(payload)


@pytest.mark.parametrize("checkpoint", ["../model.pt", "subdir/model.pt", r"C:\\model.pt"])
def test_chgnet_fit_schema_rejects_checkpoint_paths(checkpoint: str) -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["checkpoint"] = checkpoint

    with pytest.raises(ValueError, match="filename identifier"):
        CHGNetTabularFitModelRequest.model_validate(payload)


def test_chgnet_fit_schema_rejects_unknown_pretrained_model() -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["model_name"] = "future-model"

    with pytest.raises(ValueError, match="model_name"):
        CHGNetTabularFitModelRequest.model_validate(payload)


def test_chgnet_fit_schema_rejects_encoder_injection() -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["encoder"] = {"type": "unsafe"}

    with pytest.raises(ValueError, match="server-side"):
        CHGNetTabularFitModelRequest.model_validate(payload)


def test_chgnet_fit_service_passes_structure_contract_to_tabular_optimizer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="chgnet_gp")

        def fit(self, frame: Any) -> FakeOptimizer:
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    request = CHGNetTabularFitModelRequest.model_validate(_fit_payload())
    optimizer = service.fit_chgnet_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    kwargs = captured["kwargs"]
    assert kwargs["structure_col"] == "phase"
    assert list(kwargs["structure_catalog"]) == ["alpha", "beta"]
    assert kwargs["structure_catalog"]["alpha"]["elements"] == ["Si"]
    assert "structure_graph_builder" not in kwargs
    assert kwargs["bounds"] == {"temperature": [850.0, 1100.0]}
    assert list(captured["frame"]["phase"]) == ["alpha", "beta", "alpha", "beta"]


def test_chgnet_fit_service_resolves_checkpoint_under_allowlisted_root(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    checkpoint = tmp_path / "chgnet.pt"
    checkpoint.write_bytes(b"placeholder")

    class FakeOptimizer:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="chgnet_gp")

        def fit(self, frame: Any) -> FakeOptimizer:
            return self

    monkeypatch.setenv("BOCHAN_CHGNET_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["checkpoint"] = "chgnet.pt"
    request = CHGNetTabularFitModelRequest.model_validate(payload)
    service.fit_chgnet_tabular_optimizer(request)

    model_kwargs = captured["kwargs"]["model_config"]["model_kwargs"]
    assert model_kwargs["checkpoint"] == str(checkpoint.resolve())


def test_chgnet_fit_endpoint_uses_pretrained_contract(monkeypatch) -> None:
    monkeypatch.setattr(chgnet_encoder_module, "_upstream_chgnet_class", lambda: FakeCHGNet)
    from fastapi.testclient import TestClient

    client = TestClient(create_app(title="CHGNet fit endpoint"))
    response = client.post("/api/v1/tabular/chgnet/models", json=_fit_payload())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["metadata"]["chgnet"]["model_name"] == "0.3.0"
    assert payload["metadata"]["chgnet"]["encoder_initialization"] == "pretrained"
    assert payload["metadata"]["chgnet"]["structure_ids"] == ["alpha", "beta"]
    assert payload["metadata"]["chgnet"]["output_dependency"] == "independent"


def test_chgnet_candidate_service_forwards_structure_subset() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def candidate(self, **kwargs: Any):
            captured.update(kwargs)
            return pd.DataFrame([{"phase": "beta", "temperature": 1015.0}]), 0.75

    request = CHGNetTabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["beta"],
        }
    )
    response = service.chgnet_candidate_response("model-1", FakeOptimizer(), request)

    assert captured["structure_ids"] == ["beta"]
    assert captured["return_dataframe"] is True
    assert response.candidates == [{"phase": "beta", "temperature": 1015.0}]
    assert response.acq_value == pytest.approx(0.75)
