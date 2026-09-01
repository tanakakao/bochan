"""FastAPI coverage for structure-aware M3GNet tabular models."""

# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("fastapi")
pytest.importorskip("pymatgen")
pd = pytest.importorskip("pandas")

from bochan.composition.encoders import m3gnet as m3gnet_encoder_module
from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.schemas.m3gnet_tabular import (
    M3GNetTabularCandidateRequest,
    M3GNetTabularFitModelRequest,
)
from bochan.serving.fastapi.services import m3gnet_tabular as service

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
    """Differentiable extensive M3GNet stand-in for HTTP tests."""

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


def _install_fake_m3gnet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        m3gnet_encoder_module,
        "_matgl_api",
        lambda: (lambda model_name: FakeM3GNet(), FakeM3GNetConverter),
    )
    monkeypatch.setattr(m3gnet_encoder_module, "_unwrap_pretrained_model", lambda loaded: loaded)


def _structure(scale: float = 5.43) -> dict[str, object]:
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


def _fit_payload(model_type: str = "m3gnet_gp") -> dict[str, object]:
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
            "model_kwargs": {"latent_dim": 3, "model_name": _MODEL_NAME},
        },
        "fit_config": {"skip_fit": True},
    }


def test_m3gnet_fastapi_routes_are_registered() -> None:
    app = create_app(title="M3GNet API test")
    paths = set(app.openapi()["paths"])

    assert "/api/v1/tabular/m3gnet/models" in paths
    assert "/api/v1/tabular/m3gnet/models/{model_id}/predict" in paths
    assert "/api/v1/tabular/m3gnet/models/{model_id}/candidates" in paths
    assert "/api/v1/tabular/m3gnet/models/{model_id}/ask" in paths
    assert "/api/v1/tabular/m3gnet/models/{model_id}/tell" in paths
    assert "/api/v1/tabular/m3gnet/models/{model_id}/save" in paths
    assert "/api/v1/tabular/m3gnet/models/load" in paths


def test_m3gnet_fit_schema_rejects_structure_ids_missing_from_catalog() -> None:
    payload = _fit_payload()
    payload["data"] = [{"phase": "gamma", "temperature": 900.0, "property": 0.4}]

    with pytest.raises(ValueError, match="unknown IDs"):
        M3GNetTabularFitModelRequest.model_validate(payload)


def test_m3gnet_fit_schema_rejects_unknown_pretrained_model() -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["model_name"] = "arbitrary-remote-model"

    with pytest.raises(ValueError, match="model_name"):
        M3GNetTabularFitModelRequest.model_validate(payload)


@pytest.mark.parametrize("name", ["encoder", "adapter", "structures", "graph_converter"])
def test_m3gnet_fit_schema_rejects_object_injection(name: str) -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"][name] = {"type": "unsafe"}

    with pytest.raises(ValueError, match="server-side"):
        M3GNetTabularFitModelRequest.model_validate(payload)


def test_m3gnet_fit_schema_uses_api_safe_dkl_training_alias() -> None:
    payload = _fit_payload("m3gnet_dkl")
    payload["model_config"]["model_kwargs"]["encoder_training"] = "FULL"

    request = M3GNetTabularFitModelRequest.model_validate(payload)

    assert request.bo_model_config.model_kwargs["encoder_training"] == "full"


def test_m3gnet_fit_service_passes_structure_contract_to_tabular_optimizer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="m3gnet_gp")

        def fit(self, frame: Any) -> FakeOptimizer:
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    request = M3GNetTabularFitModelRequest.model_validate(_fit_payload())
    optimizer = service.fit_m3gnet_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    kwargs = captured["kwargs"]
    assert kwargs["structure_col"] == "phase"
    assert list(kwargs["structure_catalog"]) == ["alpha", "beta"]
    assert kwargs["structure_catalog"]["alpha"]["elements"] == ["Si", "Si"]
    assert "structure_graph_builder" not in kwargs
    assert kwargs["bounds"] == {"temperature": [850.0, 1100.0]}
    assert list(captured["frame"]["phase"]) == ["alpha", "beta", "alpha", "beta"]


def test_m3gnet_fit_endpoint_reports_representation_contract(monkeypatch) -> None:
    _install_fake_m3gnet(monkeypatch)
    from fastapi.testclient import TestClient

    client = TestClient(create_app(title="M3GNet fit endpoint"))
    response = client.post("/api/v1/tabular/m3gnet/models", json=_fit_payload())

    assert response.status_code == 200, response.text
    metadata = response.json()["metadata"]["m3gnet"]
    assert metadata["model_name"] == _MODEL_NAME
    assert metadata["encoder_initialization"] == "pretrained"
    assert metadata["representation_mode"] == "mean_node"
    assert metadata["structure_ids"] == ["alpha", "beta"]
    assert metadata["output_dependency"] == "independent"


def test_m3gnet_candidate_service_forwards_structure_subset() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def candidate(self, **kwargs: Any):
            captured.update(kwargs)
            return pd.DataFrame([{"phase": "beta", "temperature": 1015.0}]), 0.75

    request = M3GNetTabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["beta"],
        }
    )
    response = service.m3gnet_candidate_response("model-1", FakeOptimizer(), request)

    assert captured["structure_ids"] == ["beta"]
    assert captured["return_dataframe"] is True
    assert response.candidates == [{"phase": "beta", "temperature": 1015.0}]
    assert response.acq_value == pytest.approx(0.75)
