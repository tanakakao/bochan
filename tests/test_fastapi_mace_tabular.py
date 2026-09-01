"""FastAPI coverage for structure-aware MACE tabular models."""

# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("fastapi")
pytest.importorskip("mace")
pd = pytest.importorskip("pandas")

from bochan.composition.encoders import mace as mace_encoder_module
from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.schemas.mace_tabular import (
    MACETabularCandidateRequest,
    MACETabularFitModelRequest,
)
from bochan.serving.fastapi.services import mace_tabular as service

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
    """Differentiable MACE stand-in for HTTP tests."""

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
        node_feats = torch.cat([first, equivariant, final], dim=-1)
        return {
            "node_feats": node_feats,
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


def _install_fake_mace(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _fit_payload(model_type: str = "mace_gp") -> dict[str, object]:
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


def _multitask_payload(model_type: str = "mace_multitask") -> dict[str, object]:
    payload = _fit_payload(model_type)
    payload["data"] = [
        {
            "phase": row["phase"],
            "temperature": row["temperature"],
            "strength": 100.0 + index * 10.0,
            "conductivity": 2.0 + index * 0.2,
        }
        for index, row in enumerate(payload["data"])
    ]
    payload["target_cols"] = ["strength", "conductivity"]
    return payload


def test_mace_fastapi_routes_are_registered() -> None:
    app = create_app(title="MACE API test")
    paths = set(app.openapi()["paths"])

    assert "/api/v1/tabular/mace/models" in paths
    assert "/api/v1/tabular/mace/models/{model_id}/predict" in paths
    assert "/api/v1/tabular/mace/models/{model_id}/candidates" in paths
    assert "/api/v1/tabular/mace/models/{model_id}/ask" in paths
    assert "/api/v1/tabular/mace/models/{model_id}/tell" in paths
    assert "/api/v1/tabular/mace/models/{model_id}/save" in paths
    assert "/api/v1/tabular/mace/models/load" in paths


def test_mace_fit_schema_rejects_structure_ids_missing_from_catalog() -> None:
    payload = _fit_payload()
    payload["data"] = [
        {"phase": "gamma", "temperature": 900.0, "property": 0.4}
    ]

    with pytest.raises(ValueError, match="unknown IDs"):
        MACETabularFitModelRequest.model_validate(payload)


def test_mace_fit_schema_rejects_unknown_pretrained_model() -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"]["model_name"] = "arbitrary-remote-model"

    with pytest.raises(ValueError, match="model_name"):
        MACETabularFitModelRequest.model_validate(payload)


@pytest.mark.parametrize("name", ["encoder", "adapter", "structures", "batch_builder"])
def test_mace_fit_schema_rejects_object_injection(name: str) -> None:
    payload = _fit_payload()
    payload["model_config"]["model_kwargs"][name] = {"type": "unsafe"}

    with pytest.raises(ValueError, match="server-side"):
        MACETabularFitModelRequest.model_validate(payload)


def test_mace_fit_schema_normalizes_api_safe_representation_options() -> None:
    payload = _fit_payload("mace_dkl")
    payload["model_config"]["model_kwargs"].update(
        {
            "encoder_training": "FULL",
            "pooling": "SUM",
            "num_layers": 1,
            "head": "Default",
        }
    )

    request = MACETabularFitModelRequest.model_validate(payload)
    kwargs = request.bo_model_config.model_kwargs

    assert kwargs["encoder_training"] == "full"
    assert kwargs["pooling"] == "sum"
    assert kwargs["num_layers"] == 1
    assert kwargs["head"] == "Default"


def test_mace_fit_service_passes_structure_contract_to_tabular_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="mace_gp")

        def fit(self, frame: Any) -> FakeOptimizer:
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    request = MACETabularFitModelRequest.model_validate(_fit_payload())
    optimizer = service.fit_mace_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    kwargs = captured["kwargs"]
    assert kwargs["structure_col"] == "phase"
    assert list(kwargs["structure_catalog"]) == ["alpha", "beta"]
    assert kwargs["structure_catalog"]["alpha"]["elements"] == ["Si", "Si"]
    assert "structure_graph_builder" not in kwargs
    assert kwargs["bounds"] == {"temperature": [850.0, 1100.0]}
    assert list(captured["frame"]["phase"]) == ["alpha", "beta", "alpha", "beta"]


def test_mace_fit_endpoint_reports_representation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)
    from fastapi.testclient import TestClient

    client = TestClient(create_app(title="MACE fit endpoint"))
    response = client.post("/api/v1/tabular/mace/models", json=_fit_payload())

    assert response.status_code == 200, response.text
    metadata = response.json()["metadata"]["mace"]
    assert metadata["model_name"] == _MODEL_NAME
    assert metadata["encoder_initialization"] == "pretrained"
    assert metadata["representation_mode"] == "invariant_l0"
    assert metadata["num_layers"] == 2
    assert metadata["num_interactions"] == 2
    assert metadata["pooling"] == "mean"
    assert metadata["head"] == "Default"
    assert metadata["available_heads"] == ["Default"]
    assert metadata["cutoff"] == pytest.approx(5.0)
    assert metadata["structure_ids"] == ["alpha", "beta"]
    assert metadata["output_dependency"] == "independent"


def test_mace_multitask_fit_endpoint_reports_correlated_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mace(monkeypatch)
    from fastapi.testclient import TestClient

    client = TestClient(create_app(title="MACE multitask endpoint"))
    response = client.post(
        "/api/v1/tabular/mace/models",
        json=_multitask_payload(),
    )

    assert response.status_code == 200, response.text
    metadata = response.json()["metadata"]["mace"]
    assert metadata["multi_output"] is True
    assert metadata["num_outputs"] == 2
    assert metadata["output_dependency"] == "correlated"
    assert metadata["shared_encoder"] is True
    assert metadata["task_kernel"] == "MultitaskKernel"


def test_mace_candidate_service_forwards_structure_subset() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def candidate(self, **kwargs: Any):
            captured.update(kwargs)
            return pd.DataFrame([{"phase": "beta", "temperature": 1015.0}]), 0.75

    request = MACETabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["beta"],
        }
    )
    response = service.mace_candidate_response("model-1", FakeOptimizer(), request)

    assert captured["structure_ids"] == ["beta"]
    assert captured["return_dataframe"] is True
    assert response.candidates == [{"phase": "beta", "temperature": 1015.0}]
    assert response.acq_value == pytest.approx(0.75)
