"""FastAPI coverage for structure-aware ALIGNN tabular models."""

# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

from bochan.serving.fastapi import create_app
from bochan.serving.fastapi.schemas.alignn_tabular import (
    ALIGNNTabularCandidateRequest,
    ALIGNNTabularFitModelRequest,
)
from bochan.serving.fastapi.schemas.tabular import TabularPredictRequest
from bochan.serving.fastapi.services import alignn_tabular as service


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


def _fit_payload() -> dict[str, object]:
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
            "model_type": "alignn_gp",
            "model_kwargs": {"latent_dim": 8},
        },
        "fit_config": {"skip_fit": True},
    }


def test_alignn_fastapi_routes_are_registered() -> None:
    app = create_app(title="ALIGNN API test")
    paths = set(app.openapi()["paths"])

    assert "/api/v1/tabular/alignn/models" in paths
    assert "/api/v1/tabular/alignn/models/{model_id}/predict" in paths
    assert "/api/v1/tabular/alignn/models/{model_id}/candidates" in paths
    assert "/api/v1/tabular/alignn/models/{model_id}/ask" in paths


def test_alignn_fit_schema_rejects_structure_ids_missing_from_catalog() -> None:
    payload = _fit_payload()
    payload["data"] = [
        {"phase": "gamma", "temperature": 900.0, "property": 0.4},
    ]

    with pytest.raises(ValueError, match="unknown IDs"):
        ALIGNNTabularFitModelRequest.model_validate(payload)


def test_alignn_fit_service_passes_structure_contract_to_tabular_optimizer(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    fake_builder = SimpleNamespace(config={"neighbor_strategy": "k-nearest"})

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="alignn_gp")

        def fit(self, frame):
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    monkeypatch.setattr(service, "graph_builder_from_request", lambda request: fake_builder)

    request = ALIGNNTabularFitModelRequest.model_validate(_fit_payload())
    optimizer = service.fit_alignn_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    kwargs = captured["kwargs"]
    assert kwargs["structure_col"] == "phase"
    assert list(kwargs["structure_catalog"]) == ["alpha", "beta"]
    assert kwargs["structure_catalog"]["alpha"]["elements"] == ["Si"]
    assert kwargs["structure_graph_builder"] is fake_builder
    assert kwargs["bounds"] == {"temperature": [850.0, 1100.0]}
    assert list(captured["frame"]["phase"]) == ["alpha", "beta", "alpha", "beta"]


def test_alignn_fit_service_normalizes_numeric_structure_ids(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_builder = SimpleNamespace(config={"neighbor_strategy": "k-nearest"})

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="alignn_gp")

        def fit(self, frame):
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    monkeypatch.setattr(service, "graph_builder_from_request", lambda request: fake_builder)

    payload = _fit_payload()
    payload["data"] = [
        {"phase": 1, "temperature": 900.0, "property": 0.4},
        {"phase": 2, "temperature": 950.0, "property": 0.8},
    ]
    payload["structure_catalog"] = {"1": _structure(5.43), "2": _structure(5.50)}
    request = ALIGNNTabularFitModelRequest.model_validate(payload)
    service.fit_alignn_tabular_optimizer(request)

    assert list(captured["frame"]["phase"]) == ["1", "2"]


def test_alignn_predict_service_normalizes_structure_ids() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        structure = SimpleNamespace(column="phase")

        def predict(self, frame, **kwargs):
            captured["frame"] = frame.copy()
            captured["kwargs"] = kwargs
            return pd.DataFrame(
                {
                    "phase": frame["phase"],
                    "temperature": frame["temperature"],
                    "property_mean": [0.5] * len(frame),
                }
            )

    request = TabularPredictRequest.model_validate(
        {
            "data": [{"phase": 1, "temperature": 975.0}],
            "include_input": True,
        }
    )
    response = service.alignn_predict_response("model-1", FakeOptimizer(), request)

    assert list(captured["frame"]["phase"]) == ["1"]
    assert response.records == [
        {"phase": "1", "temperature": 975.0, "property_mean": 0.5}
    ]


def test_alignn_candidate_service_forwards_structure_subset() -> None:
    captured: dict[str, object] = {}

    class FakeOptimizer:
        def candidate(self, **kwargs):
            captured.update(kwargs)
            return (
                pd.DataFrame(
                    [{"phase": "beta", "temperature": 1015.0}]
                ),
                0.75,
            )

    request = ALIGNNTabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["beta"],
        }
    )
    response = service.alignn_candidate_response(
        "model-1",
        FakeOptimizer(),
        request,
    )

    assert captured["structure_ids"] == ["beta"]
    assert captured["return_dataframe"] is True
    assert response.columns == ["phase", "temperature"]
    assert response.candidates == [{"phase": "beta", "temperature": 1015.0}]
    assert response.acq_value == pytest.approx(0.75)


def test_alignn_graph_schema_rejects_line_graph_disable() -> None:
    payload = _fit_payload()
    payload["structure_graph_config"] = {"compute_line_graph": False}

    with pytest.raises(ValueError, match="compute_line_graph=True"):
        ALIGNNTabularFitModelRequest.model_validate(payload)
