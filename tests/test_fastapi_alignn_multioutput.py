"""FastAPI coverage for independent multi-output ALIGNN models."""

# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

from bochan.serving.fastapi.schemas.alignn_tabular import ALIGNNTabularFitModelRequest
from bochan.serving.fastapi.services import alignn_tabular as service


class FakeALIGNNGPModel(SimpleNamespace):
    pass


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


def _payload(*, task_type: str = "regression") -> dict[str, object]:
    return {
        "data": [
            {
                "phase": "alpha",
                "temperature": 900.0,
                "pressure": 0.8,
                "strength": 100.0,
                "conductivity": 2.1,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "pressure": 1.0,
                "strength": 115.0,
                "conductivity": 2.4,
            },
        ],
        "input_cols": ["temperature", "phase", "pressure"],
        "target_cols": ["strength", "conductivity"],
        "bounds": {
            "temperature": [850.0, 1100.0],
            "pressure": [0.5, 2.0],
        },
        "structure_col": "phase",
        "structure_catalog": {
            "alpha": _structure(5.43),
            "beta": _structure(5.50),
        },
        "model_config": {
            "task_type": task_type,
            "model_type": "alignn_gp",
            "model_kwargs": {"latent_dim": 8},
        },
        "fit_config": {"skip_fit": True},
    }


def test_alignn_fit_schema_accepts_multiple_continuous_targets() -> None:
    request = ALIGNNTabularFitModelRequest.model_validate(_payload())

    assert request.target_cols == ["strength", "conductivity"]
    assert request.bo_model_config.task_type == "regression"


def test_alignn_fit_schema_accepts_explicit_multi_objective_task() -> None:
    request = ALIGNNTabularFitModelRequest.model_validate(
        _payload(task_type="multi_objective")
    )

    assert request.bo_model_config.task_type == "multi_objective"
    assert request.target_cols == ["strength", "conductivity"]


def test_alignn_fit_schema_rejects_explicit_multi_output_config() -> None:
    payload = _payload()
    payload["multi_output_config"] = {"output_names": ["strength", "conductivity"]}

    with pytest.raises(ValueError, match="derives independent multi-output models automatically"):
        ALIGNNTabularFitModelRequest.model_validate(payload)


def test_alignn_fit_service_forwards_multi_target_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_builder = SimpleNamespace(config={"neighbor_strategy": "pure_torch"})

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit(self, frame):
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    monkeypatch.setattr(service, "graph_builder_from_request", lambda request: fake_builder)

    request = ALIGNNTabularFitModelRequest.model_validate(_payload())
    optimizer = service.fit_alignn_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    assert captured["kwargs"]["target_cols"] == ["strength", "conductivity"]
    assert captured["kwargs"]["structure_col"] == "phase"


def test_alignn_fit_response_exposes_independent_output_metadata(monkeypatch) -> None:
    first = FakeALIGNNGPModel(
        material_encoder=SimpleNamespace(initialization="random"),
        process_dim=2,
        structure_feature_cache_enabled=True,
    )
    second = FakeALIGNNGPModel(
        material_encoder=SimpleNamespace(initialization="random"),
        process_dim=2,
        structure_feature_cache_enabled=True,
    )
    model = SimpleNamespace(models=[first, second])
    bundle = SimpleNamespace(
        model=model,
        model_type="alignn_gp",
        input_type="normal",
        cat_dims=[],
    )
    dataset = SimpleNamespace(
        feature_names=["phase", "temperature", "pressure"],
        target_names=["strength", "conductivity"],
        category_maps={"phase": {"alpha": 0, "beta": 1}},
    )
    optimizer = SimpleNamespace(
        bo=SimpleNamespace(bundle=bundle),
        dataset=dataset,
        structure=SimpleNamespace(
            column="phase",
            structure_ids=("alpha", "beta"),
            num_structures=2,
            graph_builder=SimpleNamespace(config={"neighbor_strategy": "pure_torch"}),
        ),
    )
    response = SimpleNamespace(metadata={})
    monkeypatch.setattr(service, "build_fit_response", lambda model_id, owner: response)

    result = service.build_alignn_fit_response("model-1", optimizer)
    metadata = result.metadata["alignn"]

    assert metadata["multi_output"] is True
    assert metadata["num_outputs"] == 2
    assert metadata["output_names"] == ["strength", "conductivity"]
    assert metadata["output_dependency"] == "independent"
    assert [entry["name"] for entry in metadata["output_models"]] == [
        "strength",
        "conductivity",
    ]
