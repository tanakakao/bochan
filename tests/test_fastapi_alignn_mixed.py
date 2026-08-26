"""FastAPI coverage for mixed ALIGNN structure/process models."""

# ruff: noqa: E402

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pd = pytest.importorskip("pandas")

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


def _mixed_fit_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "phase": "alpha",
                "temperature": 900.0,
                "furnace": "A",
                "pressure": 0.8,
                "atmosphere": "air",
                "property": 0.4,
            },
            {
                "phase": "beta",
                "temperature": 950.0,
                "furnace": "B",
                "pressure": 1.0,
                "atmosphere": "N2",
                "property": 0.8,
            },
            {
                "phase": "alpha",
                "temperature": 1000.0,
                "furnace": "A",
                "pressure": 1.2,
                "atmosphere": "Ar",
                "property": 0.7,
            },
        ],
        "input_cols": [
            "temperature",
            "furnace",
            "phase",
            "pressure",
            "atmosphere",
        ],
        "categorical_cols": ["furnace", "atmosphere"],
        "target_cols": "property",
        "bounds": {
            "temperature": [850.0, 1100.0],
            "pressure": [0.5, 2.0],
        },
        "category_maps": {
            "furnace": {"A": 0, "B": 1},
            "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
        },
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


def test_alignn_fit_schema_accepts_categorical_process_columns_without_bounds() -> None:
    request = ALIGNNTabularFitModelRequest.model_validate(_mixed_fit_payload())

    assert request.categorical_cols == ["furnace", "atmosphere"]
    assert request.bounds == {
        "temperature": [850.0, 1100.0],
        "pressure": [0.5, 2.0],
    }
    assert request.category_maps == {
        "furnace": {"A": 0, "B": 1},
        "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
    }


def test_alignn_fit_schema_still_requires_continuous_process_bounds() -> None:
    payload = _mixed_fit_payload()
    payload["bounds"] = {"temperature": [850.0, 1100.0]}

    with pytest.raises(ValueError, match="missing bounds for \\['pressure'\\]"):
        ALIGNNTabularFitModelRequest.model_validate(payload)


def test_alignn_fit_schema_rejects_unknown_categorical_process_column() -> None:
    payload = _mixed_fit_payload()
    payload["categorical_cols"] = ["furnace", "not_an_input"]

    with pytest.raises(ValueError, match="categorical_cols must be included in input_cols"):
        ALIGNNTabularFitModelRequest.model_validate(payload)


def test_alignn_fit_service_forwards_mixed_process_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_builder = SimpleNamespace(config={"neighbor_strategy": "pure_torch"})

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.model_config = SimpleNamespace(model_type="alignn_gp")

        def fit(self, frame):
            captured["frame"] = frame.copy()
            return self

    monkeypatch.setattr(service, "TabularBayesianOptimizer", FakeOptimizer)
    monkeypatch.setattr(service, "graph_builder_from_request", lambda request: fake_builder)

    request = ALIGNNTabularFitModelRequest.model_validate(_mixed_fit_payload())
    optimizer = service.fit_alignn_tabular_optimizer(request)

    assert isinstance(optimizer, FakeOptimizer)
    kwargs = captured["kwargs"]
    assert kwargs["categorical_cols"] == ["furnace", "atmosphere"]
    assert kwargs["category_maps"] == {
        "furnace": {"A": 0, "B": 1},
        "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
    }
    assert kwargs["bounds"] == {
        "temperature": [850.0, 1100.0],
        "pressure": [0.5, 2.0],
    }
    assert list(captured["frame"]["furnace"]) == ["A", "B", "A"]
    assert list(captured["frame"]["atmosphere"]) == ["air", "N2", "Ar"]


def test_alignn_fit_response_exposes_resolved_mixed_contract(monkeypatch) -> None:
    model = SimpleNamespace(
        material_encoder=SimpleNamespace(initialization="random"),
        process_dim=2,
    )
    bundle = SimpleNamespace(
        model=model,
        model_type="alignn_gp",
        input_type="mixed",
        cat_dims=[2, 4],
    )
    dataset = SimpleNamespace(
        feature_names=[
            "phase",
            "temperature",
            "furnace",
            "pressure",
            "atmosphere",
        ],
        category_maps={
            "phase": {"alpha": 0, "beta": 1},
            "furnace": {"A": 0, "B": 1},
            "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
        },
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

    assert metadata["input_type"] == "mixed"
    assert metadata["continuous_process_cols"] == ["temperature", "pressure"]
    assert metadata["categorical_process_cols"] == ["furnace", "atmosphere"]
    assert metadata["categorical_process_dims"] == [2, 4]
    assert metadata["category_maps"] == {
        "furnace": {"A": 0, "B": 1},
        "atmosphere": {"air": 0, "N2": 1, "Ar": 2},
    }


def test_alignn_mixed_predict_and_candidate_responses_keep_original_categories() -> None:
    class FakePredictOptimizer:
        structure = SimpleNamespace(column="phase")

        def predict(self, frame, **kwargs):
            return pd.DataFrame(
                {
                    "phase": frame["phase"],
                    "furnace": frame["furnace"],
                    "atmosphere": frame["atmosphere"],
                    "property_mean": [0.5] * len(frame),
                }
            )

    prediction_request = TabularPredictRequest.model_validate(
        {
            "data": [
                {
                    "phase": "alpha",
                    "temperature": 975.0,
                    "furnace": "B",
                    "pressure": 1.1,
                    "atmosphere": "Ar",
                }
            ],
            "include_input": True,
        }
    )
    prediction = service.alignn_predict_response(
        "model-1",
        FakePredictOptimizer(),
        prediction_request,
    )
    assert prediction.records == [
        {
            "phase": "alpha",
            "furnace": "B",
            "atmosphere": "Ar",
            "property_mean": 0.5,
        }
    ]

    class FakeCandidateOptimizer:
        def candidate(self, **kwargs):
            return (
                pd.DataFrame(
                    [
                        {
                            "phase": "beta",
                            "temperature": 1015.0,
                            "furnace": "B",
                            "pressure": 1.3,
                            "atmosphere": "N2",
                        }
                    ]
                ),
                0.75,
            )

    candidate_request = ALIGNNTabularCandidateRequest.model_validate(
        {
            "acquisition_config": {"name": "logei"},
            "optimize_config": {"q": 1},
            "structure_ids": ["beta"],
        }
    )
    candidate = service.alignn_candidate_response(
        "model-1",
        FakeCandidateOptimizer(),
        candidate_request,
    )
    assert candidate.candidates == [
        {
            "phase": "beta",
            "temperature": 1015.0,
            "furnace": "B",
            "pressure": 1.3,
            "atmosphere": "N2",
        }
    ]
