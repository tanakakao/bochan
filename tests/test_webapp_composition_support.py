from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bochan.serving.webapp import app
from bochan.serving.webapp import workflows_tabular
from bochan.serving.webapp.composition_web_routes import (
    register_composition_routes,
)
from bochan.serving.webapp.composition_web_support import (
    _ACTIVE_CONFIG,
    normalize_web_composition_settings,
)
from bochan.serving.webapp.visualization_sessions import (
    VisualizationSession,
    visualization_options,
)
from bochan.tabular.config import TabularDataConfig
from bochan.tabular.converter import dataframe_to_tensors


def test_normalize_web_composition_settings_supports_element_constraints() -> None:
    settings = normalize_web_composition_settings(
        {
            "column": "formula",
            "elements": ["Fe", "Co", "Ni"],
            "representation": "ilr",
            "normalization": "atomic_fraction",
            "bounds": {"Fe": [0.1, 0.8]},
            "steps": {"Fe": 0.01},
            "required_components": ["Fe"],
            "element_constraints": [
                {
                    "terms": [
                        {"element": "Co", "coefficient": 1.0},
                        {"element": "Fe", "coefficient": -0.5},
                    ],
                    "operator": "=",
                    "rhs": 0.0,
                    "basis": "atomic_amount",
                }
            ],
        }
    )

    assert settings["column"] == "formula"
    assert settings["representation"] == "ilr"
    assert settings["total"] == 1.0
    assert settings["bounds"]["Fe"] == (0.1, 0.8)
    assert settings["steps"]["Fe"] == 0.01
    assert settings["element_constraints"][0]["terms"][0] == {
        "site": "composition",
        "element": "Co",
        "coefficient": 1.0,
    }


def test_composition_validate_endpoint_infers_elements_and_normalizes_ratios() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/composition/validate",
        json={
            "formulas": ["Fe2Co5Ni4", "Fe4Co10Ni8"],
            "settings": {
                "column": "formula",
                "representation": "ilr",
                "normalization": "atomic_fraction",
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["elements"] == ["Fe", "Co", "Ni"]
    assert len(payload["feature_names"]) == 2
    first = payload["rows"][0]["fractions"]
    second = payload["rows"][1]["fractions"]
    assert first == second
    assert abs(sum(first.values()) - 1.0) < 1e-12
    assert abs(first["Fe"] - 2.0 / 11.0) < 1e-12


def test_typed_composition_regression_endpoint_injects_settings() -> None:
    test_app = FastAPI()

    @test_app.post("/api/v1/regression/run")
    def base_run(request: Any) -> dict[str, Any]:
        return request.model_dump()

    register_composition_routes(test_app)
    response = TestClient(test_app).post(
        "/api/v1/composition/regression/run",
        json={
            "run": {
                "dataset_id": "dataset-1",
                "feature_columns": ["formula", "temperature"],
                "target_column": "property",
                "search_space": [
                    {"name": "formula", "type": "auto"},
                    {
                        "name": "temperature",
                        "type": "numeric",
                        "lower": 800.0,
                        "upper": 1200.0,
                    },
                ],
            },
            "composition": {
                "column": "formula",
                "elements": ["Fe", "Co", "Ni"],
                "representation": "ilr",
                "element_constraints": [
                    {
                        "terms": [
                            {"element": "Co", "coefficient": 1.0},
                            {"element": "Fe", "coefficient": -0.5},
                        ],
                        "operator": "=",
                        "rhs": 0.0,
                    }
                ],
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    composition = payload["model_kwargs"]["web_composition"]
    assert composition["column"] == "formula"
    assert composition["representation"] == "ilr"
    assert composition["element_constraints"][0]["operator"] == "="


def test_ordinary_constraint_uses_shifted_index_after_ilr_expansion() -> None:
    constraint = SimpleNamespace(
        enabled=True,
        sense="le",
        rhs=1000.0,
        terms=[SimpleNamespace(column="temperature", coefficient=1.0)],
    )
    token = _ACTIVE_CONFIG.set(
        {
            "column": "formula",
            "feature_names": ["formula__ilr__1", "formula__ilr__2"],
        }
    )
    try:
        equality, inequality = workflows_tabular.botorch_linear_constraints(
            [constraint],
            feature_columns=["formula", "temperature"],
        )
    finally:
        _ACTIVE_CONFIG.reset(token)

    assert equality == []
    indices, coefficients, rhs = inequality[0]
    assert torch.equal(indices, torch.tensor([2]))
    assert torch.equal(coefficients, torch.tensor([-1.0], dtype=torch.double))
    assert rhs == -1000.0


def test_web_converter_accepts_pandas_string_dtype_categories() -> None:
    frame = pd.DataFrame(
        {
            "formula__ilr__1": [0.1, 0.2, 0.3],
            "phase": pd.Series(["beta", "beta", "beta"], dtype="string"),
        }
    )
    config = TabularDataConfig(
        input_cols=["formula__ilr__1", "phase"],
        categorical_cols=["phase"],
        category_maps={"phase": {"alpha": 0, "beta": 1}},
        bounds={
            "formula__ilr__1": [-8.0, 8.0],
            "phase": [0.0, 1.0],
        },
    )

    dataset = dataframe_to_tensors(frame, config)

    assert dataset.X[:, 1].tolist() == [1.0, 1.0, 1.0]
    assert dataset.category_maps == {"phase": {"alpha": 0, "beta": 1}}


def test_composition_formula_is_categorical_visualization_control() -> None:
    session = VisualizationSession(
        optimizer=SimpleNamespace(model=SimpleNamespace()),
        tabular_optimizer=SimpleNamespace(
            dataset=SimpleNamespace(cat_dims=[], category_maps={})
        ),
        data=pd.DataFrame(
            {
                "formula": pd.Series(["Al2O3", "Fe2O3"], dtype="string"),
                "temperature": [900.0, 1000.0],
            }
        ),
        encoded_targets=pd.DataFrame(),
        feature_columns=["formula", "temperature"],
        target_columns=["property"],
        target_metadata={"property": {"internal_task": "regression"}},
        hybrid_model=False,
    )

    options = visualization_options(session)

    assert options["numeric_features"] == ["temperature"]
    assert options["feature_controls"]["formula"] == {
        "kind": "categorical",
        "values": ["Al2O3", "Fe2O3"],
        "default": "Al2O3",
    }
    assert options["feature_controls"]["temperature"] == {
        "kind": "numeric",
        "min": 900.0,
        "max": 1000.0,
        "default": 950.0,
    }


def test_web_source_exposes_single_composition_and_linear_constraint_controls() -> None:
    source = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")
    main_source = Path("web/src/main.tsx").read_text(encoding="utf-8")

    assert "通常カテゴリ" in source
    assert "組成式" in source
    assert 'value="ilr"' in source
    assert 'value="clr"' in source
    assert 'value="alr"' in source
    assert "元素間の線形制約" in source
    assert "web_composition" in source
    assert "組成式のモデル変換" in source
    assert "組成候補の元素制約" in source
    assert "installCompositionExtension" in main_source
