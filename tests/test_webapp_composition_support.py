from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bochan.serving.webapp import app
from bochan.serving.webapp.composition.support import (
    apply_composition_best_subset_optimizer_kwargs,
    composition_model_feature_columns,
    composition_site,
    normalize_web_composition_settings,
)
from bochan.serving.webapp.routers.composition import create_composition_router
from bochan.serving.webapp.services.visualization_sessions import (
    VisualizationSession,
    visualization_options,
)
from bochan.serving.webapp.workflows import tabular as workflows_tabular
from bochan.tabular.config import TabularDataConfig
from bochan.tabular.data import dataframe_to_tensors


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
    assert settings["support_selection"] == "repair"
    assert settings["total"] == 1.0
    assert settings["bounds"]["Fe"] == (0.1, 0.8)
    assert settings["steps"]["Fe"] == 0.01
    assert settings["element_constraints"][0]["terms"][0] == {
        "site": "composition",
        "element": "Co",
        "coefficient": 1.0,
    }


def test_normalize_web_composition_settings_supports_best_subset() -> None:
    settings = normalize_web_composition_settings(
        {
            "column": "formula",
            "elements": ["Al", "Ti", "V", "Cr", "Nb"],
            "representation": "fractions",
            "min_components": 3,
            "max_components": 3,
            "required_components": ["Al"],
            "forbidden_components": ["Cr"],
            "support_selection": "best_subset",
            "best_subset_strategy": "beam",
            "best_subset_max_combinations": 1000,
            "best_subset_beam_width": 6,
            "best_subset_beam_steps": 5,
            "best_subset_max_evaluations": 120,
            "bounds": {
                "Al": [0.05, 0.8],
                "Cr": [0.0, 0.8],
            },
        }
    )

    assert settings["support_selection"] == "best_subset"
    assert settings["best_subset_strategy"] == "beam"
    assert settings["best_subset_max_combinations"] == 1000
    assert settings["best_subset_beam_width"] == 6
    assert settings["best_subset_beam_steps"] == 5
    assert settings["best_subset_max_evaluations"] == 120
    assert settings["forbidden_components"] == ["Cr"]
    assert settings["bounds"]["Cr"] == (0.0, 0.0)

    site = composition_site(settings)
    assert site["support_selection"] == "best_subset"
    assert site["forbidden_components"] == ["Cr"]

    optimizer_kwargs: dict[str, Any] = {"batch_limit": 4}
    apply_composition_best_subset_optimizer_kwargs(optimizer_kwargs, settings)
    assert optimizer_kwargs == {
        "batch_limit": 4,
        "best_subset_strategy": "beam",
        "best_subset_max_combinations": 1000,
        "best_subset_beam_width": 6,
        "best_subset_beam_steps": 5,
        "best_subset_max_evaluations": 120,
    }


def test_web_best_subset_contract_rejects_semantically_invalid_settings() -> None:
    base = {
        "column": "formula",
        "elements": ["Al", "Ti", "V"],
        "representation": "fractions",
        "min_components": 2,
        "max_components": 2,
        "support_selection": "best_subset",
    }
    with pytest.raises(ValueError, match="representation='fractions'"):
        normalize_web_composition_settings({**base, "representation": "ilr"})
    with pytest.raises(ValueError, match="min_components == max_components"):
        normalize_web_composition_settings({**base, "max_components": 3})
    with pytest.raises(ValueError, match="continuous fractions"):
        normalize_web_composition_settings({**base, "steps": {"Ti": 0.05}})
    with pytest.raises(ValueError, match="both required and forbidden"):
        normalize_web_composition_settings(
            {
                **base,
                "required_components": ["Al"],
                "forbidden_components": ["Al"],
            }
        )


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

    def base_run(request: Any) -> dict[str, Any]:
        return request.model_dump()

    test_app.include_router(
        create_composition_router(run_regression=base_run),
        prefix="/api/v1",
    )
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
    assert composition["support_selection"] == "repair"
    assert composition["element_constraints"][0]["operator"] == "="


def test_typed_composition_regression_endpoint_transports_best_subset() -> None:
    test_app = FastAPI()

    def base_run(request: Any) -> dict[str, Any]:
        return request.model_dump()

    test_app.include_router(
        create_composition_router(run_regression=base_run),
        prefix="/api/v1",
    )
    response = TestClient(test_app).post(
        "/api/v1/composition/regression/run",
        json={
            "run": {
                "dataset_id": "dataset-1",
                "feature_columns": ["formula", "temperature"],
                "target_column": "property",
            },
            "composition": {
                "column": "formula",
                "elements": ["Al", "Ti", "V", "Cr"],
                "representation": "fractions",
                "min_components": 3,
                "max_components": 3,
                "required_components": ["Al"],
                "forbidden_components": ["Cr"],
                "support_selection": "best_subset",
                "best_subset_strategy": "auto",
                "best_subset_max_combinations": 500,
            },
        },
    )

    assert response.status_code == 200, response.text
    composition = response.json()["model_kwargs"]["web_composition"]
    assert composition["support_selection"] == "best_subset"
    assert composition["forbidden_components"] == ["Cr"]
    assert composition["best_subset_strategy"] == "auto"
    assert composition["best_subset_max_combinations"] == 500


def test_ordinary_constraint_uses_shifted_index_after_ilr_expansion() -> None:
    constraint = SimpleNamespace(
        enabled=True,
        sense="le",
        rhs=1000.0,
        terms=[SimpleNamespace(column="temperature", coefficient=1.0)],
    )
    feature_columns = composition_model_feature_columns(
        ["formula", "temperature"],
        {
            "column": "formula",
            "feature_names": ["formula__ilr__1", "formula__ilr__2"],
        },
    )
    equality, inequality = workflows_tabular.botorch_linear_constraints(
        [constraint],
        feature_columns=feature_columns,
    )
    assert equality == []
    indices, coefficients, rhs = inequality[0]
    assert torch.equal(indices, torch.tensor([2]))
    assert torch.equal(
        coefficients,
        torch.tensor([-1.0], dtype=torch.double),
    )
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


def test_web_source_exposes_react_owned_composition_controls() -> None:
    api = Path("web/src/api.ts").read_text(encoding="utf-8")
    kind = Path("web/src/components/CompositionKindControl.tsx").read_text(
        encoding="utf-8"
    )
    model = Path("web/src/components/CompositionModelSettings.tsx").read_text(
        encoding="utf-8"
    )
    candidate = Path(
        "web/src/components/CompositionCandidateConstraints.tsx"
    ).read_text(encoding="utf-8")
    best_subset = Path(
        "web/src/components/CompositionBestSubsetSettings.tsx"
    ).read_text(encoding="utf-8")
    search_variables = Path(
        "web/src/components/SearchVariableSettings.tsx"
    ).read_text(encoding="utf-8")
    extension = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")
    main_source = Path("web/src/main.tsx").read_text(encoding="utf-8")

    assert "通常" in kind
    assert "組成式" in kind
    assert "web_composition" in api
    assert 'value="ilr"' in model
    assert 'value="clr"' in model
    assert 'value="alr"' in model
    assert "組成式のモデル変換" in model
    assert "元素間の線形制約" in candidate
    assert "組成候補の元素制約" in candidate
    assert "Acquisition-aware Best Subset" in best_subset
    assert "禁止元素" in best_subset
    assert "Auto（小規模Exact / 大規模Beam）" in best_subset
    assert "CompositionBestSubsetSettings" in search_variables
    assert "best_subset_strategy" in extension
    assert "forbidden_components" in extension
    assert "installCompositionRuntime" not in main_source
    assert "installCompositionPrepareControls" not in main_source
    assert "installCompositionExtension" not in main_source
    assert not Path("web/src/compositionRuntime.ts").exists()
