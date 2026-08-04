from __future__ import annotations

from fastapi.testclient import TestClient

from bochan.serving.webapp import app
from bochan.serving.webapp.composition_web_support import (
    normalize_web_composition_settings,
)


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


def test_web_source_exposes_single_composition_and_linear_constraint_controls() -> None:
    source = open("web/src/compositionExtension.tsx", encoding="utf-8").read()
    main_source = open("web/src/main.tsx", encoding="utf-8").read()

    assert "通常カテゴリ" in source
    assert "組成式" in source
    assert 'value="ilr"' in source
    assert 'value="clr"' in source
    assert 'value="alr"' in source
    assert "元素間の線形制約" in source
    assert "web_composition" in source
    assert "一列" not in source or "単一" in source
    assert "installCompositionExtension" in main_source
