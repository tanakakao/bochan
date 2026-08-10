from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[1]


def _classification_request(*, model_type: str, acquisition: str):
    from bochan.serving.webapp.app import RegressionRunRequest

    return RegressionRunRequest(
        dataset_id="unused",
        feature_columns=["x"],
        target_column="target",
        target_columns=["target"],
        directions={"target": "maximize"},
        model_type=model_type,
        model_kwargs={
            "web_target_settings": [
                {
                    "target": "target",
                    "task_type": "classification",
                    "optimize": True,
                    "direction": "maximize",
                    "goal": "none",
                    "value": None,
                    "target_class": 1,
                }
            ]
        },
        search_space=[
            {
                "name": "x",
                "type": "numeric",
                "lower": 0.0,
                "upper": 1.0,
                "fixed": False,
            }
        ],
        acquisition={
            "name": acquisition,
            "acqf_kwargs": {"web_family": "active_learning"},
        },
        optimizer={"name": "ga", "q": 1, "sequential": True},
    )


def test_tabpfn_classification_bald_is_rejected_before_model_construction() -> None:
    from bochan.serving.webapp.target_settings import _resolve_target_settings

    request = _classification_request(model_type="tabpfn", acquisition="BALD")

    with pytest.raises(ValueError, match="does not expose independent epistemic"):
        _resolve_target_settings(
            request,
            target_columns=["target"],
            directions={"target": "maximize"},
        )


def test_other_external_classifiers_keep_bald_compatibility() -> None:
    from bochan.serving.webapp.target_settings import _resolve_target_settings

    request = _classification_request(model_type="random_forest", acquisition="BALD")
    settings, _ = _resolve_target_settings(
        request,
        target_columns=["target"],
        directions={"target": "maximize"},
    )

    assert settings[0]["task_type"] == "classification"


def test_web_ui_uses_model_capabilities_for_task_acquisition_and_search_filters() -> None:
    model_options = (ROOT / "web" / "src" / "modelOptions.ts").read_text(encoding="utf-8")
    settings_page = (ROOT / "web" / "src" / "pages" / "SettingsPage.tsx").read_text(encoding="utf-8")
    optimize_page = (ROOT / "web" / "src" / "pages" / "OptimizePage.tsx").read_text(encoding="utf-8")

    assert "modelSupportsTaskType" in model_options
    assert "modelSupportsTaskType(option.value, task)" in settings_page
    assert "requiresDerivativeFreeSearch" in optimize_page
    assert 'name.toLowerCase() !== "bald"' in optimize_page
    assert 'modelType === "tabpfn"' in optimize_page
