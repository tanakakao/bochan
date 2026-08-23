from __future__ import annotations

from pathlib import Path

import pytest

from bochan.serving.webapp.routers.capabilities import WEB_CAPABILITIES
from bochan.serving.webapp.workflows.tabular import _resolve_crabnet_web_model


ROOT = Path(__file__).resolve().parents[1]


def _composition_config() -> dict[str, object]:
    return {
        "column": "formula",
        "elements": ["Ba", "Sr", "Ti", "O"],
    }


def _encoded(*, categorical: bool) -> dict[str, object]:
    return {
        "feature_columns": ["formula", "temperature", "atmosphere"],
        "cat_dims": [2] if categorical else [],
    }


def test_web_crabnet_mixed_gp_accepts_categorical_process_and_freezes_encoder() -> None:
    model_kwargs, metadata = _resolve_crabnet_web_model(
        model_type="crabnet_mixed_gp",
        model_kwargs={"checkpoint": " /srv/crabnet.pth "},
        target_columns=["property"],
        internal_tasks=["regression"],
        feature_columns=["formula", "temperature", "atmosphere"],
        encoded_features=_encoded(categorical=True),
        composition_config=_composition_config(),
        input_perturbation=False,
    )

    assert model_kwargs["checkpoint"] == "/srv/crabnet.pth"
    assert metadata is not None
    assert metadata["encoder_training"] == "frozen"
    assert metadata["continuous_process_columns"] == ["temperature"]
    assert metadata["categorical_process_columns"] == ["atmosphere"]
    assert metadata["categorical_representation"] == "categorical_kernel"


def test_web_crabnet_mixed_gp_requires_categorical_process() -> None:
    with pytest.raises(ValueError, match="requires at least one categorical process column"):
        _resolve_crabnet_web_model(
            model_type="crabnet_mixed_gp",
            model_kwargs={},
            target_columns=["property"],
            internal_tasks=["regression"],
            feature_columns=["formula", "temperature"],
            encoded_features={
                "feature_columns": ["formula", "temperature"],
                "cat_dims": [],
            },
            composition_config=_composition_config(),
            input_perturbation=False,
        )


def test_web_standard_crabnet_points_categorical_process_to_mixed_model() -> None:
    with pytest.raises(ValueError, match="Use crabnet_mixed_gp"):
        _resolve_crabnet_web_model(
            model_type="crabnet_gp",
            model_kwargs={},
            target_columns=["property"],
            internal_tasks=["regression"],
            feature_columns=["formula", "temperature", "atmosphere"],
            encoded_features=_encoded(categorical=True),
            composition_config=_composition_config(),
            input_perturbation=False,
        )


def test_web_capabilities_and_react_options_publish_crabnet_mixed_gp() -> None:
    assert "crabnet_mixed_gp" in WEB_CAPABILITIES["model_types"]
    crabnet = WEB_CAPABILITIES["crabnet"]
    assert "crabnet_mixed_gp" in crabnet["model_types"]
    assert crabnet["mixed_process_model_types"] == [
        "crabnet_mixed_gp",
        "crabnet_mixed_dkl",
    ]

    options = (ROOT / "web" / "src" / "modelOptions.ts").read_text(encoding="utf-8")
    settings = (ROOT / "web" / "src" / "pages" / "SettingsPage.tsx").read_text(encoding="utf-8")
    assert 'value: "crabnet_mixed_gp"' in options
    assert "isCrabNetMixedModelType" in settings
