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


def test_web_crabnet_mixed_dkl_accepts_categories_and_encoder_training() -> None:
    model_kwargs, metadata = _resolve_crabnet_web_model(
        model_type="crabnet_mixed_dkl",
        model_kwargs={
            "checkpoint": " /srv/crabnet.pth ",
            "encoder_training": "full",
        },
        target_columns=["property"],
        internal_tasks=["regression"],
        feature_columns=["formula", "temperature", "atmosphere"],
        encoded_features=_encoded(categorical=True),
        composition_config=_composition_config(),
        input_perturbation=False,
    )

    assert model_kwargs["checkpoint"] == "/srv/crabnet.pth"
    assert model_kwargs["encoder_training"] == "full"
    assert metadata is not None
    assert metadata["encoder_training"] == "full"
    assert metadata["continuous_process_columns"] == ["temperature"]
    assert metadata["categorical_process_columns"] == ["atmosphere"]
    assert metadata["categorical_representation"] == "embedding"
    assert metadata["n_outputs"] == 1
    assert metadata["output_structure"] == "single"


def test_web_crabnet_mixed_dkl_accepts_independent_multioutput_regression() -> None:
    _, metadata = _resolve_crabnet_web_model(
        model_type="crabnet_mixed_dkl",
        model_kwargs={"encoder_training": "partial"},
        target_columns=["permittivity", "loss"],
        internal_tasks=["regression", "regression"],
        feature_columns=["formula", "temperature", "atmosphere"],
        encoded_features=_encoded(categorical=True),
        composition_config=_composition_config(),
        input_perturbation=False,
    )

    assert metadata is not None
    assert metadata["n_outputs"] == 2
    assert metadata["output_structure"] == "independent_model_list"


def test_web_other_crabnet_variants_remain_single_output() -> None:
    with pytest.raises(ValueError, match="Use crabnet_mixed_dkl"):
        _resolve_crabnet_web_model(
            model_type="crabnet_mixed_gp",
            model_kwargs={},
            target_columns=["permittivity", "loss"],
            internal_tasks=["regression", "regression"],
            feature_columns=["formula", "temperature", "atmosphere"],
            encoded_features=_encoded(categorical=True),
            composition_config=_composition_config(),
            input_perturbation=False,
        )


def test_web_crabnet_mixed_dkl_rejects_non_regression_multioutput() -> None:
    with pytest.raises(ValueError, match="continuous regression targets only"):
        _resolve_crabnet_web_model(
            model_type="crabnet_mixed_dkl",
            model_kwargs={},
            target_columns=["property", "class"],
            internal_tasks=["regression", "binary"],
            feature_columns=["formula", "temperature", "atmosphere"],
            encoded_features=_encoded(categorical=True),
            composition_config=_composition_config(),
            input_perturbation=False,
        )


def test_web_crabnet_mixed_dkl_requires_categorical_process() -> None:
    with pytest.raises(ValueError, match="requires at least one categorical process column"):
        _resolve_crabnet_web_model(
            model_type="crabnet_mixed_dkl",
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


def test_web_crabnet_dkl_points_categories_to_mixed_dkl() -> None:
    with pytest.raises(ValueError, match="Use crabnet_mixed_dkl"):
        _resolve_crabnet_web_model(
            model_type="crabnet_dkl",
            model_kwargs={},
            target_columns=["property"],
            internal_tasks=["regression"],
            feature_columns=["formula", "temperature", "atmosphere"],
            encoded_features=_encoded(categorical=True),
            composition_config=_composition_config(),
            input_perturbation=False,
        )


def test_web_capabilities_and_react_options_publish_crabnet_mixed_dkl() -> None:
    assert "crabnet_mixed_dkl" in WEB_CAPABILITIES["model_types"]
    crabnet = WEB_CAPABILITIES["crabnet"]
    assert "crabnet_mixed_dkl" in crabnet["model_types"]
    assert "crabnet_mixed_dkl" in crabnet["mixed_process_model_types"]
    assert crabnet["mixed_categorical_embedding"] is True
    assert crabnet["single_output_regression_only"] is False
    assert crabnet["independent_multi_output_model_types"] == ["crabnet_mixed_dkl"]
    assert crabnet["multi_output_structure"] == "model_list"

    options = (ROOT / "web" / "src" / "modelOptions.ts").read_text(encoding="utf-8")
    settings = (ROOT / "web" / "src" / "components" / "CrabNetModelSettings.tsx").read_text(
        encoding="utf-8"
    )
    page = (ROOT / "web" / "src" / "pages" / "SettingsPage.tsx").read_text(encoding="utf-8")
    validation = (ROOT / "web" / "src" / "context" / "workbenchValidation.ts").read_text(
        encoding="utf-8"
    )
    api = (ROOT / "web" / "src" / "api.ts").read_text(encoding="utf-8")

    assert 'value: "crabnet_mixed_dkl"' in options
    assert 'modelType === "crabnet_mixed_dkl"' in settings
    assert "isCrabNetDKLModelType" in settings
    assert 'option.value === "crabnet_mixed_dkl"' in page
    assert 'modelType === "crabnet_mixed_dkl" && targetColumns.length > 1' in page
    assert "isCrabNetMixedModelType" in validation
    assert 'modelType === "crabnet_mixed_dkl" && targetColumns.length > 1' in validation
    assert 'modelType === "crabnet_mixed_dkl"' in api
    assert "isCrabNetDKLRunModel" in api
