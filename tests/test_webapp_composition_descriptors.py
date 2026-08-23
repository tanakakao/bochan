from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bochan.api import InputTransformConfig, ModelConfig
from bochan.serving.webapp.composition.descriptors import (
    CompositionDescriptorInputTransform,
    build_composition_descriptor_input_transform,
    descriptor_feature_names,
)
from bochan.serving.webapp.composition.support import (
    normalize_web_composition_settings,
)
from bochan.serving.webapp.tabular_backend import (
    _descriptor_augmented_model_config,
)


def _composition_config(**updates):
    config = {
        "column": "formula",
        "elements": ["Fe", "Co"],
        "normalization": "atomic_fraction",
        "representation": "fractions",
        "reference_element": None,
        "feature_names": [
            "formula__fraction__Fe",
            "formula__fraction__Co",
        ],
        "include_descriptors": True,
        "descriptor_properties": ["atomic_number"],
        "descriptor_statistics": ["mean", "std"],
        "descriptor_include_num_elements": True,
        "descriptor_include_mixing_entropy": True,
        "element_properties": {},
    }
    config.update(updates)
    return config


def test_web_composition_settings_accept_descriptor_configuration() -> None:
    settings = normalize_web_composition_settings(
        {
            "column": "formula",
            "elements": ["Fe", "Co"],
            "include_descriptors": True,
            "descriptor_properties": ["atomic_number"],
            "descriptor_statistics": ["mean", "range"],
            "descriptor_include_num_elements": True,
            "descriptor_include_mixing_entropy": False,
        }
    )

    assert settings["include_descriptors"] is True
    assert settings["descriptor_properties"] == ["atomic_number"]
    assert settings["descriptor_statistics"] == ["mean", "range"]
    assert settings["descriptor_include_num_elements"] is True
    assert settings["descriptor_include_mixing_entropy"] is False


def test_descriptor_transform_appends_derived_values_without_new_decision_variables() -> None:
    config = _composition_config(descriptor_statistics=["mean"])
    bounds = torch.tensor(
        [[0.0, 0.0, 800.0], [1.0, 1.0, 1200.0]],
        dtype=torch.double,
    )
    transform, names, augmented_bounds = build_composition_descriptor_input_transform(
        feature_names=[
            "formula__fraction__Fe",
            "formula__fraction__Co",
            "temperature",
        ],
        bounds=bounds,
        categorical_idx=None,
        config=config,
        normalize=False,
    )

    raw = torch.tensor([[0.25, 0.75, 1000.0]], dtype=torch.double)
    transformed = transform(raw)

    assert isinstance(transform, CompositionDescriptorInputTransform)
    assert names == [
        "formula__descriptor__atomic_number__mean",
        "formula__descriptor__num_elements",
        "formula__descriptor__mixing_entropy",
    ]
    assert transformed.shape[-1] == raw.shape[-1] + len(names)
    assert transformed[0, :3].tolist() == raw[0].tolist()
    assert transformed[0, 3].item() == pytest.approx(26.75)
    assert transformed[0, 4].item() == pytest.approx(2.0)
    expected_entropy = -(
        0.25 * torch.log(torch.tensor(0.25))
        + 0.75 * torch.log(torch.tensor(0.75))
    )
    assert transformed[0, 5].item() == pytest.approx(float(expected_entropy))
    assert augmented_bounds.shape == (2, transformed.shape[-1])


def test_ilr_descriptor_transform_preserves_gradient_to_composition_coordinates() -> None:
    config = _composition_config(
        representation="ilr",
        feature_names=["formula__ilr__1"],
        descriptor_statistics=["mean"],
        descriptor_include_num_elements=False,
        descriptor_include_mixing_entropy=True,
    )
    bounds = torch.tensor(
        [[-8.0, 800.0], [8.0, 1200.0]],
        dtype=torch.double,
    )
    transform, names, _ = build_composition_descriptor_input_transform(
        feature_names=["formula__ilr__1", "temperature"],
        bounds=bounds,
        categorical_idx=None,
        config=config,
        normalize=False,
    )
    raw = torch.tensor([[0.3, 1000.0]], dtype=torch.double, requires_grad=True)

    transformed = transform(raw)
    transformed[..., -len(names):].sum().backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert abs(raw.grad[0, 0].item()) > 0


def test_descriptor_feature_names_use_property_major_order() -> None:
    names = descriptor_feature_names(
        _composition_config(
            descriptor_properties=["atomic_number", "atomic_weight"],
            descriptor_statistics=["mean", "range"],
            descriptor_include_num_elements=False,
            descriptor_include_mixing_entropy=False,
        )
    )
    assert names == [
        "formula__descriptor__atomic_number__mean",
        "formula__descriptor__atomic_number__range",
        "formula__descriptor__atomic_weight__mean",
        "formula__descriptor__atomic_weight__range",
    ]


def test_web_model_config_gets_descriptor_input_transform() -> None:
    composition = _composition_config(descriptor_statistics=["mean"])
    model_config = ModelConfig(
        task_type="regression",
        model_type="base",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=False,
        ),
    )
    encoded = {
        "feature_columns": [
            "formula__fraction__Fe",
            "formula__fraction__Co",
            "temperature",
        ],
        "bounds": [[0.0, 0.0, 800.0], [1.0, 1.0, 1200.0]],
        "cat_dims": [],
    }

    resolved = _descriptor_augmented_model_config(
        model_config=model_config,
        encoded_features=encoded,
        composition_config=composition,
    )

    assert resolved.input_transform is not None
    assert resolved.input_transform_config is None
    assert composition["descriptor_feature_names"]
    assert len(composition["model_feature_names"]) > len(encoded["feature_columns"])


def test_descriptor_mode_rejects_input_perturbation_and_crabnet_combination() -> None:
    composition = _composition_config()
    encoded = {
        "feature_columns": [
            "formula__fraction__Fe",
            "formula__fraction__Co",
        ],
        "bounds": [[0.0, 0.0], [1.0, 1.0]],
        "cat_dims": [],
    }
    with pytest.raises(ValueError, match="input perturbation"):
        _descriptor_augmented_model_config(
            model_config=ModelConfig(
                task_type="regression",
                model_type="base",
                input_transform_config=InputTransformConfig(
                    normalize=True,
                    perturbation=True,
                ),
            ),
            encoded_features=encoded,
            composition_config=dict(composition),
        )

    with pytest.raises(ValueError, match="CrabNet"):
        _descriptor_augmented_model_config(
            model_config=ModelConfig(
                task_type="regression",
                model_type="crabnet_gp",
            ),
            encoded_features=encoded,
            composition_config=dict(composition),
        )


def test_react_composition_settings_expose_descriptor_controls() -> None:
    extension = Path("web/src/compositionExtension.ts").read_text(encoding="utf-8")
    model_settings = Path(
        "web/src/components/CompositionModelSettings.tsx"
    ).read_text(encoding="utf-8")

    assert "includeDescriptors" in extension
    assert "descriptor_properties" in extension
    assert "descriptor_statistics" in extension
    assert "元素物性記述子を追加" in model_settings
    assert "探索変数にせず" in model_settings
