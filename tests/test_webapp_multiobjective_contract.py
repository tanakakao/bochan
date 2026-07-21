from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from bochan.serving.webapp.app import WEB_CAPABILITIES, RegressionRunRequest
from bochan.serving.webapp.target_roles import (
    apply_target_roles,
    build_target_constraint_config,
    level_set_thresholds,
    objective_values_direct,
    objective_weights,
    optimized_targets,
    output_spec_kwargs,
    select_optimized_values,
)
from bochan.serving.webapp.target_settings import (
    _build_outcome_constraint_config,
    _encode_targets,
    _resolve_target_settings,
    _resolve_targets,
)


def test_web_capabilities_advertise_requested_models() -> None:
    assert WEB_CAPABILITIES["model_types"] == [
        "base",
        "deepgp",
        "deepkernel",
        "saas",
        "pca",
        "rembo",
        "robust",
        "hetero",
        "multitask",
    ]
    assert {"EHVI", "NEHVI"}.issubset(WEB_CAPABILITIES["acquisitions"])


def test_regression_request_accepts_multiple_targets_and_family_marker() -> None:
    request = RegressionRunRequest(
        dataset_id="dataset-1",
        feature_columns=["x1", "x2"],
        target_columns=["strength", "cost"],
        directions={"strength": "maximize", "cost": "minimize"},
        outcome_constraints=[
            {"target": "strength", "operator": ">=", "value": 10.0},
            {"target": "cost", "operator": "<=", "value": 5.0},
        ],
        acquisition={
            "name": "variance",
            "acqf_kwargs": {"web_family": "active_learning"},
        },
    )

    targets, directions = _resolve_targets(request)

    assert targets == ["strength", "cost"]
    assert directions == {"strength": "maximize", "cost": "minimize"}
    assert request.acquisition.acqf_kwargs["web_family"] == "active_learning"
    assert len(request.outcome_constraints) == 2


def test_minimize_target_constraint_is_transformed_to_model_space() -> None:
    request = SimpleNamespace(
        outcome_constraints=[
            SimpleNamespace(target="strength", operator=">=", value=10.0),
            SimpleNamespace(target="cost", operator="<=", value=5.0),
        ]
    )

    config = _build_outcome_constraint_config(
        request,
        target_columns=["strength", "cost"],
        directions={"strength": "maximize", "cost": "minimize"},
    )

    assert config is not None
    assert config.output_indices == [0, 1]
    assert config.operators == ["ge", "ge"]
    assert config.thresholds == [10.0, -5.0]


def test_target_settings_accept_optional_constraints_and_class_metadata() -> None:
    request = SimpleNamespace(
        target_columns=["yield", "quality", "rank"],
        target_column=None,
        directions={},
        direction="maximize",
        model_kwargs={
            "web_target_settings": [
                {
                    "target": "quality",
                    "task_type": "classification",
                    "goal": "above",
                    "value": 0.6,
                    "target_classes": ["good", "excellent"],
                },
                {
                    "target": "rank",
                    "task_type": "ordinal",
                    "goal": "target",
                    "class_order": ["C", "B", "A"],
                    "target_values": ["B", "A"],
                },
                {
                    "target": "yield",
                    "task_type": "regression",
                    "goal": "none",
                    "value": None,
                },
            ],
            "web_target_roles": {
                "yield": {"optimize": True, "direction": "minimize"},
                "quality": {"optimize": False, "direction": "maximize"},
                "rank": {"optimize": True, "direction": "maximize"},
            },
            "n_components": 2,
        },
    )
    targets, directions = _resolve_targets(request)

    settings, model_kwargs = _resolve_target_settings(
        request,
        target_columns=targets,
        directions=directions,
    )
    settings, model_kwargs = apply_target_roles(
        settings,
        model_kwargs,
        directions=directions,
    )

    assert [setting["target"] for setting in settings] == targets
    assert settings[0]["goal"] == "none"
    assert settings[0]["direction"] == "minimize"
    assert settings[1]["target_classes"] == ["good", "excellent"]
    assert settings[1]["optimize"] is False
    assert settings[2]["class_order"] == ["C", "B", "A"]
    assert settings[2]["target_values"] == ["B", "A"]
    assert optimized_targets(settings) == ["yield", "rank"]
    assert model_kwargs == {"n_components": 2}


def test_target_value_cannot_be_constraint_only() -> None:
    with pytest.raises(ValueError, match="cannot be disabled"):
        apply_target_roles(
            [
                {
                    "target": "yield",
                    "task_type": "regression",
                    "goal": "target",
                    "value": 5.0,
                }
            ],
            {"web_target_roles": {"yield": {"optimize": False}}},
            directions={"yield": "maximize"},
        )


def test_target_encoding_supports_binary_target_class() -> None:
    data = pd.DataFrame({"quality": ["bad", "good", "good"]})
    settings = [
        {
            "target": "quality",
            "task_type": "classification",
            "goal": "none",
            "value": None,
            "target_class": "bad",
            "target_classes": ["bad"],
            "class_order": [],
            "target_values": [],
            "direction": "minimize",
            "optimize": True,
            "legacy": False,
        }
    ]

    encoded, metadata = _encode_targets(data, settings)
    spec = output_spec_kwargs(metadata["quality"])

    assert encoded["quality"].tolist() == [0.0, 1.0, 1.0]
    assert metadata["quality"]["internal_task"] == "binary"
    assert metadata["quality"]["class_indices"] == [0]
    assert metadata["quality"]["target_classes"] == ["bad"]
    assert spec["positive_class"] == 0
    assert spec["utility_values"] == [1.0, 0.0]
    assert spec["sign"] == -1.0


def test_multiclass_supports_multiple_target_classes_and_probability_constraint() -> None:
    data = pd.DataFrame({"class": ["a", "b", "c", "a"]})
    settings = [
        {
            "target": "class",
            "task_type": "classification",
            "goal": "above",
            "value": 0.7,
            "target_class": None,
            "target_classes": ["a", "c"],
            "class_order": [],
            "target_values": [],
            "direction": "maximize",
            "optimize": False,
            "legacy": False,
        }
    ]

    _, metadata = _encode_targets(data, settings)
    spec = output_spec_kwargs(metadata["class"])

    assert metadata["class"]["internal_task"] == "multiclass"
    assert metadata["class"]["class_indices"] == [0, 2]
    assert metadata["class"]["target_classes"] == ["a", "c"]
    assert metadata["class"]["configured_value"] == 0.7
    assert spec["utility_values"] == [1.0, 0.0, 1.0]
    assert spec["sign"] == 1.0


def test_ordinal_supports_custom_order_and_multiple_target_values() -> None:
    data = pd.DataFrame({"rank": ["low", "medium", "high", "low"]})
    settings = [
        {
            "target": "rank",
            "task_type": "ordinal",
            "goal": "target",
            "value": None,
            "target_class": None,
            "target_classes": [],
            "class_order": ["high", "medium", "low"],
            "target_values": ["high", "medium"],
            "direction": "minimize",
            "optimize": True,
            "legacy": False,
        }
    ]

    encoded, metadata = _encode_targets(data, settings)
    spec = output_spec_kwargs(metadata["rank"])

    assert encoded["rank"].tolist() == [2.0, 1.0, 0.0, 2.0]
    assert metadata["rank"]["class_order"] == ["high", "medium", "low"]
    assert metadata["rank"]["class_indices"] == [0, 1]
    assert metadata["rank"]["target_values"] == ["high", "medium"]
    assert spec["utility_values"] == [0, 0, -1]
    assert spec["sign"] == 1.0


def test_regression_direction_is_independent_from_constraint_sense() -> None:
    data = pd.DataFrame({"yield": [1.0, 5.0, 9.0]})
    settings = [
        {
            "target": "yield",
            "task_type": "regression",
            "goal": "above",
            "value": 5.0,
            "direction": "minimize",
            "optimize": True,
            "legacy": False,
        }
    ]
    _, metadata = _encode_targets(data, settings)
    spec = output_spec_kwargs(metadata["yield"])
    config = build_target_constraint_config(
        SimpleNamespace(outcome_constraints=[]),
        target_settings=settings,
        target_metadata=metadata,
        target_columns=["yield"],
        directions={"yield": "minimize"},
        hybrid_model=True,
    )

    assert spec == {"sign": -1.0, "eq_target": None}
    assert config is not None
    assert len(config.constraints) == 1
    assert config.constraints[0].sense == "le"
    assert config.constraints[0].threshold == -5.0


def test_direct_objective_values_and_output_selection_use_roles() -> None:
    values = torch.tensor([[1.0, 10.0, 3.0], [2.0, 8.0, 5.0]])
    settings = [
        {"target": "a", "goal": "none", "direction": "maximize", "optimize": True},
        {"target": "b", "goal": "below", "direction": "minimize", "optimize": False},
        {"target": "c", "goal": "target", "value": 4.0, "direction": "maximize", "optimize": True},
    ]

    transformed = objective_values_direct(values, settings)
    selected = select_optimized_values(
        transformed,
        target_columns=["a", "b", "c"],
        objective_targets=["a", "c"],
    )

    assert transformed.tolist() == [[1.0, -10.0, -1.0], [2.0, -8.0, -1.0]]
    assert selected.tolist() == [[1.0, -1.0], [2.0, -1.0]]
    assert objective_weights(
        target_columns=["a", "b", "c"],
        objective_targets=["a", "c"],
    ) == [1.0, 0.0, 1.0]


def test_level_set_thresholds_follow_direction_and_objective_selection() -> None:
    metadata = {
        "yield": {
            "goal": "above",
            "direction": "minimize",
            "internal_task": "regression",
            "configured_value": 5.0,
        },
        "quality": {
            "goal": "above",
            "direction": "maximize",
            "internal_task": "binary",
            "configured_value": 0.8,
        },
        "rank": {
            "goal": "target",
            "direction": "maximize",
            "internal_task": "ordinal",
            "class_index": 2,
        },
    }

    thresholds = level_set_thresholds(
        target_columns=["yield", "quality", "rank"],
        target_metadata=metadata,
        objective_targets=["yield", "rank"],
    )

    assert thresholds == [-5.0, 0.0, 0.0]
