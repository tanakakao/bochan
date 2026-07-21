from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from bochan.serving.webapp.app import RegressionRunRequest, WEB_CAPABILITIES
from bochan.serving.webapp.target_settings import (
    _build_outcome_constraint_config,
    _encode_targets,
    _output_spec_kwargs,
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


def test_regression_request_accepts_multiple_targets_and_constraints() -> None:
    request = RegressionRunRequest(
        dataset_id="dataset-1",
        feature_columns=["x1", "x2"],
        target_columns=["strength", "cost"],
        directions={"strength": "maximize", "cost": "minimize"},
        outcome_constraints=[
            {"target": "strength", "operator": ">=", "value": 10.0},
            {"target": "cost", "operator": "<=", "value": 5.0},
        ],
        acquisition={"name": "NEHVI"},
    )

    targets, directions = _resolve_targets(request)

    assert targets == ["strength", "cost"]
    assert directions == {"strength": "maximize", "cost": "minimize"}
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
            "n_components": 2,
        },
    )
    targets, directions = _resolve_targets(request)

    settings, model_kwargs = _resolve_target_settings(
        request,
        target_columns=targets,
        directions=directions,
    )

    assert [setting["target"] for setting in settings] == targets
    assert settings[0]["goal"] == "none"
    assert settings[1]["target_classes"] == ["good", "excellent"]
    assert settings[2]["class_order"] == ["C", "B", "A"]
    assert settings[2]["target_values"] == ["B", "A"]
    assert model_kwargs == {"n_components": 2}


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
            "legacy": False,
        }
    ]

    encoded, metadata = _encode_targets(data, settings)
    spec = _output_spec_kwargs(metadata["quality"])

    assert encoded["quality"].tolist() == [0.0, 1.0, 1.0]
    assert metadata["quality"]["internal_task"] == "binary"
    assert metadata["quality"]["class_indices"] == [0]
    assert metadata["quality"]["target_classes"] == ["bad"]
    assert spec["positive_class"] == 0
    assert spec["utility_values"] == [1.0, 0.0]


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
            "legacy": False,
        }
    ]

    _, metadata = _encode_targets(data, settings)
    spec = _output_spec_kwargs(metadata["class"])

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
            "legacy": False,
        }
    ]

    encoded, metadata = _encode_targets(data, settings)
    spec = _output_spec_kwargs(metadata["rank"])

    assert encoded["rank"].tolist() == [2.0, 1.0, 0.0, 2.0]
    assert metadata["rank"]["class_order"] == ["high", "medium", "low"]
    assert metadata["rank"]["class_indices"] == [0, 1]
    assert metadata["rank"]["target_values"] == ["high", "medium"]
    assert spec["utility_values"] == [0, 0, -1]


def test_regression_without_constraint_keeps_raw_maximization_objective() -> None:
    data = pd.DataFrame({"yield": [1.0, 5.0, 9.0]})
    settings = [
        {
            "target": "yield",
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "target_class": None,
            "target_classes": [],
            "class_order": [],
            "target_values": [],
            "legacy": False,
        }
    ]

    _, metadata = _encode_targets(data, settings)
    spec = _output_spec_kwargs(metadata["yield"])

    assert metadata["yield"]["configured_value"] is None
    assert spec == {"sign": 1.0, "eq_target": None}
