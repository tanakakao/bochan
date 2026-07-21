from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bochan.serving.webapp.app import RegressionRunRequest, WEB_CAPABILITIES
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


def test_target_settings_are_one_to_one_and_keep_target_order() -> None:
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
                    "goal": "target",
                    "value": "good",
                },
                {
                    "target": "rank",
                    "task_type": "ordinal",
                    "goal": "above",
                    "value": "B",
                },
                {
                    "target": "yield",
                    "task_type": "regression",
                    "goal": "target",
                    "value": 5.0,
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
    assert [setting["task_type"] for setting in settings] == [
        "regression",
        "classification",
        "ordinal",
    ]
    assert model_kwargs == {"n_components": 2}


def test_target_encoding_supports_regression_classification_and_ordinal() -> None:
    data = pd.DataFrame(
        {
            "yield": [1.0, 5.0, 9.0],
            "quality": ["bad", "good", "good"],
            "rank": ["A", "B", "C"],
        }
    )
    settings = [
        {
            "target": "yield",
            "task_type": "regression",
            "goal": "target",
            "value": 5.0,
            "legacy": False,
        },
        {
            "target": "quality",
            "task_type": "classification",
            "goal": "target",
            "value": "good",
            "legacy": False,
        },
        {
            "target": "rank",
            "task_type": "ordinal",
            "goal": "above",
            "value": "B",
            "legacy": False,
        },
    ]

    encoded, metadata = _encode_targets(data, settings)

    assert encoded.to_dict(orient="list") == {
        "yield": [1.0, 5.0, 9.0],
        "quality": [0.0, 1.0, 1.0],
        "rank": [0.0, 1.0, 2.0],
    }
    assert metadata["yield"]["internal_task"] == "regression"
    assert metadata["quality"]["internal_task"] == "binary"
    assert metadata["quality"]["class_index"] == 1
    assert metadata["rank"]["internal_task"] == "ordinal"
    assert metadata["rank"]["class_index"] == 1


def test_multiclass_probability_threshold_requires_target_value_mode() -> None:
    data = pd.DataFrame({"class": ["a", "b", "c", "a"]})
    settings = [
        {
            "target": "class",
            "task_type": "classification",
            "goal": "above",
            "value": 0.7,
            "legacy": False,
        }
    ]

    with pytest.raises(ValueError, match="exactly two classes"):
        _encode_targets(data, settings)
