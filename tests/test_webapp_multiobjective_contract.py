from __future__ import annotations

from types import SimpleNamespace

from bochan.serving.webapp.app import RegressionRunRequest, WEB_CAPABILITIES
from bochan.serving.webapp.workflows import (
    _build_outcome_constraint_config,
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
