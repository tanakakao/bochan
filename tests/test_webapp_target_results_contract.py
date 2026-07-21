from __future__ import annotations

import pytest

from bochan.serving.webapp.target_results import _setting_constraint_result


def test_regression_without_constraint_is_always_feasible() -> None:
    result = _setting_constraint_result(
        {"target": "yield", "goal": "none"},
        {
            "internal_task": "regression",
            "configured_value": None,
            "class_index": None,
        },
        predicted_mean=5.4,
        row_index=0,
        class_probabilities={},
    )

    assert result["ok"] is True
    assert result["violation"] == 0.0


def test_regression_target_goal_is_not_reported_as_hard_constraint() -> None:
    result = _setting_constraint_result(
        {"target": "yield", "goal": "target"},
        {
            "internal_task": "regression",
            "configured_value": 5.0,
            "class_index": None,
        },
        predicted_mean=5.4,
        row_index=0,
        class_probabilities={},
    )

    assert result["ok"] is True
    assert result["violation"] == 0.0


def test_ordinal_above_uses_expected_rank_threshold() -> None:
    result = _setting_constraint_result(
        {"target": "grade", "goal": "above"},
        {
            "internal_task": "ordinal",
            "configured_value": "B",
            "class_index": 1,
        },
        predicted_mean=1.2,
        row_index=0,
        class_probabilities={},
    )

    assert result["ok"] is True
    assert result["threshold_rank"] == 1.0
    assert result["value"] == "B"


def test_selected_class_probability_below_reports_violation() -> None:
    result = _setting_constraint_result(
        {"target": "defect", "goal": "below"},
        {
            "internal_task": "binary",
            "configured_value": 0.2,
            "class_index": 0,
            "target_classes": ["ok"],
        },
        predicted_mean=0.35,
        row_index=0,
        class_probabilities={},
    )

    assert result["ok"] is False
    assert result["target_classes"] == ["ok"]
    assert result["violation"] == pytest.approx(0.15)
