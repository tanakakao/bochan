from __future__ import annotations

import pytest

from bochan.api import AcquisitionConfig
from bochan.tabular.optimizer_api import (
    _resolve_acquisition_config_columns,
    _resolve_outcome_constraint_config_columns,
)


def _constraint_config(*, target_class=None, target_classes=None):
    constraint = {
        "output": "y_ord_str",
        "threshold": 0.75,
        "sense": "ge",
    }
    if target_class is not None:
        constraint["target_class"] = target_class
    if target_classes is not None:
        constraint["target_classes"] = target_classes
    return {"constraints": [constraint]}


def test_string_target_classes_are_resolved_before_config_construction() -> None:
    resolved = _resolve_outcome_constraint_config_columns(
        _constraint_config(target_classes=["a", "c"]),
        ["property", "y_ord_str"],
        {"y_ord_str": {"a": 0, "b": 1, "c": 2}},
    )

    config = AcquisitionConfig(
        name="ei",
        outcome_constraint_config=resolved,
    )
    spec = config.outcome_constraint_config.wrapper_constraints()[0]

    assert spec.target_classes == (0, 2)


def test_single_string_target_classes_value_is_treated_as_one_label() -> None:
    resolved = _resolve_outcome_constraint_config_columns(
        _constraint_config(target_classes="a"),
        ["property", "y_ord_str"],
        {"y_ord_str": {"a": 0, "b": 1, "c": 2}},
    )

    assert resolved["constraints"][0]["target_classes"] == [0]


def test_nested_acquisition_config_resolves_string_target_class() -> None:
    resolved = _resolve_acquisition_config_columns(
        {
            "name": "ucb",
            "outcome_constraint_config": _constraint_config(target_class="b"),
        },
        ["property", "y_ord_str"],
        {"y_ord_str": {"a": 0, "b": 1, "c": 2}},
    )

    assert resolved["outcome_constraint_config"]["constraints"][0]["target_class"] == 1


def test_numeric_string_target_class_preserves_previous_index_behavior() -> None:
    resolved = _resolve_outcome_constraint_config_columns(
        _constraint_config(target_class="2"),
        ["property", "y_ord_str"],
        {},
    )

    assert resolved["constraints"][0]["target_class"] == 2


def test_unknown_string_target_class_reports_available_labels() -> None:
    with pytest.raises(KeyError, match="Available labels"):
        _resolve_outcome_constraint_config_columns(
            _constraint_config(target_classes=["unknown"]),
            ["property", "y_ord_str"],
            {"y_ord_str": {"a": 0, "b": 1, "c": 2}},
        )
