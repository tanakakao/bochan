from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.feasible import constraint_value_from_class_probs
from bochan.serving.webapp.search_settings import (
    botorch_linear_constraints,
    build_target_constraint_config,
    feature_constraint_results,
    normalize_feature_constraints,
    resolve_search_method,
)


def test_classification_above_and_below_use_selected_class_probability() -> None:
    request = SimpleNamespace(outcome_constraints=[])
    base_setting = {
        "target": "quality",
        "task_type": "classification",
        "optimize": False,
        "direction": "maximize",
        "value": 0.5,
        "target_class": "b",
        "target_classes": ["b"],
        "class_order": [],
        "target_values": [],
        "legacy": False,
    }
    metadata = {
        "quality": {
            **base_setting,
            "internal_task": "binary",
            "configured_value": 0.5,
            "class_indices": [1],
            "class_index": 1,
            "num_classes": 2,
        }
    }

    above = build_target_constraint_config(
        request,
        target_settings=[{**base_setting, "goal": "above"}],
        target_metadata=metadata,
        target_columns=["quality"],
        directions={"quality": "maximize"},
        hybrid_model=True,
    )
    below = build_target_constraint_config(
        request,
        target_settings=[{**base_setting, "goal": "below"}],
        target_metadata=metadata,
        target_columns=["quality"],
        directions={"quality": "maximize"},
        hybrid_model=True,
    )

    above_spec = above.constraints[0]
    below_spec = below.constraints[0]
    assert above_spec.target_class == 1
    assert below_spec.target_class == 1
    assert above_spec.sense == "ge"
    assert below_spec.sense == "le"

    probs = torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double)
    above_values = constraint_value_from_class_probs(probs, above_spec)
    below_values = constraint_value_from_class_probs(probs, below_spec)
    torch.testing.assert_close(
        above_values,
        torch.tensor([-0.3, 0.3], dtype=torch.double),
    )
    torch.testing.assert_close(
        below_values,
        torch.tensor([0.3, -0.3], dtype=torch.double),
    )


def test_feature_constraints_convert_to_botorch_convention() -> None:
    constraints = normalize_feature_constraints(
        [
            {
                "name": "upper",
                "terms": [{"column": "x1", "coefficient": 2.0}],
                "sense": "le",
                "rhs": 4.0,
                "enabled": True,
            },
            {
                "name": "lower",
                "terms": [{"column": "x2", "coefficient": 3.0}],
                "sense": "ge",
                "rhs": 6.0,
                "enabled": True,
            },
            {
                "name": "equal",
                "terms": [{"column": "x1", "coefficient": 1.0}],
                "sense": "eq",
                "rhs": 1.0,
                "enabled": True,
            },
        ],
        feature_columns=["x1", "x2"],
    )

    equality, inequality = botorch_linear_constraints(
        constraints,
        feature_columns=["x1", "x2"],
    )

    assert len(equality) == 1
    assert len(inequality) == 2
    torch.testing.assert_close(inequality[0][1], torch.tensor([-2.0], dtype=torch.double))
    assert inequality[0][2] == -4.0
    torch.testing.assert_close(inequality[1][1], torch.tensor([3.0], dtype=torch.double))
    assert inequality[1][2] == 6.0


def test_feature_constraint_results_distinguish_senses() -> None:
    constraints = normalize_feature_constraints(
        [
            {
                "name": "above",
                "terms": [{"column": "x", "coefficient": 1.0}],
                "sense": "ge",
                "rhs": 0.5,
            },
            {
                "name": "below",
                "terms": [{"column": "x", "coefficient": 1.0}],
                "sense": "le",
                "rhs": 0.5,
            },
        ],
        feature_columns=["x"],
    )

    results = feature_constraint_results({"x": 0.8}, constraints)
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False


def test_search_method_routing() -> None:
    assert resolve_search_method("normal", multi_objective=False) == (
        "optimize_acqf",
        {},
        False,
    )
    assert resolve_search_method("optimize_acqf", multi_objective=False) == (
        "optimize_acqf",
        {},
        False,
    )
    assert resolve_search_method("torch", multi_objective=False) == (
        "torch",
        {},
        False,
    )
    assert resolve_search_method("pso", multi_objective=False) == (
        "evo",
        {"method": "pso"},
        False,
    )
    assert resolve_search_method("thompson_sampling", multi_objective=False) == (
        "thompson_sampling",
        {},
        False,
    )
    assert resolve_search_method("nsgaii", multi_objective=True) == (
        "optimize_acqf",
        {},
        True,
    )


def test_nsgaii_rejects_single_objective() -> None:
    try:
        resolve_search_method("nsgaii", multi_objective=False)
    except ValueError as exc:
        assert "multi-objective" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected single-objective NSGA-II to be rejected.")
