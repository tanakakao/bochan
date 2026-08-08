from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.serving.webapp.level_set_settings import (
    configure_level_set_acqf_kwargs,
    level_set_output_weights,
)
from bochan.serving.webapp.search_settings import build_target_constraint_config


def _meta(target: str, value: float) -> dict[str, object]:
    return {
        "target": target,
        "internal_task": "regression",
        "goal": "above",
        "configured_value": value,
        "direction": "maximize",
        "class_index": None,
        "class_indices": [],
        "num_classes": None,
    }


def _setting(
    target: str,
    *,
    optimize: bool = True,
    weight: float = 1.0,
    goal: str = "above",
) -> dict[str, object]:
    return {
        "target": target,
        "task_type": "regression",
        "optimize": optimize,
        "direction": "maximize",
        "goal": goal,
        "value": 1.0,
        "level_set_weight": weight,
        "legacy": False,
    }


def test_level_set_output_weights_keep_constraint_only_outputs_at_zero() -> None:
    actual = level_set_output_weights(
        target_columns=["a", "guard", "b"],
        target_settings=[
            _setting("a", weight=2.0),
            _setting("guard", optimize=False, weight=9.0),
            _setting("b", weight=0.5),
        ],
        objective_targets=["a", "b"],
    )
    assert actual == pytest.approx([2.0, 0.0, 0.5])


def test_level_set_output_weights_require_positive_total() -> None:
    with pytest.raises(ValueError, match="positive weight"):
        level_set_output_weights(
            target_columns=["a", "b"],
            target_settings=[_setting("a", weight=0.0), _setting("b", weight=0.0)],
            objective_targets=["a", "b"],
        )


@pytest.mark.parametrize(
    ("name", "parameter", "kwarg", "expected"),
    [
        ("straddle", 1.7, "beta", 1.7),
        ("boundaryvariance", 0.4, "tau", 0.4),
        ("icu", 0.3, "bandwidth", 0.3),
    ],
)
def test_level_set_parameter_is_routed_to_acquisition(
    name: str,
    parameter: float,
    kwarg: str,
    expected: float,
) -> None:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {"web_level_set_parameter": parameter}

    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key=name,
        train_x=train_x,
        target_columns=["y"],
        target_settings=[_setting("y")],
        target_metadata={"y": _meta("y", 0.5)},
        objective_targets=["y"],
        input_perturbation=False,
        n_w=4,
    )

    assert "web_level_set_parameter" not in kwargs
    assert kwargs[kwarg] == pytest.approx(expected)


def test_icu_zero_parameter_keeps_automatic_bandwidth() -> None:
    kwargs: dict[str, object] = {"web_level_set_parameter": 0.0}
    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key="icu",
        train_x=torch.tensor([[0.0]], dtype=torch.double),
        target_columns=["y"],
        target_settings=[_setting("y")],
        target_metadata={"y": _meta("y", 0.5)},
        objective_targets=["y"],
        input_perturbation=False,
        n_w=4,
    )
    assert "bandwidth" not in kwargs


def test_boundary_variance_rejects_zero_tau() -> None:
    kwargs: dict[str, object] = {"web_level_set_parameter": 0.0}
    with pytest.raises(ValueError, match="tau must be greater than zero"):
        configure_level_set_acqf_kwargs(
            kwargs,
            acq_key="boundaryvariance",
            train_x=torch.tensor([[0.0]], dtype=torch.double),
            target_columns=["y"],
            target_settings=[_setting("y")],
            target_metadata={"y": _meta("y", 0.5)},
            objective_targets=["y"],
            input_perturbation=False,
            n_w=4,
        )


def test_level_set_input_perturbation_cvar_builds_score_objective() -> None:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs: dict[str, object] = {"web_level_set_parameter": 1.96}

    configure_level_set_acqf_kwargs(
        kwargs,
        acq_key="straddle",
        train_x=train_x,
        target_columns=["a", "b"],
        target_settings=[_setting("a", weight=2.0), _setting("b", weight=1.0)],
        target_metadata={"a": _meta("a", 0.2), "b": _meta("b", 0.8)},
        objective_targets=["a", "b"],
        input_perturbation=True,
        n_w=8,
        risk_type="cvar",
        risk_alpha=0.25,
    )

    assert kwargs["n_w"] == 8
    objective = kwargs["objective"]
    assert objective.__class__.__name__ == "MultiOutputRegressionLevelSetScoreObjective"
    assert objective.n_w == 8
    assert objective.risk_type == "cvar"
    assert objective.alpha == pytest.approx(0.25)
    assert kwargs["output_weights"] == pytest.approx([2.0, 1.0])


def test_optimized_lse_boundary_is_not_a_hard_outcome_constraint() -> None:
    settings = [
        _setting("boundary", optimize=True, goal="above"),
        _setting("guard", optimize=False, goal="below"),
    ]
    metadata = {
        "boundary": _meta("boundary", 0.5),
        "guard": {**_meta("guard", 0.2), "goal": "below"},
    }

    config = build_target_constraint_config(
        SimpleNamespace(outcome_constraints=[]),
        target_settings=settings,
        target_metadata=metadata,
        target_columns=["boundary", "guard"],
        directions={"boundary": "maximize", "guard": "maximize"},
        hybrid_model=True,
        exclude_optimized_boundaries=True,
    )

    assert config is not None
    assert config.constraints is not None
    assert len(config.constraints) == 1
    constraint = config.constraints[0]
    assert constraint.output == "guard"
    assert constraint.sense == "le"
    assert constraint.threshold == pytest.approx(0.2)
