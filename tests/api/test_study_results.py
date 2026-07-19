import math

import pytest
import torch

from bochan.api import BochanStudy, TrialState


def _single_objective_study(*, direction="maximize"):
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        metadata={"feature_names": ["temperature", "pressure"]},
        acq_config={
            "name": "EI",
            "objective_config": {"direction": direction},
        },
    )
    study.add_observations(
        torch.tensor(
            [[0.1, 0.2], [0.8, 0.4], [0.3, 0.9]],
            dtype=torch.double,
        ),
        torch.tensor([0.5, 1.4, 0.9], dtype=torch.double),
    )
    return study


def test_study_exposes_optuna_like_best_properties():
    study = _single_objective_study()

    assert study.best_trial.trial_id == 1
    assert study.best_value == pytest.approx(1.4)
    assert torch.equal(
        study.best_x,
        torch.tensor([0.8, 0.4], dtype=torch.double),
    )
    assert study.best_params == {
        "temperature": pytest.approx(0.8),
        "pressure": pytest.approx(0.4),
    }

    result = study.best_result()
    assert result["trial_id"] == 1
    assert result["value"] == pytest.approx(1.4)
    assert result["values"] == pytest.approx([1.4])
    assert result["direction"] == "maximize"
    assert result["params"] == study.best_params


def test_study_best_direction_is_inferred_from_objective_config():
    study = _single_objective_study(direction="minimize")

    assert study.best_trial.trial_id == 0
    assert study.best_value == pytest.approx(0.5)
    assert [trial.trial_id for trial in study.best_trials(top_k=2)] == [0, 2]


def test_study_best_helpers_filter_non_finite_completed_values():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds)
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
        torch.tensor([float("nan"), float("inf"), 0.7], dtype=torch.double),
    )

    assert study.best_trial.trial_id == 2
    assert math.isfinite(study.best_value)


def test_study_best_properties_reject_multiobjective_ambiguity():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        model_config={"task_type": "multi_objective", "model_type": "base"},
    )
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double),
        torch.tensor([[1.0, 4.0], [2.0, 2.0], [3.0, 3.0]], dtype=torch.double),
    )

    with pytest.raises(RuntimeError, match="no single best trial"):
        _ = study.best_trial

    assert study.get_best_trial(output_index=1, direction="minimize").trial_id == 1
    assert study.get_best_value(output_index=0, direction="maximize") == pytest.approx(3.0)


def test_study_returns_non_dominated_pareto_trials_for_mixed_directions():
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    study = BochanStudy(
        bounds=bounds,
        model_config={"task_type": "multi_objective", "model_type": "base"},
    )
    study.add_observations(
        torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.double),
        torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 3.0],
                [3.0, 2.0],
                [2.5, 2.5],
            ],
            dtype=torch.double,
        ),
    )

    pareto = study.pareto_trials(directions=["maximize", "minimize"])

    assert [trial.trial_id for trial in pareto] == [0, 2]
    assert all(trial.state == TrialState.COMPLETED for trial in pareto)


def test_best_params_accepts_explicit_names_and_validates_width():
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    study = BochanStudy(bounds=bounds)
    study.add_observations(
        torch.tensor([[0.2, 0.7]], dtype=torch.double),
        torch.tensor([1.0], dtype=torch.double),
    )

    assert study.get_best_params(param_names=["a", "b"]) == {
        "a": pytest.approx(0.2),
        "b": pytest.approx(0.7),
    }
    with pytest.raises(ValueError, match="must contain 2 names"):
        study.get_best_params(param_names=["only_one"])
