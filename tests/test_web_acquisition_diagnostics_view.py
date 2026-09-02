from types import SimpleNamespace

from bochan.serving.webapp.services.acquisition_diagnostics import (
    acquisition_diagnostics_view_for_optimizer,
    build_acquisition_diagnostics_view,
)


def test_build_acquisition_diagnostics_view_reports_summary_cards_and_warnings():
    diagnostics = {
        "training_rows": 3,
        "baseline_rows": 2,
        "baseline_source": "automatic",
        "baseline_filtered": True,
        "partial_observation": True,
        "observed_per_output": [2, 2],
        "objective_output_indices": [0],
        "known_observation_variance": True,
        "failed_rows": 1,
        "pending_rows": 1,
        "failed_excluded_from_objective_training": True,
        "pending_excluded_from_objective_training": True,
    }

    view = build_acquisition_diagnostics_view(diagnostics)

    assert view["available"] is True
    cards = {card["key"]: card["value"] for card in view["cards"]}
    assert cards == {
        "training_rows": 3,
        "baseline_rows": 2,
        "failed_rows": 1,
        "pending_rows": 1,
        "known_observation_variance": True,
    }
    assert view["details"] == {
        "baseline_source": "automatic",
        "baseline_filtered": True,
        "partial_observation": True,
        "observed_per_output": [2, 2],
        "objective_output_indices": [0],
        "known_observation_variance": True,
    }
    assert len(view["warnings"]) == 4


def test_build_acquisition_diagnostics_view_is_safe_when_unavailable():
    view = build_acquisition_diagnostics_view(None)

    assert view == {
        "available": False,
        "cards": [],
        "warnings": [],
        "details": None,
        "observation_report": None,
    }


def test_optimizer_view_includes_canonical_observation_report():
    report = {
        "n_rows": 5,
        "n_completed": 4,
        "n_success": 3,
        "n_failed": 1,
        "n_pending": 1,
        "observed_per_output": [2, 2],
        "known_observation_variance": True,
    }
    optimizer = SimpleNamespace(
        last_acquisition_diagnostics={
            "training_rows": 3,
            "baseline_rows": 2,
            "baseline_source": "automatic",
            "baseline_filtered": True,
            "partial_observation": True,
            "observed_per_output": [2, 2],
            "objective_output_indices": [0],
            "known_observation_variance": True,
            "failed_rows": 1,
            "pending_rows": 1,
            "failed_excluded_from_objective_training": True,
            "pending_excluded_from_objective_training": True,
        },
        observations=SimpleNamespace(report=lambda: report),
    )

    view = acquisition_diagnostics_view_for_optimizer(optimizer)

    assert view["available"] is True
    assert view["observation_report"] == report
