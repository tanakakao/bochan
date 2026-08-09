from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from bochan.serving.webapp import target_missing_policy as policy
from bochan.serving.webapp import target_settings


def _request(*, targets: list[str], model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        target_columns=targets,
        target_column=targets[0] if targets else None,
        model_type=model_type,
        model_kwargs={},
        search_space=[],
    )


def test_single_objective_drops_missing_target_rows() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, None, 3.0]})
    with policy.target_missing_run(_request(targets=["y"], model_type="base")) as report:
        cleaned = target_settings._clean_rows(
            data,
            ["x"],
            ["y"],
            drop_missing=True,
        )

    assert cleaned.to_dict("list") == {"x": [0.0, 2.0], "y": [1.0, 3.0]}
    assert report["policy"] == "drop_rows"
    assert report["target_missing_detected"] is True


def test_multiobjective_non_multitask_drops_any_missing_target_row() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y1": [1.0, None, 3.0],
            "y2": [2.0, 4.0, None],
        }
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="base")
    ):
        cleaned = target_settings._clean_rows(
            data,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )

    assert cleaned.to_dict("list") == {"x": [0.0], "y1": [1.0], "y2": [2.0]}


def test_multitask_preserves_partial_targets_and_drops_unusable_rows() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, None, 3.0],
            "y1": [1.0, None, 3.0, None],
            "y2": [None, 4.0, 5.0, None],
        }
    )
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ) as report:
        cleaned = target_settings._clean_rows(
            data,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )

    assert cleaned["x"].tolist() == [0.0, 1.0]
    assert cleaned["y1"].isna().tolist() == [False, True]
    assert cleaned["y2"].isna().tolist() == [True, False]
    assert report["policy"] == "wide_multitask"
    assert report["target_missing_detected"] is True
    assert report["target_missing_counts"] == {"y1": 1, "y2": 1}
    assert report["dropped_feature_rows"] == 1
    assert report["dropped_all_target_missing_rows"] == 1
    assert report["acquisition_baseline_completed"] is False


def test_multitask_requires_an_observation_for_every_target() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0], "y1": [1.0, 2.0], "y2": [None, None]})
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ), pytest.raises(ValueError, match="without observations"):
        target_settings._clean_rows(
            data,
            ["x"],
            ["y1", "y2"],
            drop_missing=True,
        )


def test_target_encoder_preserves_regression_nan_cells() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y1": [1.0, None],
            "y2": [None, 2.0],
        }
    )
    settings = [
        {
            "target": "y1",
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "legacy": False,
        },
        {
            "target": "y2",
            "task_type": "regression",
            "goal": "none",
            "value": None,
            "legacy": False,
        },
    ]
    with policy.target_missing_run(
        _request(targets=["y1", "y2"], model_type="multitask")
    ):
        encoded, metadata = target_settings._encode_targets(data, settings)

    assert encoded["y1"].isna().tolist() == [False, True]
    assert encoded["y2"].isna().tolist() == [True, False]
    assert metadata["y1"]["internal_task"] == "regression"
    assert metadata["y2"]["internal_task"] == "regression"


def test_target_settings_facade_uses_source_level_policy_functions() -> None:
    assert target_settings._clean_rows is policy.clean_rows
    assert target_settings._encode_targets is policy.encode_targets
    assert target_settings._resolve_target_settings is policy.resolve_target_settings


def test_model_variant_reports_native_wide_multitask_model() -> None:
    class WideModel:
        train_Y_wide = object()
        num_tasks = 2

    assert policy.model_variant(WideModel()) == (
        "wide_multitask",
        "multitask",
    )
