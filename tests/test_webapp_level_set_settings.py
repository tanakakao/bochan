from __future__ import annotations

import pytest

from bochan.api import AcquisitionConfig
from bochan.serving.webapp.target_roles import level_set_thresholds, output_spec_kwargs


def _regression_meta(*, value: float = 2.5) -> dict[str, object]:
    return {
        "target": "y",
        "internal_task": "regression",
        "goal": "above",
        "configured_value": value,
        "direction": "maximize",
        "class_index": None,
        "class_indices": [],
        "num_classes": None,
    }


def _multiclass_meta(*, threshold: float = 0.7) -> dict[str, object]:
    return {
        "target": "class_y",
        "internal_task": "multiclass",
        "goal": "above",
        "configured_value": threshold,
        "direction": "maximize",
        "class_index": 0,
        "class_indices": [0, 2],
        "num_classes": 3,
    }


def _ordinal_meta(*, goal: str = "above") -> dict[str, object]:
    return {
        "target": "rank_y",
        "internal_task": "ordinal",
        "goal": goal,
        "configured_value": "medium" if goal != "target" else ["medium"],
        "direction": "maximize",
        "class_index": 1,
        "class_indices": [1],
        "num_classes": 3,
    }


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("straddle", "qRegressionStraddle"),
        ("boundary_variance", "qRegressionBoundaryVariance"),
        ("ICU", "qRegressionICU"),
    ],
)
def test_single_output_web_level_set_uses_hybrid_objective_acquisition(
    name: str,
    expected_cls: str,
) -> None:
    thresholds = level_set_thresholds(
        target_columns=["y"],
        target_metadata={"y": _regression_meta()},
        objective_targets=["y"],
    )

    config = AcquisitionConfig(
        name=name,
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0],
            "output_reduction": "weighted_mean",
        },
    )

    assert config.acqf_cls is not None
    assert config.acqf_cls.__name__ == expected_cls
    assert config.acqf_kwargs["threshold"] == pytest.approx(2.5)
    assert "thresholds" not in config.acqf_kwargs
    assert "output_weights" not in config.acqf_kwargs
    assert "output_reduction" not in config.acqf_kwargs


def test_multiclass_level_set_uses_selected_class_probability_objective() -> None:
    meta = _multiclass_meta(threshold=0.65)
    spec = output_spec_kwargs(meta)
    thresholds = level_set_thresholds(
        target_columns=["class_y"],
        target_metadata={"class_y": meta},
        objective_targets=["class_y"],
    )
    config = AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0],
            "output_reduction": "weighted_mean",
        },
    )

    assert spec["utility_values"] == [1.0, 0.0, 1.0]
    assert config.acqf_cls is not None
    assert config.acqf_cls.__name__ == "qRegressionStraddle"
    assert config.acqf_kwargs["threshold"] == pytest.approx(0.65)


def test_ordinal_level_set_uses_rank_objective_boundary() -> None:
    meta = _ordinal_meta(goal="above")
    spec = output_spec_kwargs(meta)
    thresholds = level_set_thresholds(
        target_columns=["rank_y"],
        target_metadata={"rank_y": meta},
        objective_targets=["rank_y"],
    )
    config = AcquisitionConfig(
        name="boundary_variance",
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0],
            "output_reduction": "weighted_mean",
        },
    )

    assert spec["utility_values"] == [0, 1, 2]
    assert config.acqf_cls is not None
    assert config.acqf_cls.__name__ == "qRegressionBoundaryVariance"
    assert config.acqf_kwargs["threshold"] == pytest.approx(1.0)


def test_ordinal_target_level_set_uses_zero_distance_contour() -> None:
    meta = _ordinal_meta(goal="target")
    spec = output_spec_kwargs(meta)
    thresholds = level_set_thresholds(
        target_columns=["rank_y"],
        target_metadata={"rank_y": meta},
        objective_targets=["rank_y"],
    )

    assert spec["utility_values"] == [-1, 0, -1]
    assert list(thresholds) == [0.0]


def test_multi_output_web_level_set_keeps_per_output_thresholds_and_weights() -> None:
    multiclass = _multiclass_meta(threshold=0.6)
    ordinal = _ordinal_meta(goal="above")
    thresholds = level_set_thresholds(
        target_columns=["class_y", "rank_y"],
        target_metadata={"class_y": multiclass, "rank_y": ordinal},
        objective_targets=["class_y", "rank_y"],
    )
    config = AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0, 1.0],
            "output_reduction": "weighted_mean",
        },
    )

    assert config.acqf_cls is not None
    assert config.acqf_cls.__name__ == "qMultiOutputRegressionStraddle"
    assert config.acqf_kwargs["thresholds"] == pytest.approx([0.6, 1.0])
    assert config.acqf_kwargs["output_weights"] == [1.0, 1.0]
    assert config.acqf_kwargs["output_reduction"] == "weighted_mean"
