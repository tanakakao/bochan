from __future__ import annotations

from pathlib import Path

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.api import AcquisitionConfig
from bochan.models.hybrid import HybridMultiOutputModel, OutputSpec
from bochan.serving.webapp.targets.roles import level_set_thresholds, output_spec_kwargs


def _regression_meta(
    *,
    value: float = 2.5,
    goal: str = "above",
) -> dict[str, object]:
    return {
        "target": "y",
        "internal_task": "regression",
        "goal": goal,
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


def test_regression_target_level_set_routes_user_threshold_to_zero_distance_contour() -> None:
    meta = _regression_meta(value=3.75, goal="target")
    spec = output_spec_kwargs(meta)
    thresholds = level_set_thresholds(
        target_columns=["y"],
        target_metadata={"y": meta},
        objective_targets=["y"],
    )
    config = AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0],
            "output_reduction": "weighted_mean",
        },
    )

    # The Web hybrid output becomes -|y - 3.75|, so the equivalent LSE
    # acquisition boundary is zero.  This is exactly beta*sigma - |mu - h|
    # for Straddle and the same threshold-centering used by BV / ICU.
    assert spec["eq_target"] == pytest.approx(3.75)
    assert list(thresholds) == [0.0]
    assert config.acqf_cls is not None
    assert config.acqf_cls.__name__ == "qRegressionStraddle"
    assert config.acqf_kwargs["threshold"] == pytest.approx(0.0)


def test_regression_target_level_set_actual_straddle_uses_user_threshold() -> None:
    threshold = 3.75
    meta = _regression_meta(value=threshold, goal="target")
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.tensor([[2.0], [3.0], [5.0]], dtype=torch.double)
    base_model = SingleTaskGP(train_x, train_y)
    base_model.eval()
    hybrid_model = HybridMultiOutputModel(
        [
            OutputSpec(
                name="y",
                task_type="regression",
                model=base_model,
                **output_spec_kwargs(meta),
            )
        ]
    )
    thresholds = level_set_thresholds(
        target_columns=["y"],
        target_metadata={"y": meta},
        objective_targets=["y"],
    )
    config = AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "thresholds": thresholds,
            "output_weights": [1.0],
            "output_reduction": "weighted_mean",
        },
    )
    assert config.acqf_cls is not None
    acquisition = config.acqf_cls(model=hybrid_model, **config.acqf_kwargs)

    candidate = torch.tensor([[[0.35]]], dtype=torch.double)
    posterior = base_model.posterior(candidate, observation_noise=False)
    mean = posterior.mean.squeeze(-1)
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1)
    expected = 1.96 * std - (mean - threshold).abs()
    actual = acquisition(candidate)

    assert acquisition.threshold.item() == pytest.approx(0.0)
    assert hybrid_model.specs[0].eq_target == pytest.approx(threshold)
    assert torch.allclose(actual, expected.squeeze(-1), atol=1e-8, rtol=1e-6)


def test_lse_uses_wider_untouched_candidate_distance_default() -> None:
    source = Path("web/src/context/useWorkbenchRunSettings.ts").read_text(
        encoding="utf-8"
    )

    assert "const DEFAULT_CANDIDATE_DISTANCE_RATIO = 1e-3;" in source
    assert "const DEFAULT_LSE_CANDIDATE_DISTANCE_RATIO = 1e-2;" in source
    assert "minimumCandidateDistanceTouched" in source
    assert 'nextFamily === "level_set_estimation"' in source
    assert "restored.minimumCandidateDistanceRatio" in source


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
