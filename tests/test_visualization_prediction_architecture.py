from __future__ import annotations

from pathlib import Path

import bochan.visualization as visualization
from bochan.visualization import input_perturbation, utils
from bochan.visualization.data import frames


def test_prediction_helper_has_single_concrete_owner() -> None:
    assert frames.prediction_mean_std is input_perturbation.prediction_mean_std
    assert input_perturbation.prediction_mean_std.__module__ == (
        "bochan.visualization.input_perturbation"
    )
    assert "prediction_mean_std" not in utils.__dict__


def test_package_root_does_not_patch_prediction_utils() -> None:
    source = Path(visualization.__file__).read_text(encoding="utf-8")

    assert "_utils.prediction_mean_std" not in source
    assert "_prediction_mean_std_with_input_perturbation" not in source
