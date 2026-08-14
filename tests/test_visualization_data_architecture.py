from __future__ import annotations

from pathlib import Path

import bochan.visualization as visualization
import bochan.visualization.data as data
from bochan.visualization.data import frames, grids, study, ternary


def test_visualization_data_is_a_responsibility_package() -> None:
    package_dir = Path(data.__file__).resolve().parent

    assert hasattr(data, "__path__")
    assert package_dir.name == "data"
    assert not (package_dir.parent / "data.py").exists()


def test_visualization_data_functions_have_concrete_owners() -> None:
    assert data.prediction_dataframe is frames.prediction_dataframe
    assert data.candidates_dataframe is frames.candidates_dataframe
    assert data.grid_1d_plot is grids.grid_1d_plot
    assert data.grid_2d is grids.grid_2d
    assert data.tri_grid is ternary.tri_grid
    assert data.study_target_dataframe is study.study_target_dataframe

    assert frames.prediction_dataframe.__module__ == (
        "bochan.visualization.data.frames"
    )
    assert grids.grid_2d.__module__ == "bochan.visualization.data.grids"
    assert ternary.tri_grid.__module__ == "bochan.visualization.data.ternary"
    assert study.study_target_dataframe.__module__ == (
        "bochan.visualization.data.study"
    )


def test_prediction_data_uses_canonical_perturbation_aware_prediction() -> None:
    assert frames.prediction_mean_std.__module__ == (
        "bochan.visualization.input_perturbation"
    )
    assert visualization.prediction_dataframe is data.prediction_dataframe
    assert "prediction_mean_std" not in data.__dict__
