from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import pytest

from bochan.visualization import show_triscatter_with_acqf
from bochan.visualization.ternary import (
    _normalized_ternary_coordinates,
    _ternary_coordinates,
)


def _slice_grid(total: float, levels: int = 5) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for first in range(levels + 1):
        for second in range(levels + 1 - first):
            a = first / levels
            b = second / levels
            rows.append((a * total, b * total, (1.0 - a - b) * total))
    return np.asarray(rows, dtype=float)


def test_ternary_coordinates_transpose_row_major_dataframe() -> None:
    frame = pd.DataFrame(
        {
            "Fe": [0.0, 0.2, 0.4],
            "Co": [0.4, 0.2, 0.0],
            "Ni": [0.4, 0.4, 0.4],
        }
    )

    coordinates = _ternary_coordinates(frame)

    assert coordinates.shape == (3, 3)
    assert np.allclose(coordinates[:, 0], [0.0, 0.4, 0.4])


def test_fixed_total_coordinates_are_normalized_for_plotly() -> None:
    coordinates = _slice_grid(0.8).T

    normalized, total = _normalized_ternary_coordinates(coordinates)

    assert total == pytest.approx(0.8)
    assert np.allclose(normalized.sum(axis=0), 1.0)
    assert np.allclose(normalized * total, coordinates, atol=1e-12)


def test_public_ternary_plot_accepts_row_major_fixed_total_grid() -> None:
    grid = _slice_grid(0.8)
    frame = pd.DataFrame(grid, columns=["Fe", "Co", "Ni"])
    values = np.linspace(0.0, 1.0, len(frame))
    observed = frame.iloc[::4].reset_index(drop=True)
    targets = pd.DataFrame(
        {"property": np.linspace(0.1, 0.9, len(observed))}
    )

    figure = show_triscatter_with_acqf(
        "Fe",
        "Co",
        "Ni",
        "property",
        (values, frame),
        observed,
        targets,
        show_type="pred",
    )

    contour_traces = []
    for trace in figure.data:
        if getattr(trace, "type", "") != "scatterternary":
            break
        contour_traces.append(trace)

    assert contour_traces
    for trace in contour_traces:
        if trace.a is None or trace.b is None or trace.c is None:
            continue
        totals = (
            np.asarray(trace.a, dtype=float)
            + np.asarray(trace.b, dtype=float)
            + np.asarray(trace.c, dtype=float)
        )
        assert np.allclose(totals, 0.8, atol=1e-8)


def test_public_ternary_plot_rejects_value_coordinate_length_mismatch() -> None:
    grid = _slice_grid(0.8)
    frame = pd.DataFrame(grid, columns=["Fe", "Co", "Ni"])
    observed = frame.iloc[:3].reset_index(drop=True)
    targets = pd.DataFrame({"property": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="same length"):
        show_triscatter_with_acqf(
            "Fe",
            "Co",
            "Ni",
            "property",
            (np.ones(len(frame) - 1), frame),
            observed,
            targets,
            show_type="pred",
        )


def test_web_runtime_no_longer_owns_ternary_plot_patch() -> None:
    compat_path = Path("src/bochan/serving/webapp/ternary_plot_grid_compat.py")
    runtime_path = Path("src/bochan/serving/webapp/runtime_adapters.py")

    assert not compat_path.exists()
    runtime_source = runtime_path.read_text(encoding="utf-8")
    assert "ternary_plot_grid_compat" not in runtime_source
    assert "install_ternary_plot_grid_compat" not in runtime_source


def test_web_runtime_install_does_not_replace_plotly_ternary_factory() -> None:
    original = ff.create_ternary_contour

    from bochan.serving.webapp.runtime_adapters import install_web_runtime_adapters

    install_web_runtime_adapters()

    assert ff.create_ternary_contour is original
