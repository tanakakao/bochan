from __future__ import annotations

import numpy as np
import pandas as pd

from bochan.serving.webapp.composition_multielement_ternary import (
    _ternary_slice_grid,
)
from bochan.serving.webapp.ternary_plot_grid_compat import (
    _normalized_ternary_coordinates,
    _ternary_coordinates,
    install_ternary_plot_grid_compat,
)


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


def test_ternary_slice_coordinates_are_normalized_for_plotly() -> None:
    coordinates = _ternary_slice_grid(0.8, divisions=5).T

    normalized, total = _normalized_ternary_coordinates(coordinates)

    assert total == 0.8
    assert np.allclose(normalized.sum(axis=0), 1.0)
    assert np.allclose(normalized * total, coordinates, atol=1e-12)


def test_ternary_plot_accepts_dataframe_grid_for_multielement_slice() -> None:
    install_ternary_plot_grid_compat()
    from bochan.visualization import show_triscatter_with_acqf

    grid = _ternary_slice_grid(0.8, divisions=5)
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

    ternary_traces = [
        trace
        for trace in figure.data
        if getattr(trace, "type", "") == "scatterternary"
    ]
    assert ternary_traces
    for trace in ternary_traces:
        if trace.a is None or trace.b is None or trace.c is None:
            continue
        totals = (
            np.asarray(trace.a, dtype=float)
            + np.asarray(trace.b, dtype=float)
            + np.asarray(trace.c, dtype=float)
        )
        assert np.allclose(totals, 0.8, atol=1e-8)
