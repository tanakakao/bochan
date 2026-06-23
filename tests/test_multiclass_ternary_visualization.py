from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import (
    multiclass_tri_grid,
    show_multiclass_triscatter_from_optimizer,
    show_triscatter_with_acqf_from_optimizer,
)
from bochan.visualization.plots import (
    show_triscatter_with_acqf_from_optimizer as plots_show_triscatter,
)


class _TernaryMulticlassModel:
    num_classes = 3

    def __init__(self) -> None:
        self.train_X = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.4, 0.3, 0.3],
                [0.2, 0.6, 0.2],
                [0.2, 0.2, 0.6],
            ],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        logits = 4.0 * X[..., :3]
        return torch.softmax(logits, dim=-1)


class _TernaryOptimizer:
    def __init__(self) -> None:
        self.model = _TernaryMulticlassModel()
        self.train_X = self.model.train_X
        self.train_Y = self.model.train_Y
        self.model_config = SimpleNamespace(task_type="multiclass")
        self.bundle = SimpleNamespace(
            model=self.model,
            task_type="multiclass",
            metadata={
                "feature_cols": ["a", "b", "c"],
                "target_cols": ["class"],
                "class_labels": ["A", "B", "C"],
            },
            cat_dims=[],
        )


def test_multiclass_tri_grid_contains_probability_diagnostics() -> None:
    optimizer = _TernaryOptimizer()

    data = multiclass_tri_grid(
        optimizer,
        ["a", "b", "c"],
        feature_cols=["a", "b", "c"],
        observed_labels=optimizer.train_Y,
        sum_value=1.0,
        n=9,
    )

    n_points = 9 * 10 // 2
    assert data["grid"].shape == (n_points, 3)
    assert data["probabilities"].shape == (n_points, 3)
    assert data["class_index"].shape == (n_points,)
    assert data["confidence"].shape == (n_points,)
    assert data["entropy"].shape == (n_points,)
    assert data["margin"].shape == (n_points,)
    np.testing.assert_allclose(data["grid"].sum(axis=1), 1.0)
    np.testing.assert_allclose(data["probabilities"].sum(axis=1), 1.0)
    np.testing.assert_allclose(
        data["confidence"],
        data["probabilities"].max(axis=-1),
    )


def test_multiclass_ternary_plot_uses_class_hue_and_confidence() -> None:
    optimizer = _TernaryOptimizer()

    figure = show_multiclass_triscatter_from_optimizer(
        optimizer,
        "a",
        "b",
        "c",
        "class",
        feature_cols=["a", "b", "c"],
        target_cols=["class"],
        sum_value=1.0,
        n=10,
    )

    predicted_traces = [
        trace for trace in figure.data if str(trace.name).startswith("predicted:")
    ]
    assert {trace.name for trace in predicted_traces} == {
        "predicted: A",
        "predicted: B",
        "predicted: C",
    }
    assert all(trace.type == "scatterternary" for trace in figure.data)
    assert any(trace.name == "low-margin boundary" for trace in figure.data)
    assert figure.layout.ternary.sum == 1


def test_existing_ternary_wrapper_dispatches_multiclass_prediction() -> None:
    optimizer = _TernaryOptimizer()

    figure = show_triscatter_with_acqf_from_optimizer(
        optimizer,
        "a",
        "b",
        "c",
        "class",
        feature_cols=["a", "b", "c"],
        target_cols=["class"],
        sum_value=1.0,
        n=8,
        show_type="pred",
    )

    assert any(str(trace.name).startswith("predicted:") for trace in figure.data)
    assert all(trace.type == "scatterternary" for trace in figure.data)


def test_direct_plots_import_dispatches_multiclass_ternary() -> None:
    optimizer = _TernaryOptimizer()

    figure = plots_show_triscatter(
        optimizer,
        "a",
        "b",
        "c",
        "class",
        feature_cols=["a", "b", "c"],
        target_cols=["class"],
        sum_value=1.0,
        n=7,
        show_type="pred",
    )

    assert any(str(trace.name).startswith("predicted:") for trace in figure.data)


def test_multiclass_ternary_entropy_mode_is_available() -> None:
    optimizer = _TernaryOptimizer()

    figure = show_triscatter_with_acqf_from_optimizer(
        optimizer,
        "a",
        "b",
        "c",
        "class",
        feature_cols=["a", "b", "c"],
        target_cols=["class"],
        sum_value=1.0,
        n=8,
        show_type="pred",
        multiclass_mode="entropy",
        boundary_margin=None,
    )

    assert figure.data[0].type == "scatterternary"
    assert figure.data[0].marker.cmin == 0.0
    assert figure.data[0].marker.cmax == 1.0
    assert figure.data[0].marker.colorbar.title.text == "normalized entropy"
