from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import (
    multiclass_grid_2d,
    multiclass_prediction_dataframe,
    multiclass_probabilities,
    show_1dplot_from_optimizer,
    show_scatter_with_acqf_from_optimizer,
)
from bochan.visualization.plots import (
    show_1dplot_from_optimizer as plots_show_1dplot_from_optimizer,
)


class _MulticlassModel:
    num_classes = 3

    def __init__(self) -> None:
        self.train_X = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.8, 0.2],
                [1.0, 1.0],
            ],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        logits = torch.stack(
            [
                2.0 * (1.0 - x0),
                1.5 - 3.0 * torch.abs(x0 - x1),
                2.0 * x0,
            ],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)

    def posterior(self, X: torch.Tensor) -> SimpleNamespace:
        probabilities = self.class_probs(X)
        return SimpleNamespace(
            mean=probabilities,
            variance=probabilities * (1.0 - probabilities),
        )


class _MulticlassOptimizer:
    def __init__(self) -> None:
        self.model = _MulticlassModel()
        self.train_X = self.model.train_X
        self.train_Y = self.model.train_Y
        self.bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )
        self.model_config = SimpleNamespace(task_type="multiclass")
        self.bundle = SimpleNamespace(
            model=self.model,
            task_type="multiclass",
            metadata={
                "feature_cols": ["x0", "x1"],
                "target_cols": ["class"],
                "class_labels": ["alpha", "beta", "gamma"],
            },
            cat_dims=[],
        )


class _MultiOutputMulticlassModel:
    def class_probs_list(self, X: torch.Tensor) -> list[torch.Tensor]:
        first = torch.softmax(
            torch.stack([X[..., 0], 1.0 - X[..., 0], X[..., 1]], dim=-1),
            dim=-1,
        )
        second = torch.softmax(
            torch.stack([X[..., 1], 1.0 - X[..., 1]], dim=-1),
            dim=-1,
        )
        return [first, second]


def test_multiclass_prediction_dataframe_has_one_probability_per_class() -> None:
    optimizer = _MulticlassOptimizer()

    frame = multiclass_prediction_dataframe(
        optimizer,
        optimizer.train_X,
        observed_labels=optimizer.train_Y,
    )

    assert list(frame.columns) == ["alpha", "beta", "gamma"]
    assert frame.shape == (5, 3)
    np.testing.assert_allclose(frame.sum(axis=1), 1.0, atol=1e-12)


def test_multiclass_probabilities_select_multi_output() -> None:
    model = _MultiOutputMulticlassModel()
    X = torch.tensor([[0.1, 0.9], [0.7, 0.2]], dtype=torch.double)

    probabilities = multiclass_probabilities(model, X, output_index=1)

    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_multiclass_line_plot_draws_every_class_probability() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_1dplot_from_optimizer(
        optimizer,
        feature="x0",
        target="class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
        value_dict={"x1": 0.5},
        n=21,
    )

    line_names = [
        trace.name
        for trace in figure.data
        if getattr(trace, "mode", None) == "lines"
    ]
    assert line_names == [
        "P(class=alpha)",
        "P(class=beta)",
        "P(class=gamma)",
    ]
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]


def test_direct_plots_import_uses_multiclass_dispatch() -> None:
    optimizer = _MulticlassOptimizer()

    figure = plots_show_1dplot_from_optimizer(
        optimizer,
        feature="x0",
        target="class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
        value_dict={"x1": 0.5},
        n=11,
    )

    assert sum(getattr(trace, "mode", None) == "lines" for trace in figure.data) == 3


def test_multiclass_grid_contains_class_confidence_entropy_and_margin() -> None:
    optimizer = _MulticlassOptimizer()

    data = multiclass_grid_2d(
        optimizer,
        ["x0", "x1"],
        feature_cols=["x0", "x1"],
        observed_labels=optimizer.train_Y,
        n=9,
    )

    assert data["class_index"].shape == (9, 9)
    assert data["confidence"].shape == (9, 9)
    assert data["entropy"].shape == (9, 9)
    assert data["margin"].shape == (9, 9)
    assert data["probabilities"].shape == (9, 9, 3)
    np.testing.assert_allclose(
        data["confidence"],
        data["probabilities"].max(axis=-1),
    )
    assert np.all((data["entropy"] >= 0.0) & (data["entropy"] <= 1.0))
    assert np.all((data["margin"] >= 0.0) & (data["margin"] <= 1.0))


def test_multiclass_prediction_heatmap_uses_class_hue_and_confidence() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_scatter_with_acqf_from_optimizer(
        optimizer,
        "x0",
        "x1",
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
        n=12,
        show_type="pred",
    )

    assert figure.data[0].type == "heatmap"
    assert figure.data[0].customdata.shape == (12, 12, 4)
    assert any(trace.type == "contour" for trace in figure.data)
    assert "predicted class" in figure.data[0].hovertemplate


def test_multiclass_entropy_heatmap_is_available() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_scatter_with_acqf_from_optimizer(
        optimizer,
        "x0",
        "x1",
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
        n=8,
        show_type="pred",
        multiclass_mode="entropy",
    )

    assert figure.data[0].type == "heatmap"
    assert figure.data[0].zmin == 0.0
    assert figure.data[0].zmax == 1.0
    assert figure.data[0].colorbar.title.text == "normalized entropy"
