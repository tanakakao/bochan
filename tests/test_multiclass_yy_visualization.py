from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import show_yyplot_from_optimizer
from bochan.visualization.plots import (
    show_yyplot_from_optimizer as plots_show_yyplot_from_optimizer,
)


class _MulticlassModel:
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
        logits = torch.stack(
            [
                2.0 * (1.0 - X[..., 0]),
                1.5 - 3.0 * torch.abs(X[..., 0] - X[..., 1]),
                2.0 * X[..., 0],
            ],
            dim=-1,
        )
        return torch.softmax(logits, dim=-1)


class _MulticlassOptimizer:
    def __init__(self) -> None:
        self.model = _MulticlassModel()
        self.train_X = self.model.train_X
        self.train_Y = self.model.train_Y
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


def _expected_correct_probabilities(optimizer: _MulticlassOptimizer) -> np.ndarray:
    probabilities = (
        optimizer.model.class_probs(optimizer.train_X).detach().cpu().numpy()
    )
    labels = optimizer.train_Y.detach().cpu().numpy()
    return probabilities[np.arange(len(labels)), labels]


def test_multiclass_yyplot_uses_probability_of_observed_label() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )

    marker_traces = [trace for trace in figure.data if trace.mode == "markers"]
    plotted = np.concatenate([np.asarray(trace.y, dtype=float) for trace in marker_traces])
    expected = _expected_correct_probabilities(optimizer)

    np.testing.assert_allclose(np.sort(plotted), np.sort(expected), atol=1e-12)
    assert [trace.name for trace in marker_traces] == ["alpha", "beta", "gamma"]
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]
    assert figure.layout.xaxis.title.text == "正解ラベル"
    assert figure.layout.yaxis.title.text == "正解ラベルに対する予測確率"


def test_multiclass_yyplot_hover_contains_predicted_label_and_confidence() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )

    hovertemplate = figure.data[0].hovertemplate
    assert "予測ラベル" in hovertemplate
    assert "最大確率" in hovertemplate
    assert len(figure.layout.shapes) == 1
    assert np.isclose(figure.layout.shapes[0].y0, 1.0 / 3.0)


def test_direct_plots_import_uses_multiclass_yy_dispatch() -> None:
    optimizer = _MulticlassOptimizer()

    figure = plots_show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )

    assert [trace.name for trace in figure.data if trace.mode == "markers"] == [
        "alpha",
        "beta",
        "gamma",
    ]
