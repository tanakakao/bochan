from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import show_yyplot_from_optimizer
from bochan.visualization.plots import (
    show_yyplot_from_optimizer as plots_show_yyplot_from_optimizer,
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


def test_multiclass_yyplot_draws_every_probability_against_true_label() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )
    expected = optimizer.model.class_probs(optimizer.train_X).detach().numpy()

    assert [trace.name for trace in figure.data] == [
        "P(class=alpha)",
        "P(class=beta)",
        "P(class=gamma)",
    ]
    for class_index, trace in enumerate(figure.data):
        np.testing.assert_allclose(trace.y, expected[:, class_index])

    assert list(figure.layout.xaxis.ticktext) == ["alpha", "beta", "gamma"]
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]
    assert figure.layout.xaxis.title.text == "正解ラベル"
    assert figure.layout.yaxis.title.text == "予測確率"


def test_multiclass_yyplot_x_position_tracks_observed_label() -> None:
    optimizer = _MulticlassOptimizer()

    figure = show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )

    observed = optimizer.train_Y.detach().numpy()
    offsets = np.linspace(-0.3, 0.3, 3)
    for class_index, trace in enumerate(figure.data):
        np.testing.assert_allclose(trace.x, observed + offsets[class_index])


def test_direct_plots_import_uses_multiclass_yyplot_dispatch() -> None:
    optimizer = _MulticlassOptimizer()

    figure = plots_show_yyplot_from_optimizer(
        optimizer,
        "class",
        feature_cols=["x0", "x1"],
        target_cols=["class"],
    )

    assert len(figure.data) == 3
    assert all(str(trace.name).startswith("P(class=") for trace in figure.data)
    assert "正解ラベル" in figure.data[0].hovertemplate
