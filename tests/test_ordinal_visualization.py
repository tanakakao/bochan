from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bochan.visualization import (
    ordinal_grid_2d,
    ordinal_probabilities,
    ordinal_tri_grid,
    show_1dplot_from_optimizer,
    show_scatter_with_acqf_from_optimizer,
    show_triscatter_with_acqf_from_optimizer,
)
from bochan.visualization.plots import (
    show_1dplot_from_optimizer as plots_show_1dplot,
)


class _OrdinalModel:
    num_classes = 4

    def __init__(self, *, ternary: bool = False) -> None:
        if ternary:
            self.train_X = torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.3, 0.2],
                    [0.2, 0.5, 0.3],
                    [0.2, 0.3, 0.5],
                ],
                dtype=torch.double,
            )
        else:
            self.train_X = torch.tensor(
                [
                    [0.0, 0.0],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.6, 0.4],
                    [0.8, 0.2],
                    [1.0, 1.0],
                ],
                dtype=torch.double,
            )
        self.train_Y = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)

    def _latent(self, X: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(
            1,
            X.shape[-1] + 1,
            dtype=X.dtype,
            device=X.device,
        )
        return (X * weights).sum(dim=-1) - 1.0

    def posterior(self, X: torch.Tensor) -> SimpleNamespace:
        mean = self._latent(X).unsqueeze(-1)
        variance = torch.full_like(mean, 0.04)
        return SimpleNamespace(mean=mean, variance=variance)

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        latent = self._latent(X)
        cuts = torch.tensor([-0.5, 0.5, 1.5], dtype=X.dtype, device=X.device)
        cdf = torch.sigmoid(cuts - latent.unsqueeze(-1))
        probabilities = torch.cat(
            [
                cdf[..., :1],
                cdf[..., 1:] - cdf[..., :-1],
                1.0 - cdf[..., -1:],
            ],
            dim=-1,
        )
        return probabilities / probabilities.sum(dim=-1, keepdim=True)


class _OrdinalOptimizer:
    def __init__(self, *, ternary: bool = False) -> None:
        self.model = _OrdinalModel(ternary=ternary)
        self.train_X = self.model.train_X
        self.train_Y = self.model.train_Y
        dimension = self.train_X.shape[-1]
        self.bounds = torch.stack(
            [
                torch.zeros(dimension, dtype=torch.double),
                torch.ones(dimension, dtype=torch.double),
            ]
        )
        feature_cols = ["a", "b", "c"] if ternary else ["x0", "x1"]
        self.model_config = SimpleNamespace(task_type="ordinal")
        self.bundle = SimpleNamespace(
            model=self.model,
            task_type="ordinal",
            metadata={
                "feature_cols": feature_cols,
                "target_cols": ["level"],
                "class_labels": ["low", "mid-low", "mid-high", "high"],
            },
            cat_dims=[],
        )


def test_ordinal_probabilities_are_ordered_category_probabilities() -> None:
    optimizer = _OrdinalOptimizer()

    probabilities = ordinal_probabilities(optimizer, optimizer.train_X)

    assert probabilities.shape == (6, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert probabilities[0].argmax() < probabilities[-1].argmax()


def test_ordinal_1d_probability_display_is_default() -> None:
    optimizer = _OrdinalOptimizer()

    figure = show_1dplot_from_optimizer(
        optimizer,
        "x0",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        value_dict={"x1": 0.5},
        n=15,
    )

    assert sum(str(trace.name).startswith("P(level=") for trace in figure.data) == 4
    assert not any(trace.name == "level 予測平均" for trace in figure.data)
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]


def test_ordinal_1d_latent_display_remains_available() -> None:
    optimizer = _OrdinalOptimizer()

    figure = show_1dplot_from_optimizer(
        optimizer,
        "x0",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        value_dict={"x1": 0.5},
        n=15,
        ordinal_display="latent",
    )

    assert any(trace.name == "level 予測平均" for trace in figure.data)
    assert not any(str(trace.name).startswith("P(level=") for trace in figure.data)


def test_ordinal_1d_probability_display_draws_every_category() -> None:
    optimizer = _OrdinalOptimizer()

    figure = show_1dplot_from_optimizer(
        optimizer,
        "x0",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        value_dict={"x1": 0.5},
        n=17,
        ordinal_display="probability",
    )

    probability_lines = [
        trace.name
        for trace in figure.data
        if str(trace.name).startswith("P(level=")
    ]
    assert probability_lines == [
        "P(level=low)",
        "P(level=mid-low)",
        "P(level=mid-high)",
        "P(level=high)",
    ]
    assert list(figure.layout.yaxis.range) == [0.0, 1.0]


def test_direct_plots_import_supports_ordinal_probability_switch() -> None:
    optimizer = _OrdinalOptimizer()

    figure = plots_show_1dplot(
        optimizer,
        "x0",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        value_dict={"x1": 0.5},
        ordinal_display="probability",
        n=11,
    )

    assert sum(str(trace.name).startswith("P(level=") for trace in figure.data) == 4


def test_ordinal_grid_contains_probability_diagnostics() -> None:
    optimizer = _OrdinalOptimizer()

    data = ordinal_grid_2d(
        optimizer,
        ["x0", "x1"],
        feature_cols=["x0", "x1"],
        observed_labels=optimizer.train_Y,
        n=8,
    )

    assert data["probabilities"].shape == (8, 8, 4)
    assert data["class_index"].shape == (8, 8)
    assert data["confidence"].shape == (8, 8)
    assert data["entropy"].shape == (8, 8)
    assert data["margin"].shape == (8, 8)
    assert np.all((data["entropy"] >= 0.0) & (data["entropy"] <= 1.0))


def test_ordinal_2d_probability_display_and_uncertainty_modes() -> None:
    optimizer = _OrdinalOptimizer()

    class_figure = show_scatter_with_acqf_from_optimizer(
        optimizer,
        "x0",
        "x1",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        show_type="pred",
        ordinal_display="probability",
        ordinal_mode="class_confidence",
        n=10,
    )
    entropy_figure = show_scatter_with_acqf_from_optimizer(
        optimizer,
        "x0",
        "x1",
        "level",
        feature_cols=["x0", "x1"],
        target_cols=["level"],
        show_type="pred",
        ordinal_display="probability",
        ordinal_mode="entropy",
        n=7,
    )

    assert class_figure.data[0].type == "heatmap"
    assert any(trace.type == "contour" for trace in class_figure.data)
    assert entropy_figure.data[0].colorbar.title.text == "normalized entropy"


def test_ordinal_ternary_probability_display() -> None:
    optimizer = _OrdinalOptimizer(ternary=True)

    data = ordinal_tri_grid(
        optimizer,
        ["a", "b", "c"],
        feature_cols=["a", "b", "c"],
        observed_labels=optimizer.train_Y,
        sum_value=1.0,
        n=8,
    )
    figure = show_triscatter_with_acqf_from_optimizer(
        optimizer,
        "a",
        "b",
        "c",
        "level",
        feature_cols=["a", "b", "c"],
        target_cols=["level"],
        sum_value=1.0,
        show_type="pred",
        ordinal_display="probability",
        ordinal_mode="margin",
        n=8,
    )

    assert data["probabilities"].shape == (8 * 9 // 2, 4)
    assert figure.data[0].type == "scatterternary"
    assert figure.data[0].marker.colorbar.title.text == "top-2 probability margin"


def test_invalid_ordinal_display_is_rejected() -> None:
    optimizer = _OrdinalOptimizer()

    with pytest.raises(ValueError, match="ordinal_display"):
        show_1dplot_from_optimizer(
            optimizer,
            "x0",
            "level",
            feature_cols=["x0", "x1"],
            target_cols=["level"],
            ordinal_display="unknown",
        )
