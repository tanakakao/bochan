"""Regression tests for categorical values in result visualizations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bochan.visualization.data import grid_1d_plot
from bochan.visualization.utils import encode_category_value, fixed_row_from


class _CategoricalVisualizationOptimizer:
    """Small prediction object exposing the metadata used by visualization helpers."""

    def __init__(self) -> None:
        self.train_X = torch.tensor(
            [
                [0.0, 100.0],
                [1.0, 150.0],
                [0.0, 200.0],
            ],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
        self.bounds = torch.tensor(
            [[0.0, 100.0], [1.0, 200.0]],
            dtype=torch.double,
        )
        self.cat_dims = [0]
        self.labels = {"material": {"a": 0, "b": 1}}
        self.last_prediction_X: torch.Tensor | None = None

    def predict(self, X: torch.Tensor, *, return_type: str):
        assert return_type == "mean_variance"
        self.last_prediction_X = X.detach().clone()
        mean = X[:, 1:2] / 100.0 + X[:, 0:1]
        variance = torch.full_like(mean, 0.04)
        return mean, variance


def test_fixed_row_encodes_string_category_label() -> None:
    optimizer = _CategoricalVisualizationOptimizer()

    row = fixed_row_from(
        optimizer,
        feature_cols=["material", "temperature"],
        value_dict={"material": "b", "temperature": 175},
    )

    np.testing.assert_allclose(row, [[1.0, 175.0]])


def test_grid_1d_accepts_categorical_fixed_value_from_web_ui() -> None:
    optimizer = _CategoricalVisualizationOptimizer()

    mean, std, x = grid_1d_plot(
        optimizer,
        "temperature",
        {"material": "a"},
        feature_cols=["material", "temperature"],
        target_cols=["strength"],
        n=5,
    )

    assert list(mean.columns) == ["strength"]
    assert list(std.columns) == ["strength"]
    assert len(x) == 5
    assert optimizer.last_prediction_X is not None
    assert torch.all(optimizer.last_prediction_X[:, 0] == 0)


def test_categorical_axis_is_decoded_for_display() -> None:
    optimizer = _CategoricalVisualizationOptimizer()

    _, _, x = grid_1d_plot(
        optimizer,
        "material",
        {"temperature": 150},
        feature_cols=["material", "temperature"],
        target_cols=["strength"],
        n=50,
    )

    assert x.tolist() == ["a", "b"]
    assert optimizer.last_prediction_X is not None
    assert optimizer.last_prediction_X[:, 0].tolist() == [0.0, 1.0]


def test_numeric_category_from_select_string_is_encoded() -> None:
    assert encode_category_value("20", {10: 0, 20: 1}) == 1


def test_existing_encoded_category_value_is_preserved() -> None:
    assert encode_category_value(1, {"a": 0, "b": 1}) == 1


def test_unknown_category_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown categorical value"):
        encode_category_value("c", {"a": 0, "b": 1})
