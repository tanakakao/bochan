from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from bochan.visualization import (
    multiclass_grid_2d,
    multiclass_probabilities,
    ordinal_grid_1d,
    ordinal_probabilities,
)


class _ExpandedProbabilityModel:
    num_classes = 3

    def __init__(self, n_w: int) -> None:
        self.n_w = n_w

    def class_probs(self, X: torch.Tensor) -> torch.Tensor:
        base_logits = torch.stack(
            [
                1.0 - X[..., 0],
                X[..., 0] + X[..., 1],
                1.0 - X[..., 1],
            ],
            dim=-1,
        )
        offsets = torch.linspace(
            -0.3,
            0.3,
            self.n_w,
            dtype=X.dtype,
            device=X.device,
        )
        perturbation_logits = torch.stack(
            [offsets, torch.zeros_like(offsets), -offsets],
            dim=-1,
        )
        probabilities = torch.softmax(
            base_logits.unsqueeze(-2) + perturbation_logits,
            dim=-1,
        )
        return probabilities.reshape(-1, probabilities.shape[-1])


class _PerturbedProbabilityOptimizer:
    def __init__(self, task_type: str, n_w: int = 4) -> None:
        self.model = _ExpandedProbabilityModel(n_w=n_w)
        self.train_X = torch.tensor(
            [
                [0.0, 0.0],
                [0.3, 0.7],
                [0.7, 0.3],
                [1.0, 1.0],
            ],
            dtype=torch.double,
        )
        self.train_Y = torch.tensor([0, 1, 1, 2], dtype=torch.long)
        self.bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )
        self.model_config = SimpleNamespace(
            task_type=task_type,
            input_transform_config=SimpleNamespace(
                perturbation=True,
                n_w=n_w,
            ),
        )
        self.bundle = SimpleNamespace(
            model=self.model,
            model_config=self.model_config,
            task_type=task_type,
            metadata={
                "feature_cols": ["x0", "x1"],
                "target_cols": ["y"],
                "class_labels": ["low", "middle", "high"],
            },
            cat_dims=[],
        )


def _expected_probabilities(
    optimizer: _PerturbedProbabilityOptimizer,
    X: torch.Tensor,
) -> np.ndarray:
    raw = optimizer.model.class_probs(X)
    return (
        raw.reshape(len(X), optimizer.model.n_w, -1)
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
    )


def test_ordinal_probabilities_aggregate_perturbation_rows() -> None:
    optimizer = _PerturbedProbabilityOptimizer(task_type="ordinal", n_w=4)
    X = optimizer.train_X[:2]

    probabilities = ordinal_probabilities(optimizer, X)

    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(
        probabilities,
        _expected_probabilities(optimizer, X),
        atol=1e-12,
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_multiclass_probabilities_aggregate_perturbation_rows() -> None:
    optimizer = _PerturbedProbabilityOptimizer(task_type="multiclass", n_w=4)
    X = optimizer.train_X[:2]

    probabilities = multiclass_probabilities(optimizer, X)

    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(
        probabilities,
        _expected_probabilities(optimizer, X),
        atol=1e-12,
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_ordinal_1d_grid_keeps_one_probability_row_per_grid_point() -> None:
    optimizer = _PerturbedProbabilityOptimizer(task_type="ordinal", n_w=16)

    probability_frame, x = ordinal_grid_1d(
        optimizer,
        "x1",
        feature_cols=["x0", "x1"],
        observed_labels=optimizer.train_Y,
        n=50,
    )

    assert len(x) == 50
    assert probability_frame.shape == (50, 3)
    np.testing.assert_allclose(
        probability_frame.sum(axis=1),
        1.0,
        atol=1e-12,
    )


def test_multiclass_2d_grid_keeps_one_probability_row_per_grid_point() -> None:
    optimizer = _PerturbedProbabilityOptimizer(task_type="multiclass", n_w=4)

    data = multiclass_grid_2d(
        optimizer,
        ["x0", "x1"],
        feature_cols=["x0", "x1"],
        observed_labels=optimizer.train_Y,
        n=7,
    )

    assert data["probabilities"].shape == (7, 7, 3)
    np.testing.assert_allclose(
        data["probabilities"].sum(axis=-1),
        1.0,
        atol=1e-12,
    )
