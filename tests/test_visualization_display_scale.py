from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bochan.visualization.input_perturbation import prediction_mean_std


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _RegressionHybrid:
    specs = [SimpleNamespace(task_type="regression")]

    def __init__(self) -> None:
        self.output_modes: list[str] = []

    def posterior(self, X: torch.Tensor, *, output_mode: str = "objective") -> _Posterior:
        self.output_modes.append(output_mode)
        if output_mode == "mean":
            mean = X[:, :1] + 2.0
        else:
            # Simulate a target-value objective such as -abs(y - target).
            mean = -(X[:, :1] + 2.0 - 2.5).abs()
        return _Posterior(mean, torch.full_like(mean, 0.04))


class _OrdinalHybrid:
    specs = [SimpleNamespace(task_type="ordinal")]

    def posterior(self, X: torch.Tensor, *, output_mode: str = "objective") -> _Posterior:
        del X, output_mode
        # This deliberately represents an ordinal utility, not rank.
        mean = torch.tensor([[-0.4], [-0.2]], dtype=torch.double)
        variance = torch.tensor([[0.01], [0.01]], dtype=torch.double)
        return _Posterior(mean, variance)

    def class_probs_list(self, X: torch.Tensor, *, output_indices: list[int]) -> list[torch.Tensor]:
        del X
        assert output_indices == [0]
        return [
            torch.tensor(
                [
                    [0.1, 0.2, 0.7],
                    [0.6, 0.3, 0.1],
                ],
                dtype=torch.double,
            )
        ]


def _optimizer(model: object) -> SimpleNamespace:
    train_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    return SimpleNamespace(
        train_X=train_x,
        bundle=SimpleNamespace(
            model=model,
            model_config=None,
            metadata={},
        ),
        model_config=None,
    )


def test_hybrid_regression_visualization_uses_raw_mean_not_objective_value() -> None:
    model = _RegressionHybrid()
    optimizer = _optimizer(model)

    mean, std = prediction_mean_std(optimizer, optimizer.train_X)

    assert model.output_modes == ["mean"]
    np.testing.assert_allclose(mean[:, 0], [2.0, 3.0])
    np.testing.assert_allclose(std[:, 0], [0.2, 0.2])


def test_hybrid_ordinal_visualization_uses_expected_rank() -> None:
    optimizer = _optimizer(_OrdinalHybrid())

    mean, std = prediction_mean_std(optimizer, optimizer.train_X)

    expected_mean = np.array([1.6, 0.5])
    expected_var = np.array([
        0.1 * (0.0 - 1.6) ** 2 + 0.2 * (1.0 - 1.6) ** 2 + 0.7 * (2.0 - 1.6) ** 2,
        0.6 * (0.0 - 0.5) ** 2 + 0.3 * (1.0 - 0.5) ** 2 + 0.1 * (2.0 - 0.5) ** 2,
    ])
    np.testing.assert_allclose(mean[:, 0], expected_mean)
    np.testing.assert_allclose(std[:, 0], np.sqrt(expected_var))
    assert std[:, 0] == pytest.approx(np.sqrt(expected_var))
