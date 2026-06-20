from __future__ import annotations

import numpy as np
import torch

from bochan.visualization.utils import get_model, prediction_mean_std
from tests.test_binary_epistemic_uncertainty import _EpistemicBinaryModel


class _Optimizer:
    def __init__(self, model: _EpistemicBinaryModel) -> None:
        self.model = model
        self.train_X = model.train_inputs[0]
        self.predict_calls = 0

    def predict(self, X: torch.Tensor, return_type: str):
        self.predict_calls += 1
        raise AssertionError("binary epistemic visualization must use the model")


def test_get_model_keeps_binary_wrapper_instead_of_unwrapping_latent_gp() -> None:
    model = _EpistemicBinaryModel()

    assert get_model(model) is model


def test_binary_visualization_defaults_to_probability_epistemic_std() -> None:
    model = _EpistemicBinaryModel()
    X = torch.tensor([[0.2, 0.02], [0.8, 0.02]], dtype=torch.double)

    mean, epistemic_std = prediction_mean_std(
        model,
        X,
        num_uncertainty_samples=1001,
    )
    _, observation_std = prediction_mean_std(
        model,
        X,
        uncertainty_kind="observation",
        num_uncertainty_samples=1001,
    )

    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
    assert np.all(epistemic_std[:, 0] < 0.03)
    np.testing.assert_allclose(
        observation_std[:, 0],
        np.sqrt([0.16, 0.16]),
        atol=1e-8,
    )
    assert np.all(observation_std > epistemic_std)


def test_nested_optimizer_uses_binary_epistemic_model_before_predict() -> None:
    model = _EpistemicBinaryModel()
    optimizer = _Optimizer(model)
    X = torch.tensor([[0.2, 0.02], [0.8, 0.02]], dtype=torch.double)

    mean, std = prediction_mean_std(
        optimizer,
        X,
        num_uncertainty_samples=257,
    )

    assert optimizer.predict_calls == 0
    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
    assert np.all(std[:, 0] < 0.03)
