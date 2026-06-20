from __future__ import annotations

import numpy as np
import torch

from bochan.visualization.utils import get_model, prediction_mean_std


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _LatentModel:
    def posterior(self, X: torch.Tensor) -> _Posterior:
        mean = torch.full(X.shape[:-1] + (1,), 4.0, dtype=X.dtype)
        variance = torch.ones_like(mean)
        return _Posterior(mean, variance)


class _BinaryWrapper:
    def __init__(self) -> None:
        self.model = _LatentModel()
        self.train_X = torch.zeros(2, 1, dtype=torch.double)
        self.probability_calls = 0
        self.posterior_calls = 0

    def probability_posterior(self, X: torch.Tensor) -> _Posterior:
        self.probability_calls += 1
        mean = torch.tensor([[0.2], [0.8]], dtype=X.dtype, device=X.device)
        variance = mean * (1.0 - mean)
        return _Posterior(mean, variance)

    def posterior(self, X: torch.Tensor) -> _Posterior:
        self.posterior_calls += 1
        raise AssertionError("probability_posterior must be preferred")


class _Optimizer:
    def __init__(self, model: _BinaryWrapper) -> None:
        self.model = model
        self.train_X = model.train_X
        self.predict_calls = 0

    def predict(self, X: torch.Tensor, return_type: str):
        self.predict_calls += 1
        raise AssertionError("binary probability_posterior must be preferred")


def test_get_model_keeps_binary_wrapper_instead_of_unwrapping_latent_gp() -> None:
    model = _BinaryWrapper()

    assert get_model(model) is model
    assert get_model(model) is not model.model


def test_prediction_uses_probability_posterior_for_direct_binary_model() -> None:
    model = _BinaryWrapper()
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    mean, std = prediction_mean_std(model, X)

    assert model.probability_calls == 1
    assert model.posterior_calls == 0
    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
    np.testing.assert_allclose(std[:, 0], np.sqrt([0.16, 0.16]))
    assert np.all((0.0 <= mean) & (mean <= 1.0))


def test_prediction_uses_nested_binary_probability_posterior_before_predict() -> None:
    model = _BinaryWrapper()
    optimizer = _Optimizer(model)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    mean, _ = prediction_mean_std(optimizer, X)

    assert optimizer.predict_calls == 0
    assert model.probability_calls == 1
    np.testing.assert_allclose(mean[:, 0], [0.2, 0.8])
