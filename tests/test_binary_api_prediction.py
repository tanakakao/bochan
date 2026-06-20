from __future__ import annotations

import numpy as np
import torch

from bochan.api import BayesianOptimizer, ModelBundle, ModelConfig


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _BinaryModel:
    def __init__(self) -> None:
        self.probability_calls = 0
        self.posterior_calls = 0

    def probability_posterior(self, X: torch.Tensor, **kwargs) -> _Posterior:
        self.probability_calls += 1
        mean = torch.tensor([[0.2], [0.8]], dtype=X.dtype, device=X.device)
        variance = mean * (1.0 - mean)
        if kwargs.get("observation_noise", False) is True:
            variance = variance + 0.05
        return _Posterior(mean, variance)

    def posterior(self, X: torch.Tensor, **kwargs) -> _Posterior:
        self.posterior_calls += 1
        raise AssertionError("binary API must prefer probability_posterior")


class _RegressionModel:
    def __init__(self) -> None:
        self.posterior_calls = 0

    def posterior(self, X: torch.Tensor, **kwargs) -> _Posterior:
        self.posterior_calls += 1
        mean = X[..., :1]
        variance = torch.full_like(mean, 0.25)
        return _Posterior(mean, variance)


def make_optimizer(task_type: str, model) -> BayesianOptimizer:
    config = ModelConfig(task_type=task_type, model_type="base")
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimizer = BayesianOptimizer(model_config=config)
    optimizer.model = model
    optimizer.train_X = X
    optimizer.train_Y = Y
    optimizer.bundle = ModelBundle(
        model=model,
        train_X=X,
        train_Y=Y,
        model_config=config,
        task_type=task_type,
        model_type="base",
    )
    return optimizer


def test_binary_predict_prefers_probability_posterior() -> None:
    model = _BinaryModel()
    optimizer = make_optimizer("binary", model)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    result = optimizer.predict(X, return_result=True)

    assert model.probability_calls == 1
    assert model.posterior_calls == 0
    assert result.task_type == "binary"
    assert result.prediction_space == "probability"
    assert result.variance_kind == "bernoulli_observation"
    np.testing.assert_allclose(result.mean[:, 0], [0.2, 0.8])
    np.testing.assert_allclose(result.variance[:, 0], [0.16, 0.16])


def test_binary_predict_labels_observation_noise() -> None:
    optimizer = make_optimizer("binary", _BinaryModel())
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    result = optimizer.predict(
        X,
        return_result=True,
        posterior_kwargs={"observation_noise": True},
    )

    assert result.variance_kind == "bernoulli_observation_plus_noise"
    np.testing.assert_allclose(result.variance[:, 0], [0.21, 0.21])


def test_regression_predict_keeps_standard_posterior_path() -> None:
    model = _RegressionModel()
    optimizer = make_optimizer("regression", model)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    result = optimizer.predict(X, return_result=True)

    assert model.posterior_calls == 1
    assert result.prediction_space == "outcome"
    assert result.variance_kind == "posterior"
