from __future__ import annotations

import gpytorch
import pytest
import torch

from bochan.models.regression.non_gaussian.beta.deep.beta_deepgp import (
    BetaDeepGPModel,
    BetaMixedDeepGPModel,
)


DTYPE = torch.double


def _make_beta_targets(train_x: torch.Tensor) -> torch.Tensor:
    latent = 1.5 * train_x[..., 0] - 0.75 * train_x[..., 1]
    return 0.1 + 0.8 * torch.sigmoid(latent)


def _make_continuous_model() -> BetaDeepGPModel:
    torch.manual_seed(0)
    train_x = torch.rand(10, 2, dtype=DTYPE)
    model = BetaDeepGPModel(
        train_X=train_x,
        train_Y=_make_beta_targets(train_x),
        hidden_dim=3,
        list_hidden_dims=[3],
        num_inducing=5,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _make_mixed_model() -> BetaMixedDeepGPModel:
    torch.manual_seed(0)
    continuous_x = torch.rand(10, 2, dtype=DTYPE)
    categorical_x = torch.randint(0, 2, (10, 1)).to(dtype=DTYPE)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = (
        _make_beta_targets(continuous_x)
        + 0.05 * (categorical_x.squeeze(-1) - 0.5)
    ).clamp(0.05, 0.95)
    model = BetaMixedDeepGPModel(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        hidden_dim=3,
        num_inducing=5,
    )
    model.eval()
    model.likelihood.eval()
    return model


@pytest.mark.parametrize("num_likelihood_samples", [1, 3, 7])
def test_beta_deepgp_posterior_obeys_single_output_contract(
    num_likelihood_samples: int,
) -> None:
    model = _make_continuous_model()
    test_x = torch.rand(4, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(
        num_likelihood_samples
    ):
        latent_posterior = model.latent_posterior(test_x)
        posterior = model.posterior(test_x, observation_noise=False)
        samples = posterior.rsample(sample_shape=torch.Size([5]))

    expected_shape = torch.Size([4, 1])
    assert latent_posterior.mean.shape == expected_shape
    assert latent_posterior.variance.shape == expected_shape
    assert posterior.mean.shape == expected_shape
    assert posterior.variance.shape == expected_shape
    assert posterior.event_shape == torch.Size([4, 1])
    assert samples.shape == torch.Size([5, 4, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert ((posterior.mean > 0) & (posterior.mean < 1)).all()
    assert (posterior.variance >= 0).all()


def test_beta_deepgp_posterior_preserves_test_batch_shape() -> None:
    model = _make_continuous_model()
    test_x = torch.rand(2, 3, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(4):
        posterior = model.posterior(test_x, observation_noise=True)

    assert posterior.mean.shape == torch.Size([2, 3, 1])
    assert posterior.variance.shape == torch.Size([2, 3, 1])
    assert posterior.event_shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_beta_deepgp_grid_posterior_can_reshape_for_prediction_plot() -> None:
    model = _make_continuous_model()
    n_grid = 50
    test_x = torch.rand(n_grid * n_grid, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
        posterior = model.posterior(test_x)
        mean_grid = posterior.mean[..., 0].reshape(n_grid, n_grid)
        variance_grid = posterior.variance[..., 0].reshape(n_grid, n_grid)

    assert mean_grid.shape == torch.Size([50, 50])
    assert variance_grid.shape == torch.Size([50, 50])
    assert torch.isfinite(mean_grid).all()
    assert torch.isfinite(variance_grid).all()


def test_beta_mixed_deepgp_posterior_obeys_single_output_contract() -> None:
    model = _make_mixed_model()
    continuous_x = torch.rand(2, 3, 2, dtype=DTYPE)
    categorical_x = torch.randint(0, 2, (2, 3, 1)).to(dtype=DTYPE)
    test_x = torch.cat([continuous_x, categorical_x], dim=-1)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(5):
        latent_posterior = model.latent_posterior(test_x)
        posterior = model.posterior(test_x, observation_noise=True)

    expected_shape = torch.Size([2, 3, 1])
    assert latent_posterior.mean.shape == expected_shape
    assert latent_posterior.variance.shape == expected_shape
    assert posterior.mean.shape == expected_shape
    assert posterior.variance.shape == expected_shape
    assert posterior.event_shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert ((posterior.mean > 0) & (posterior.mean < 1)).all()
    assert (posterior.variance >= 0).all()
