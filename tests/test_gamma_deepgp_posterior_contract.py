from __future__ import annotations

import gpytorch
import pytest
import torch
from gpytorch.distributions import MultivariateNormal

from bochan.models.regression.non_gaussian.gamma.deep.gamma_deepgp import (
    DeepGammaGPModel,
    DeepGammaMixedGPModel,
)


DTYPE = torch.double


def _make_gamma_targets(train_x: torch.Tensor) -> torch.Tensor:
    return 0.2 + train_x[..., 0].square() + 0.5 * train_x[..., 1]


def _make_continuous_model() -> DeepGammaGPModel:
    torch.manual_seed(0)
    train_x = torch.rand(10, 2, dtype=DTYPE)
    model = DeepGammaGPModel(
        train_X=train_x,
        train_Y=_make_gamma_targets(train_x),
        hidden_dim=3,
        hidden_dims=[3],
        num_inducing=5,
    )
    model.eval()
    model.likelihood.eval()
    return model


def _make_mixed_model() -> DeepGammaMixedGPModel:
    torch.manual_seed(0)
    continuous_x = torch.rand(10, 2, dtype=DTYPE)
    categorical_x = torch.randint(0, 2, (10, 1)).to(dtype=DTYPE)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = _make_gamma_targets(continuous_x) + 0.1 * categorical_x.squeeze(-1)
    model = DeepGammaMixedGPModel(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        hidden_dim=3,
        num_inducing=5,
    )
    model.eval()
    model.likelihood.eval()
    return model


def test_moment_matching_preserves_within_and_between_component_variance() -> None:
    component_mean = torch.tensor(
        [[-1.0, 0.0], [1.0, 2.0]],
        dtype=DTYPE,
    )
    component_covar = torch.eye(2, dtype=DTYPE).expand(2, 2, 2).clone()
    dist = MultivariateNormal(component_mean, component_covar)
    test_x = torch.zeros(2, 1, dtype=DTYPE)

    matched = DeepGammaGPModel._moment_match_deepgp_distribution(dist, X=test_x)

    expected_mean = torch.tensor([0.0, 1.0], dtype=DTYPE)
    expected_covar = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0]],
        dtype=DTYPE,
    )
    assert matched.mean.shape == torch.Size([2])
    assert matched.covariance_matrix.shape == torch.Size([2, 2])
    assert torch.allclose(matched.mean, expected_mean)
    assert torch.allclose(matched.covariance_matrix, expected_covar)


@pytest.mark.parametrize("num_likelihood_samples", [1, 3, 7])
def test_gamma_deepgp_posterior_obeys_single_output_contract(
    num_likelihood_samples: int,
) -> None:
    model = _make_continuous_model()
    test_x = torch.rand(4, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
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
    assert (posterior.mean > 0).all()
    assert (posterior.variance >= 0).all()


def test_gamma_deepgp_posterior_preserves_test_batch_shape() -> None:
    model = _make_continuous_model()
    test_x = torch.rand(2, 3, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(4):
        posterior = model.posterior(test_x, observation_noise=False)

    assert posterior.mean.shape == torch.Size([2, 3, 1])
    assert posterior.variance.shape == torch.Size([2, 3, 1])
    assert posterior.event_shape == torch.Size([3, 1])


def test_gamma_mixed_deepgp_posterior_obeys_single_output_contract() -> None:
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
    assert (posterior.mean > 0).all()
    assert (posterior.variance >= 0).all()
