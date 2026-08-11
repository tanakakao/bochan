from __future__ import annotations

from collections.abc import Callable

import gpytorch
import pytest
import torch

from bochan.models.regression.non_gaussian.negative_binomial.deep.negative_binomial_deepgp import (
    DeepNegativeBinomialGPModel,
    DeepNegativeBinomialMixedGPModel,
)
from bochan.models.regression.non_gaussian.poisson.deep.poisson_deepgp import (
    DeepPoissonGPModel,
    DeepPoissonMixedGPModel,
)


DTYPE = torch.double


def _make_count_targets(train_x: torch.Tensor) -> torch.Tensor:
    return (1.0 + 3.0 * train_x[..., 0] + 2.0 * train_x[..., 1]).round()


def _make_continuous_model(model_cls: type) -> object:
    torch.manual_seed(0)
    train_x = torch.rand(10, 2, dtype=DTYPE)
    return model_cls(
        train_X=train_x,
        train_Y=_make_count_targets(train_x),
        hidden_dim=3,
        hidden_dims=[3],
        num_inducing=5,
    ).eval()


def _make_mixed_model(model_cls: type) -> object:
    torch.manual_seed(0)
    continuous_x = torch.rand(10, 2, dtype=DTYPE)
    categorical_x = torch.randint(0, 2, (10, 1)).to(dtype=DTYPE)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = _make_count_targets(continuous_x) + categorical_x.squeeze(-1)
    return model_cls(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        hidden_dim=3,
        num_inducing=5,
    ).eval()


CONTINUOUS_FACTORIES: list[tuple[str, Callable[[], object]]] = [
    ("poisson", lambda: _make_continuous_model(DeepPoissonGPModel)),
    ("negative_binomial", lambda: _make_continuous_model(DeepNegativeBinomialGPModel)),
]

MIXED_FACTORIES: list[tuple[str, Callable[[], object]]] = [
    ("poisson_mixed", lambda: _make_mixed_model(DeepPoissonMixedGPModel)),
    ("negative_binomial_mixed", lambda: _make_mixed_model(DeepNegativeBinomialMixedGPModel)),
]


@pytest.mark.parametrize(("case_id", "model_factory"), CONTINUOUS_FACTORIES)
@pytest.mark.parametrize("num_likelihood_samples", [1, 3, 7])
def test_count_deepgp_posterior_obeys_single_output_contract(
    case_id: str,
    model_factory: Callable[[], object],
    num_likelihood_samples: int,
) -> None:
    model = model_factory()
    model.likelihood.eval()
    test_x = torch.rand(4, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
        latent_posterior = model.latent_posterior(test_x)
        posterior = model.posterior(test_x, observation_noise=False)
        samples = posterior.rsample(sample_shape=torch.Size([5]))

    expected_shape = torch.Size([4, 1])
    assert latent_posterior.mean.shape == expected_shape, case_id
    assert latent_posterior.variance.shape == expected_shape, case_id
    assert posterior.mean.shape == expected_shape, case_id
    assert posterior.variance.shape == expected_shape, case_id
    assert posterior.event_shape == torch.Size([4, 1]), case_id
    assert samples.shape == torch.Size([5, 4, 1]), case_id
    assert torch.isfinite(posterior.mean).all(), case_id
    assert torch.isfinite(posterior.variance).all(), case_id
    assert (posterior.mean > 0).all(), case_id
    assert (posterior.variance >= 0).all(), case_id


@pytest.mark.parametrize(("case_id", "model_factory"), CONTINUOUS_FACTORIES)
def test_count_deepgp_posterior_preserves_test_batch_shape(
    case_id: str,
    model_factory: Callable[[], object],
) -> None:
    model = model_factory()
    model.likelihood.eval()
    test_x = torch.rand(2, 3, 2, dtype=DTYPE)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(4):
        posterior = model.posterior(test_x, observation_noise=True)

    assert posterior.mean.shape == torch.Size([2, 3, 1]), case_id
    assert posterior.variance.shape == torch.Size([2, 3, 1]), case_id
    assert posterior.event_shape == torch.Size([3, 1]), case_id
    assert torch.isfinite(posterior.mean).all(), case_id
    assert torch.isfinite(posterior.variance).all(), case_id


@pytest.mark.parametrize(("case_id", "model_factory"), MIXED_FACTORIES)
def test_count_mixed_deepgp_posterior_obeys_single_output_contract(
    case_id: str,
    model_factory: Callable[[], object],
) -> None:
    model = model_factory()
    model.likelihood.eval()
    continuous_x = torch.rand(2, 3, 2, dtype=DTYPE)
    categorical_x = torch.randint(0, 2, (2, 3, 1)).to(dtype=DTYPE)
    test_x = torch.cat([continuous_x, categorical_x], dim=-1)

    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(5):
        latent_posterior = model.latent_posterior(test_x)
        posterior = model.posterior(test_x, observation_noise=True)

    expected_shape = torch.Size([2, 3, 1])
    assert latent_posterior.mean.shape == expected_shape, case_id
    assert latent_posterior.variance.shape == expected_shape, case_id
    assert posterior.mean.shape == expected_shape, case_id
    assert posterior.variance.shape == expected_shape, case_id
    assert posterior.event_shape == torch.Size([3, 1]), case_id
    assert torch.isfinite(posterior.mean).all(), case_id
    assert torch.isfinite(posterior.variance).all(), case_id
    assert (posterior.mean > 0).all(), case_id
    assert (posterior.variance >= 0).all(), case_id
