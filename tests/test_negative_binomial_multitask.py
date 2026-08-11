"""Contract tests for correlated Negative Binomial multitask regression."""

import pytest
import torch

from bochan.api.configs import ModelConfig
from bochan.api.factory import resolve_model_cls
from bochan.models.regression.count.negative_binomial._components import NegativeBinomialLogLikelihood
from bochan.models.regression.count.negative_binomial.base import (
    NegativeBinomialMultiTaskGPModel,
    WideNegativeBinomialMultiTaskGPModel,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return small wide count data containing both zero and missing cells."""
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [[0.0, 2.0], [1.0, float("nan")], [2.0, 3.0],
         [0.0, 1.0], [3.0, 5.0], [1.0, 4.0]], dtype=torch.double,
    )
    return X, Y


def test_parameterization_matches_pytorch_and_poisson_limit() -> None:
    """The public mean/dispersion convention matches PyTorch exactly."""
    mu = torch.tensor([2.0, 5.0], dtype=torch.double)
    r = torch.tensor([3.0, 7.0], dtype=torch.double)
    likelihood = NegativeBinomialLogLikelihood(init_total_count=r, learn_total_count=False, link="exp")
    total_count, logits = likelihood.nb_params_from_f(mu.log())
    distribution = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)
    assert torch.allclose(distribution.mean, mu)
    assert torch.allclose(distribution.variance, mu + mu.square() / r)

    large_r = torch.full_like(r, 1e8)
    poisson_limit = mu + mu.square() / large_r
    assert torch.allclose(poisson_limit, mu, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("bad", [-1.0, 1.25, float("inf")])
def test_multitask_rejects_invalid_observed_counts(bad: float) -> None:
    """Observed targets must be finite non-negative integer-like values."""
    X, Y = _data()
    Y[0, 0] = bad
    with pytest.raises(ValueError, match="targets"):
        WideNegativeBinomialMultiTaskGPModel(X, Y)


def test_wide_multitask_posterior_sampling_and_registry() -> None:
    """Wide missing data retains task correlation and response-scale semantics."""
    X, Y = _data()
    model = WideNegativeBinomialMultiTaskGPModel(
        X, Y, rank=1, num_latents=1, num_inducing_points=6,
        init_total_count=torch.tensor([4.0, 8.0], dtype=X.dtype),
    )
    posterior = model.posterior(X[:2])
    mean_posterior = model.mean_posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 2])
    assert mean_posterior.rsample(torch.Size([3])).shape == torch.Size([3, 2, 2])
    assert torch.isfinite(posterior.mean).all() and (posterior.mean > 0).all()
    assert torch.all(posterior.variance >= mean_posterior.variance)
    assert model.dispersion().shape == torch.Size([2])
    assert torch.all(model.dispersion() > 0)
    counts = model.sample_observations(X[:2], torch.Size([4]))
    assert torch.all(counts >= 0) and torch.equal(counts, counts.round())
    assert model.observed_mask[0, 0] and not model.observed_mask[1, 1]
    assert resolve_model_cls(ModelConfig(model_type="negative_binomial_wide_multitask")) is WideNegativeBinomialMultiTaskGPModel
