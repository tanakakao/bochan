"""Contract tests for Beta regression and its correlated multitask variant."""

import pytest
import torch
from botorch.sampling.get_sampler import get_sampler

from bochan.api import ModelConfig
from bochan.api.factory import build_model, resolve_model_cls
from bochan.models.components.beta import BetaLogLikelihood, prepare_beta_targets
from bochan.models.regression.non_gaussian.beta.base import (
    BetaGPModel,
    BetaMultiTaskGPModel,
    WideBetaMultiTaskGPModel,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    X = torch.rand(7, 2, dtype=torch.double)
    Y = torch.rand(7, 3, dtype=torch.double).mul(0.8).add(0.1)
    return X, Y


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, float("inf")])
def test_beta_targets_reject_invalid_values(value: float) -> None:
    """Beta validation rejects boundaries, range errors, and infinity by default."""
    X = torch.zeros(3, 1, dtype=torch.double)
    with pytest.raises(ValueError):
        prepare_beta_targets(torch.tensor([0.2, value, 0.8]), X)


def test_beta_boundary_clip_is_explicit_and_preserves_raw_targets() -> None:
    """The fixed-epsilon clip policy changes model targets but retains raw targets."""
    X = torch.linspace(0, 1, 3, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="strictly inside"):
        BetaGPModel(X, Y)
    model = BetaGPModel(X, Y, boundary_policy="clip", boundary_epsilon=1e-4)
    assert torch.equal(model.train_targets_raw, Y)
    assert torch.all((model.train_targets_model > 0) & (model.train_targets_model < 1))


def test_beta_mean_concentration_parameterization() -> None:
    """Beta shapes reproduce the requested conditional mean and variance."""
    likelihood = BetaLogLikelihood(init_concentration=9.0, learn_concentration=False)
    latent = torch.logit(torch.tensor([0.2, 0.7], dtype=torch.double))
    concentration1, concentration0 = likelihood.beta_params_from_f(latent)
    distribution = torch.distributions.Beta(concentration1, concentration0)
    mean = torch.tensor([0.2, 0.7], dtype=torch.double)
    assert torch.allclose(distribution.mean, mean)
    assert torch.allclose(distribution.variance, mean * (1 - mean) / 10.0)


def test_beta_multitask_wide_partial_posterior_and_sampler() -> None:
    """Observed cells stay correlated and missing cells are never imputed for fitting."""
    X, Y = _data()
    Y[0, 2] = torch.nan
    Y[2, 0] = torch.nan
    model = WideBetaMultiTaskGPModel(X, Y, rank=2, num_inducing_points=5)
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 3])
    assert posterior.rsample(torch.Size([4])).shape == torch.Size([4, 2, 3])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.all(model.concentration() > 0)
    assert model.model.train_targets.numel() == torch.isfinite(Y).sum()
    assert model.task_covar_matrix.shape == torch.Size([3, 3])
    assert get_sampler(posterior, torch.Size([4])) is not None


def test_beta_multitask_registry_builds_one_correlated_model() -> None:
    """Factory routing keeps multiple Beta responses in one multitask model."""
    X, Y = _data()
    config = ModelConfig(
        task_type="regression",
        model_type="beta_multitask",
        model_kwargs={"rank": 2, "num_inducing_points": 5},
    )
    bundle = build_model(X, Y, config)
    assert isinstance(bundle.model, BetaMultiTaskGPModel)
    assert bundle.metadata["model_cls"] == "WideBetaMultiTaskGPModel"


@pytest.mark.parametrize(
    "key",
    [
        "beta_base", "beta_deepgp", "beta_deepkernel", "beta_saas",
        "beta_pca", "beta_rembo", "beta_rrp", "beta_hetero", "beta_multitask",
    ],
)
def test_all_beta_families_are_public_registry_entries(key: str) -> None:
    """Every requested Beta family resolves through the lazy public registry."""
    assert isinstance(resolve_model_cls(ModelConfig(model_type=key)), type)
