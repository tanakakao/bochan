"""Contract tests for correlated Poisson multitask regression."""

from __future__ import annotations

import pytest
import torch

from bochan.api import ModelConfig
from bochan.api.factory import build_model, resolve_model_cls
from bochan.models.regression.non_gaussian.poisson.base import (
    PoissonMultiTaskGPModel,
    WidePoissonMultiTaskGPModel,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return small wide count data containing both zero and a missing cell."""
    train_X = torch.linspace(0, 1, 5, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(
        [[0, 1], [1, 2], [0, float("nan")], [2, 3], [1, 2]],
        dtype=torch.double,
    )
    return train_X, train_Y


def test_wide_poisson_multitask_posterior_contract() -> None:
    """Preserve q/task axes, task covariance, and sampling scale semantics."""
    train_X, train_Y = _data()
    model = WidePoissonMultiTaskGPModel(
        train_X, train_Y, rank=1, num_latents=1, num_inducing_points=4
    )

    latent = model.latent_posterior(train_X[:2])
    rate = model.rate_posterior(train_X[:2])
    predictive = model.posterior(train_X[:2])

    assert latent.mean.numel() == 4
    assert rate.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(rate.mean).all() and (rate.mean > 0).all()
    assert torch.all(predictive.variance >= predictive.mean)
    assert rate.rsample(torch.Size([3])).shape == torch.Size([3, 2, 2])
    counts = model.sample_observations(train_X[:2], torch.Size([3]))
    assert counts.shape == torch.Size([3, 2, 2])
    assert (counts >= 0).all() and torch.equal(counts, counts.round())
    assert model.task_covar_matrix.shape == torch.Size([2, 2])
    assert torch.isfinite(model.task_covar_matrix).all()
    assert model.observed_mask[0, 0] and not model.observed_mask[2, 1]


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [(-1.0, "non-negative"), (1.5, "integer counts"), (float("inf"), "finite")],
)
def test_poisson_multitask_rejects_invalid_observed_counts(
    bad_value: float, message: str
) -> None:
    """Reject negative, fractional, and infinite observed counts without rounding."""
    train_X, train_Y = _data()
    train_Y[0, 0] = bad_value
    with pytest.raises(ValueError, match=message):
        WidePoissonMultiTaskGPModel(train_X, train_Y)


def test_poisson_registry_and_factory_build_correlated_model() -> None:
    """Resolve the public key directly instead of splitting outputs into ModelList."""
    train_X, train_Y = _data()
    config = ModelConfig(
        model_type="poisson_wide_multitask",
        model_kwargs={"rank": 1, "num_inducing_points": 4},
    )
    assert config.outcome_transform is None
    assert resolve_model_cls(config) is WidePoissonMultiTaskGPModel
    bundle = build_model(train_X, train_Y, config)
    assert isinstance(bundle.model, WidePoissonMultiTaskGPModel)
    assert bundle.metadata["model_cls"] == "WidePoissonMultiTaskGPModel"
