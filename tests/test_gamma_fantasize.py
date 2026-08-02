"""Tests for Gamma SVGP fantasy conditioning and integrated variance."""

import pytest
import torch
from botorch.sampling import SobolQMCNormalSampler
from torch import nn

from bochan.acquisition.regression.active_learning import (
    qRegressionNegIntegratedPosteriorVariance,
)
from bochan.models.regression.non_gaussian.gamma.base import (
    GammaGPModel,
    GammaMixedGPModel,
)
from bochan.models.regression.non_gaussian.gamma.robust import gamma_heteroscedastic
from bochan.models.regression.non_gaussian.gamma.robust.gamma_heteroscedastic import (
    HeteroscedasticGammaGPModel,
    HeteroscedasticGammaMixedGPModel,
)


def _continuous_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.6, 0.9],
            [0.8, 0.2],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = 0.5 + train_X[:, :1] + 0.25 * train_X[:, 1:2]
    return train_X, train_Y


def _mixed_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 1.0],
            [0.4, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = 0.75 + train_X[:, :1] + 0.2 * train_X[:, 1:2]
    return train_X, train_Y


class _StubNoiseModel(nn.Module):
    """Minimal registered module returned instead of fitting an auxiliary GP."""

    def __init__(self, reference: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(reference.new_ones(1))


@pytest.mark.parametrize("is_mixed", [False, True])
def test_gamma_fantasize_returns_finite_batched_posterior(is_mixed):
    """Gamma continuous and mixed models expose a BoTorch-compatible fantasy model."""
    torch.manual_seed(0)
    if is_mixed:
        train_X, train_Y = _mixed_training_data()
        model = GammaMixedGPModel(
            train_X,
            train_Y,
            cat_dims=[1],
            num_inducing_points=4,
        )
        candidate = torch.tensor(
            [
                [[0.25, 0.0], [0.35, 1.0]],
                [[0.70, 1.0], [0.80, 0.0]],
            ],
            dtype=torch.double,
        )
        eval_X = torch.tensor(
            [[0.1, 0.0], [0.5, 1.0], [0.9, 0.0]],
            dtype=torch.double,
        )
    else:
        train_X, train_Y = _continuous_training_data()
        model = GammaGPModel(
            train_X,
            train_Y,
            num_inducing_points=4,
        )
        candidate = torch.tensor(
            [
                [[0.25, 0.35], [0.35, 0.45]],
                [[0.70, 0.65], [0.80, 0.75]],
            ],
            dtype=torch.double,
        )
        eval_X = torch.tensor(
            [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]],
            dtype=torch.double,
        )

    model.eval()
    model.likelihood.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]), seed=123)
    fantasy_model = model.fantasize(candidate, sampler=sampler)
    posterior = fantasy_model.posterior(eval_X)

    assert fantasy_model.batch_shape == torch.Size([2, 2])
    assert posterior.mean.shape == torch.Size([2, 2, 3, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 3, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.variance >= 0).all()


def test_gamma_nipv_uses_fantasy_model_and_is_differentiable():
    """True NIPV is selected for GammaGPModel and remains differentiable in X."""
    torch.manual_seed(0)
    train_X, train_Y = _continuous_training_data()
    model = GammaGPModel(
        train_X,
        train_Y,
        num_inducing_points=4,
    )
    model.eval()
    model.likelihood.eval()

    mc_points = torch.tensor(
        [
            [0.05, 0.10],
            [0.20, 0.80],
            [0.35, 0.30],
            [0.50, 0.50],
            [0.65, 0.90],
            [0.80, 0.20],
            [0.95, 0.85],
        ],
        dtype=torch.double,
    )
    acqf = qRegressionNegIntegratedPosteriorVariance(
        model=model,
        mc_points=mc_points,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([1]), seed=321),
    )

    assert acqf.uses_proxy is False

    candidate = torch.tensor(
        [
            [[0.25, 0.35], [0.35, 0.45]],
            [[0.70, 0.65], [0.80, 0.75]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    value = acqf(candidate)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()

    value.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_heteroscedastic_gamma_registers_noise_model_after_parent_init(monkeypatch) -> None:
    train_X, train_Y = _continuous_training_data()
    train_Yvar = torch.full_like(train_Y, 0.05)
    noise_model = _StubNoiseModel(train_X)
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_fit_noise_model_single",
        lambda **kwargs: noise_model,
    )

    model = HeteroscedasticGammaGPModel(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        num_inducing_points=3,
    )

    assert model.noise_model is noise_model
    assert model._modules["noise_model"] is noise_model
    assert "noise_model.weight" in model.state_dict()


def test_heteroscedastic_gamma_mixed_registers_noise_model_after_parent_init(monkeypatch) -> None:
    train_X, train_Y = _mixed_training_data()
    noise_model = _StubNoiseModel(train_X)
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_fit_variational_gamma_mll",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_estimate_gamma_noise_targets",
        lambda model, train_X, train_Y, **kwargs: torch.full_like(train_Y, 0.05),
    )
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_fit_noise_model_mixed",
        lambda **kwargs: noise_model,
    )

    model = HeteroscedasticGammaMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        num_inducing_points=3,
    )

    assert model.noise_model is noise_model
    assert model._modules["noise_model"] is noise_model
    assert "noise_model.weight" in model.state_dict()
