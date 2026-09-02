from __future__ import annotations

from pathlib import Path

import pytest
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from linear_operator import to_linear_operator
from torch import nn

from bochan.models.regression.gaussian.deep.deepkernel_configurable import (
    DeepKernelGaussianGPModel,
)
from bochan.models.regression.gaussian.deep.multitask_fixed_noise import (
    MultitaskFixedNoiseGaussianLikelihood,
)


def _generic_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0, 0.0], [0.3, 0.2], [0.6, 0.8], [1.0, 1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [[0.0, 1.0], [0.4, 1.3], [0.7, 1.8], [1.1, 2.2]],
        dtype=torch.double,
    )
    Yvar = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    return X, Y, Yvar


def test_multitask_fixed_noise_uses_interleaved_event_order() -> None:
    noise = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04]],
        dtype=torch.double,
    )
    likelihood = MultitaskFixedNoiseGaussianLikelihood(noise)

    diagonal = likelihood._shaped_noise_covar(torch.Size([2, 2])).diagonal()
    torch.testing.assert_close(
        diagonal,
        torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.double),
    )
    torch.testing.assert_close(likelihood.task_noise, noise)


def test_multitask_fixed_noise_matches_multitask_distribution_covariance() -> None:
    mean = torch.zeros(2, 2, dtype=torch.double)
    covariance = to_linear_operator(torch.eye(4, dtype=torch.double))
    latent = MultitaskMultivariateNormal(mean, covariance, interleaved=True)
    noise = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04]],
        dtype=torch.double,
    )
    observed = MultitaskFixedNoiseGaussianLikelihood(noise)(latent)

    torch.testing.assert_close(
        observed.lazy_covariance_matrix.diagonal()
        - latent.lazy_covariance_matrix.diagonal(),
        noise.reshape(-1),
    )


def test_multitask_fixed_noise_uses_noninterleaved_event_order() -> None:
    mean = torch.zeros(2, 2, dtype=torch.double)
    covariance = to_linear_operator(torch.eye(4, dtype=torch.double))
    latent = MultitaskMultivariateNormal(mean, covariance, interleaved=False)
    noise = torch.tensor(
        [[0.01, 0.02], [0.03, 0.04]],
        dtype=torch.double,
    )
    observed = MultitaskFixedNoiseGaussianLikelihood(noise)(latent)

    assert observed._interleaved is False
    torch.testing.assert_close(
        observed.lazy_covariance_matrix.diagonal()
        - latent.lazy_covariance_matrix.diagonal(),
        torch.tensor([0.01, 0.03, 0.02, 0.04], dtype=torch.double),
    )


def test_correlated_deepkernel_accepts_wide_train_yvar_and_exact_mll() -> None:
    X, Y, Yvar = _generic_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
        outcome_transform=None,
    ).double()

    assert isinstance(model.likelihood, MultitaskFixedNoiseGaussianLikelihood)
    torch.testing.assert_close(model.likelihood.task_noise, Yvar)

    model.train()
    output = model.deepkernel(model.transform_inputs(X))
    mll_value = model.make_mll()(output, Y)
    assert torch.isfinite(mll_value)
    (-mll_value).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.eval()
    latent = model.posterior(X[:2], observation_noise=False)
    test_noise = torch.tensor(
        [[0.05, 0.06], [0.07, 0.08]],
        dtype=torch.double,
    )
    observed = model.posterior(X[:2], observation_noise=test_noise)
    assert torch.isfinite(observed.mean).all()
    assert torch.isfinite(observed.variance).all()
    torch.testing.assert_close(
        observed.variance - latent.variance,
        test_noise,
        rtol=1e-4,
        atol=1e-6,
    )


def test_default_outcome_transform_scales_wide_variance_before_likelihood() -> None:
    X, Y, Yvar = _generic_data()
    model = DeepKernelGaussianGPModel(
        X,
        Y,
        Yvar,
        feature_extractor=nn.Identity(),
        latent_dim=2,
        input_transform=None,
    ).double()

    assert model.train_Yvar is not None
    assert model.train_Yvar.shape == Yvar.shape
    assert isinstance(model.likelihood, MultitaskFixedNoiseGaussianLikelihood)
    torch.testing.assert_close(model.likelihood.task_noise, model.train_Yvar)
    assert torch.isfinite(model.train_Yvar).all()


def test_multitask_fixed_noise_validates_task_shape_and_values() -> None:
    with pytest.raises(ValueError, match="at least two task"):
        MultitaskFixedNoiseGaussianLikelihood(torch.ones(3, 1))
    with pytest.raises(ValueError, match="strictly positive"):
        MultitaskFixedNoiseGaussianLikelihood(
            torch.tensor([[0.01, -0.02], [0.03, 0.04]])
        )
    with pytest.raises(ValueError, match="strictly positive"):
        MultitaskFixedNoiseGaussianLikelihood(
            torch.tensor([[0.01, 0.0], [0.03, 0.04]])
        )

    likelihood = MultitaskFixedNoiseGaussianLikelihood(torch.full((3, 2), 0.01))
    with pytest.raises(ValueError, match="event shape"):
        likelihood._shaped_noise_covar(torch.Size([2, 3]))
    with pytest.raises(ValueError, match="match the requested"):
        likelihood._shaped_noise_covar(
            torch.Size([2, 2]),
            noise=torch.full((3, 2), 0.02),
        )


def test_multitask_fixed_noise_fantasy_appends_data_axis() -> None:
    likelihood = MultitaskFixedNoiseGaussianLikelihood(
        torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=torch.double)
    )
    fantasy = likelihood.get_fantasy_likelihood(
        noise=torch.tensor([[0.05, 0.06]], dtype=torch.double)
    )
    assert fantasy.task_noise.shape == torch.Size([3, 2])
    torch.testing.assert_close(
        fantasy.task_noise,
        torch.tensor(
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            dtype=torch.double,
        ),
    )


@pytest.mark.parametrize(
    "filename",
    [
        "mace_multitask.py",
        "chgnet_multitask.py",
        "m3gnet_multitask.py",
        "alignn_multitask.py",
        "crabnet_multitask.py",
    ],
)
def test_material_correlated_families_forward_train_yvar(filename: str) -> None:
    source = (
        Path("src/bochan/models/regression/gaussian/deep") / filename
    ).read_text(encoding="utf-8")
    assert "does not yet support train_Yvar" not in source
    assert "train_Yvar=train_Yvar" in source
