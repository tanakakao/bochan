from __future__ import annotations

import pytest
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from torch import nn

from bochan.acquisition.regression.active_learning import (
    qRegressionBALD,
    qRegressionPosteriorVariance,
    qRegressionPredictiveEntropy,
)
from bochan.models.regression.beta.deep.deepkernel import (
    DeepKernelBetaGPModel,
    DeepKernelBetaMixedGPModel,
)


def _make_beta_targets(continuous_x: torch.Tensor) -> torch.Tensor:
    """Beta 回帰用の (0, 1) toy target を作る。"""
    latent = 1.5 * continuous_x[..., 0] - 0.75 * continuous_x[..., 1]
    return 0.1 + 0.8 * torch.sigmoid(latent)


def _make_continuous_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_x = torch.rand(12, 2, dtype=dtype)
    return train_x, _make_beta_targets(train_x)


def _make_mixed_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    continuous_x = torch.rand(12, 2, dtype=dtype)
    categorical_x = torch.randint(0, 2, (12, 1)).to(dtype=dtype)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = (
        _make_beta_targets(continuous_x)
        + 0.05 * (categorical_x.squeeze(-1) - 0.5)
    ).clamp(0.05, 0.95)
    return train_x, train_y


def _assert_module_matches_reference(module: nn.Module, reference: torch.Tensor) -> None:
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    assert parameters or buffers
    for tensor in [*parameters, *buffers]:
        if tensor.is_floating_point():
            assert tensor.dtype == reference.dtype
        assert tensor.device == reference.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deepkernel_beta_default_modules_match_training_dtype(dtype: torch.dtype) -> None:
    train_x, train_y = _make_continuous_data(dtype)

    model = DeepKernelBetaGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_inducing=6,
    )
    model.eval()
    model.likelihood.eval()

    _assert_module_matches_reference(model.model.feature_extractor, train_x)
    _assert_module_matches_reference(model.model.mean_module, train_x)
    _assert_module_matches_reference(model.model.covar_module, train_x)

    test_x = torch.rand(4, 2, dtype=dtype)
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=False)

    assert posterior.mean.shape == torch.Size([4, 1])
    assert posterior.variance.shape == torch.Size([4, 1])
    assert posterior.mean.dtype == dtype
    assert posterior.variance.dtype == dtype
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert ((posterior.mean > 0) & (posterior.mean < 1)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_deepkernel_beta_mixed_default_modules_match_training_dtype(dtype: torch.dtype) -> None:
    train_x, train_y = _make_mixed_data(dtype)

    model = DeepKernelBetaMixedGPModel(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        num_inducing=6,
    )
    model.eval()
    model.likelihood.eval()

    _assert_module_matches_reference(model.model.feature_extractor, train_x)
    _assert_module_matches_reference(model.model.mean_module, train_x)
    _assert_module_matches_reference(model.model.covar_module, train_x)

    continuous_x = torch.rand(4, 2, dtype=dtype)
    categorical_x = torch.randint(0, 2, (4, 1)).to(dtype=dtype)
    test_x = torch.cat([continuous_x, categorical_x], dim=-1)
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=False)

    assert posterior.mean.shape == torch.Size([4, 1])
    assert posterior.variance.shape == torch.Size([4, 1])
    assert posterior.mean.dtype == dtype
    assert posterior.variance.dtype == dtype
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert ((posterior.mean > 0) & (posterior.mean < 1)).all()


def test_deepkernel_beta_moves_custom_modules_before_first_forward() -> None:
    train_x, train_y = _make_continuous_data(torch.float64)
    feature_extractor = nn.Sequential(
        nn.Linear(2, 4),
        nn.SiLU(),
        nn.Linear(4, 2),
    )
    mean_module = ConstantMean()
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))

    model = DeepKernelBetaGPModel(
        train_X=train_x,
        train_Y=train_y,
        feature_extractor=feature_extractor,
        mean_module=mean_module,
        covar_module=covar_module,
        num_inducing=6,
    )

    assert model.model.feature_extractor is feature_extractor
    assert model.model.mean_module is mean_module
    assert model.model.covar_module is covar_module
    _assert_module_matches_reference(feature_extractor, train_x)
    _assert_module_matches_reference(mean_module, train_x)
    _assert_module_matches_reference(covar_module, train_x)


def test_deepkernel_beta_mixed_moves_custom_extractor_to_double() -> None:
    train_x, train_y = _make_mixed_data(torch.float64)
    feature_extractor = nn.Sequential(
        nn.Linear(2, 4),
        nn.SiLU(),
        nn.Linear(4, 2),
    )

    model = DeepKernelBetaMixedGPModel(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        feature_extractor=feature_extractor,
        num_inducing=6,
    )

    assert model.model.feature_extractor is feature_extractor
    _assert_module_matches_reference(feature_extractor, train_x)


@pytest.mark.parametrize(
    "acquisition_class",
    [
        qRegressionPredictiveEntropy,
        qRegressionBALD,
        qRegressionPosteriorVariance,
    ],
)
def test_regression_active_learning_accepts_double_beta_deepkernel(
    acquisition_class: type,
) -> None:
    train_x, train_y = _make_continuous_data(torch.float64)
    model = DeepKernelBetaGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_inducing=6,
    )
    model.eval()
    model.likelihood.eval()

    candidates = torch.rand(2, 3, 2, dtype=torch.float64, requires_grad=True)
    acquisition = acquisition_class(model=model)
    value = acquisition(candidates)

    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert candidates.grad is not None
    assert torch.isfinite(candidates.grad).all()
