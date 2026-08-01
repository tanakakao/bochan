from __future__ import annotations

import pytest
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from torch import nn

from bochan.models.regression.non_gaussian.negative_binomial.deep.negative_binomial_deepkernel import (
    DeepKernelNegativeBinomialGPModel,
    DeepKernelNegativeBinomialMixedGPModel,
)
from bochan.models.regression.non_gaussian.poisson.deep.poisson_deepkernel import (
    DeepKernelPoissonGPModel,
    DeepKernelPoissonMixedGPModel,
)


CONTINUOUS_MODEL_CLASSES = [
    DeepKernelPoissonGPModel,
    DeepKernelNegativeBinomialGPModel,
]
MIXED_MODEL_CLASSES = [
    DeepKernelPoissonMixedGPModel,
    DeepKernelNegativeBinomialMixedGPModel,
]


def _make_count_targets(continuous_x: torch.Tensor) -> torch.Tensor:
    """Count 回帰用の非負整数 toy target を作る。"""
    return (1.0 + 3.0 * continuous_x[..., 0] + 2.0 * continuous_x[..., 1]).round()


def _make_continuous_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_x = torch.rand(12, 2, dtype=dtype)
    return train_x, _make_count_targets(train_x)


def _make_mixed_data(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    continuous_x = torch.rand(12, 2, dtype=dtype)
    categorical_x = torch.randint(0, 2, (12, 1)).to(dtype=dtype)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = _make_count_targets(continuous_x) + categorical_x.squeeze(-1)
    return train_x, train_y


def _assert_module_matches_reference(module: nn.Module, reference: torch.Tensor) -> None:
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    assert parameters or buffers
    for tensor in [*parameters, *buffers]:
        if tensor.is_floating_point():
            assert tensor.dtype == reference.dtype
        assert tensor.device == reference.device


@pytest.mark.parametrize("model_cls", CONTINUOUS_MODEL_CLASSES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_count_deepkernel_default_modules_match_training_dtype(
    model_cls: type,
    dtype: torch.dtype,
) -> None:
    train_x, train_y = _make_continuous_data(dtype)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        num_inducing_points=6,
    )

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


@pytest.mark.parametrize("model_cls", MIXED_MODEL_CLASSES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_count_deepkernel_mixed_default_modules_match_training_dtype(
    model_cls: type,
    dtype: torch.dtype,
) -> None:
    train_x, train_y = _make_mixed_data(dtype)

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        num_inducing_points=6,
    )

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


@pytest.mark.parametrize("model_cls", CONTINUOUS_MODEL_CLASSES)
def test_count_deepkernel_moves_custom_modules_before_first_forward(
    model_cls: type,
) -> None:
    train_x, train_y = _make_continuous_data(torch.float64)
    feature_extractor = nn.Sequential(
        nn.Linear(2, 4),
        nn.SiLU(),
        nn.Linear(4, 2),
    )
    mean_module = ConstantMean()
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        feature_extractor=feature_extractor,
        mean_module=mean_module,
        covar_module=covar_module,
        num_inducing_points=6,
    )

    assert model.model.feature_extractor is feature_extractor
    assert model.model.mean_module is mean_module
    assert model.model.covar_module is covar_module
    _assert_module_matches_reference(feature_extractor, train_x)
    _assert_module_matches_reference(mean_module, train_x)
    _assert_module_matches_reference(covar_module, train_x)


@pytest.mark.parametrize("model_cls", MIXED_MODEL_CLASSES)
def test_count_deepkernel_mixed_moves_custom_extractor_to_double(
    model_cls: type,
) -> None:
    train_x, train_y = _make_mixed_data(torch.float64)
    feature_extractor = nn.Sequential(
        nn.Linear(2, 4),
        nn.SiLU(),
        nn.Linear(4, 2),
    )

    model = model_cls(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=[2],
        feature_extractor=feature_extractor,
        num_inducing_points=6,
    )

    assert model.model.feature_extractor is feature_extractor
    _assert_module_matches_reference(feature_extractor, train_x)
