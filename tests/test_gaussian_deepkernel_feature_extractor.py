from __future__ import annotations

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from torch import nn

from bochan.api import ModelConfig, build_model
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)


class LinearFeatureExtractor(nn.Module):
    """Small extractor with an explicit output contract for regression tests."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.projection(X)


def _continuous_data(
    *,
    num_outputs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_X = torch.rand(10, 4, dtype=torch.double)
    base = train_X[:, :2].sum(dim=-1, keepdim=True)
    if num_outputs == 1:
        return train_X, base
    return train_X, torch.cat([base, base.square()], dim=-1)


def test_custom_feature_extractor_sets_kernel_ard_to_latent_dim() -> None:
    train_X, train_Y = _continuous_data()
    extractor = LinearFeatureExtractor(input_dim=4, output_dim=2)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
        feature_extractor=extractor,
        latent_dim=2,
    )

    assert model.deepkernel.feature_extractor is extractor
    assert model.latent_dim == 2
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == 2
    assert extractor.projection.weight.dtype == train_X.dtype
    assert not any(parameter.requires_grad for parameter in extractor.parameters())

    test_X = torch.rand(3, 4, dtype=torch.double, requires_grad=True)
    posterior = model.posterior(test_X)
    posterior.mean.sum().backward()

    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.variance.shape == torch.Size([3, 1])
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()


def test_default_feature_extractor_can_project_to_independent_latent_dim() -> None:
    train_X, train_Y = _continuous_data()

    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
        latent_dim=3,
    )

    projected = model.deepkernel.feature_extractor(train_X[:2])
    assert projected.shape == torch.Size([2, 3])
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == 3


def test_custom_feature_extractor_latent_dim_can_be_inferred() -> None:
    train_X, train_Y = _continuous_data()
    extractor = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 3))

    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
        feature_extractor=extractor,
    )

    assert model.latent_dim == 3
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == 3


def test_declared_feature_width_must_match_latent_dim() -> None:
    train_X, train_Y = _continuous_data()

    with pytest.raises(ValueError, match="latent_dim does not match"):
        DeepKernelGaussianGPModel(
            train_X=train_X,
            train_Y=train_Y,
            input_transform=None,
            outcome_transform=None,
            feature_extractor=LinearFeatureExtractor(4, 3),
            latent_dim=2,
        )


def test_multi_output_custom_feature_extractor_preserves_posterior_contract() -> None:
    train_X, train_Y = _continuous_data(num_outputs=2)

    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
        feature_extractor=LinearFeatureExtractor(4, 2),
        latent_dim=2,
    )
    posterior = model.posterior(torch.rand(3, 4, dtype=torch.double))

    assert posterior.mean.shape == torch.Size([3, 2])
    assert posterior.variance.shape == torch.Size([3, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_feature_extractor_supports_changed_continuous_width() -> None:
    torch.manual_seed(0)
    continuous = torch.rand(12, 3, dtype=torch.double)
    category = torch.randint(0, 2, (12, 1)).to(dtype=torch.double)
    train_X = torch.cat([continuous[:, :1], category, continuous[:, 1:]], dim=-1)
    train_Y = continuous.sum(dim=-1, keepdim=True) + 0.1 * category
    extractor = LinearFeatureExtractor(input_dim=3, output_dim=5)

    model = DeepKernelGaussianMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        input_transform=None,
        outcome_transform=None,
        feature_extractor=extractor,
        latent_dim=5,
    )

    projected = model.deepkernel._combine_cont_and_cat(train_X[:2])
    posterior = model.posterior(train_X[:3])

    assert model.latent_dim == 5
    assert model.deepkernel.kernel_ord_dims == [0, 1, 2, 3, 4]
    assert model.deepkernel.kernel_cat_dims == [5]
    assert projected.shape == torch.Size([2, 6])
    assert torch.equal(projected[..., -1], train_X[:2, 1])
    assert posterior.mean.shape == torch.Size([3, 1])
    assert torch.isfinite(posterior.mean).all()


def test_model_registry_forwards_feature_extractor_and_latent_dim() -> None:
    train_X, train_Y = _continuous_data()
    extractor = LinearFeatureExtractor(input_dim=4, output_dim=2)

    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="deepkernel",
            model_kwargs={
                "feature_extractor": extractor,
                "latent_dim": 2,
                "input_transform": None,
                "outcome_transform": None,
            },
        ),
    )

    assert isinstance(bundle.model, DeepKernelGaussianGPModel)
    assert bundle.model.deepkernel.feature_extractor is extractor
    assert bundle.model.latent_dim == 2


def test_custom_feature_extractor_supports_qlogei_optimization() -> None:
    train_X, train_Y = _continuous_data()
    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
        feature_extractor=LinearFeatureExtractor(input_dim=4, output_dim=2),
        latent_dim=2,
    )
    fit_deepkernel_mll(model.make_mll(), num_epochs=2)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
    )

    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.stack(
            [
                torch.zeros(4, dtype=torch.double),
                torch.ones(4, dtype=torch.double),
            ]
        ),
        q=2,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == torch.Size([2, 4])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
