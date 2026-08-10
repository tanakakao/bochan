from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.regression.gaussian.high_dim.decomposition import (
    PCAGaussianMixedGPModel,
    PCAGaussianGPModel,
    REMBOGaussianMixedGPModel,
    REMBOGaussianGPModel,
)


@pytest.mark.parametrize("model_cls", [PCAGaussianGPModel, REMBOGaussianGPModel])
def test_exact_projected_models_ignore_beta_mll_kwarg(model_cls) -> None:
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_Y = train_X[:, :1].square()

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=2,
    )

    mll = model.make_mll(beta=0.01)

    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.model is model.base_model


@pytest.mark.parametrize("model_cls", [PCAGaussianGPModel, REMBOGaussianGPModel])
def test_exact_projected_models_reject_unknown_mll_kwargs(model_cls) -> None:
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_Y = train_X[:, :1].square()

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        latent_dim=2,
    )

    with pytest.raises(TypeError, match="unsupported"):
        model.make_mll(unknown_option=1)


@pytest.mark.parametrize(
    "model_cls",
    [PCAGaussianMixedGPModel, REMBOGaussianMixedGPModel],
)
def test_exact_projected_mixed_models_ignore_beta_mll_kwarg(model_cls) -> None:
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_X[:, 4] = torch.arange(12, dtype=torch.double) % 2
    train_Y = train_X[:, :1].square() + 0.1 * train_X[:, 4:5]

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[4],
        category_counts={4: 2},
        latent_dim=2,
    )

    mll = model.make_mll(beta=0.01)

    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.model is model.base_model


@pytest.mark.parametrize(
    "model_cls",
    [PCAGaussianMixedGPModel, REMBOGaussianMixedGPModel],
)
def test_exact_projected_mixed_models_reject_unknown_mll_kwargs(model_cls) -> None:
    train_X = torch.rand(12, 5, dtype=torch.double)
    train_X[:, 4] = torch.arange(12, dtype=torch.double) % 2
    train_Y = train_X[:, :1].square() + 0.1 * train_X[:, 4:5]

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[4],
        category_counts={4: 2},
        latent_dim=2,
    )

    with pytest.raises(TypeError, match="unsupported"):
        model.make_mll(unknown_option=1)
