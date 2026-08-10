from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import VariationalELBO

from bochan.models.ordinal.high_dim.decomposition import (
    PCAOrdinalGPModel,
    PCAOrdinalMixedGPModel,
    REMBOOrdinalGPModel,
    REMBOOrdinalMixedGPModel,
)


@pytest.mark.parametrize("model_cls", [PCAOrdinalGPModel, REMBOOrdinalGPModel])
def test_ordinal_projected_models_build_variational_mll_with_beta(model_cls) -> None:
    train_X = torch.rand(18, 5, dtype=torch.double)
    train_Y = torch.arange(18) % 3

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        n_components=2,
        num_inducing=8,
    )

    mll = model.make_mll(beta=0.01)

    assert isinstance(mll, VariationalELBO)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood


@pytest.mark.parametrize(
    "model_cls",
    [PCAOrdinalMixedGPModel, REMBOOrdinalMixedGPModel],
)
def test_ordinal_projected_mixed_models_build_variational_mll_with_beta(model_cls) -> None:
    train_X = torch.rand(18, 5, dtype=torch.double)
    train_X[:, 4] = torch.arange(18, dtype=torch.double) % 2
    train_Y = torch.arange(18) % 3

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[4],
        category_counts={4: 2},
        n_components=2,
        num_inducing=8,
    )

    mll = model.make_mll(beta=0.01)

    assert isinstance(mll, VariationalELBO)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood
