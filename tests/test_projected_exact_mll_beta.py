from __future__ import annotations

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.regression.gaussian.high_dim.decomposition import (
    PCASingleTaskGP,
    REMBOSingleTaskGP,
)


@pytest.mark.parametrize("model_cls", [PCASingleTaskGP, REMBOSingleTaskGP])
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


@pytest.mark.parametrize("model_cls", [PCASingleTaskGP, REMBOSingleTaskGP])
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
