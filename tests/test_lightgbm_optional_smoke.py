from __future__ import annotations

import pytest
import torch

from bochan.models.classification.binary.external import LightGBMBinaryEnsembleModel
from bochan.models.classification.multiclass.external import LightGBMMulticlassClassificationModel
from bochan.models.ordinal.external import LightGBMOrdinalEnsembleModel
from bochan.models.regression.external import (
    LightGBMEnsembleModel,
    LightGBMMixedRegressorModel,
    LightGBMRegressorModel,
)


pytest.importorskip("lightgbm")


def _regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    Y = (torch.sin(5.0 * X) + 0.3 * X).clone()
    return X, Y


def _mixed_regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.linspace(0.0, 1.0, 30, dtype=torch.double)
    cat = torch.tensor([10.0, 20.0, 30.0] * 10, dtype=torch.double)
    X = torch.stack([x0, cat], dim=-1)
    Y = (torch.sin(4.0 * x0) + 0.01 * cat).unsqueeze(-1)
    return X, Y


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 30, dtype=torch.double).unsqueeze(-1)
    Y = (X[:, 0] >= 0.5).long().unsqueeze(-1)
    return X, Y


def _multiclass_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 36, dtype=torch.double).unsqueeze(-1)
    Y = torch.zeros(36, 1, dtype=torch.long)
    Y[12:24] = 1
    Y[24:] = 2
    return X, Y


def test_real_lightgbm_regression_single_and_ensemble_smoke() -> None:
    train_X, train_Y = _regression_data()
    single = LightGBMRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()
    posterior = single.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()

    ensemble = LightGBMEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()
    ensemble_posterior = ensemble.posterior(
        torch.tensor([[0.25], [0.75]], dtype=torch.double)
    )
    assert ensemble_posterior.values.shape == torch.Size([3, 2, 1])
    assert torch.isfinite(ensemble_posterior.mean).all()


def test_real_lightgbm_mixed_native_categorical_smoke() -> None:
    train_X, train_Y = _mixed_regression_data()
    model = LightGBMMixedRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()

    posterior = model.posterior(
        torch.tensor([[0.35, 20.0], [0.75, 30.0]], dtype=torch.double)
    )
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()


def test_real_lightgbm_binary_and_multiclass_smoke() -> None:
    binary_X, binary_Y = _binary_data()
    binary = LightGBMBinaryEnsembleModel(
        train_X=binary_X,
        train_Y=binary_Y,
        ensemble_size=3,
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()
    binary_posterior = binary.posterior(
        torch.tensor([[0.25], [0.75]], dtype=torch.double)
    )
    assert binary_posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(binary_posterior.mean).all()

    multiclass_X, multiclass_Y = _multiclass_data()
    multiclass = LightGBMMulticlassClassificationModel(
        train_X=multiclass_X,
        train_Y=multiclass_Y,
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()
    multiclass_posterior = multiclass.posterior(
        torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.double)
    )
    assert multiclass_posterior.mean.shape == torch.Size([3, 3])
    torch.testing.assert_close(
        multiclass_posterior.mean.sum(dim=-1),
        torch.ones(3, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )


def test_real_lightgbm_ordinal_ensemble_smoke() -> None:
    train_X, train_Y = _multiclass_data()
    model = LightGBMOrdinalEnsembleModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        n_estimators=8,
        num_leaves=7,
        verbosity=-1,
        random_state=0,
    ).fit()

    probs = model.class_probs(torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double))
    assert probs.shape == torch.Size([3, 3])
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(3, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )
