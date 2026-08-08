from __future__ import annotations

import torch

from bochan.models.classification.external import (
    NGBoostBinaryClassificationModel,
    NGBoostMulticlassClassificationModel,
    RandomForestBinaryClassificationModel,
    RandomForestMulticlassClassificationModel,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 16, dtype=torch.double).unsqueeze(-1)
    y = (X[:, 0] >= 0.5).to(dtype=torch.double).unsqueeze(-1)
    return X, y


def _multiclass_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    y = torch.zeros(24, 1, dtype=torch.double)
    y[8:16] = 1.0
    y[16:] = 2.0
    return X, y


def test_real_random_forest_binary_and_multiclass_smoke() -> None:
    binary_X, binary_y = _binary_data()
    binary = RandomForestBinaryClassificationModel(
        train_X=binary_X,
        train_Y=binary_y,
        n_estimators=8,
        random_state=0,
    ).fit()
    binary_posterior = binary.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert binary_posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(binary_posterior.mean).all()
    assert torch.any(binary_posterior.epistemic_variance >= 0)

    multiclass_X, multiclass_y = _multiclass_data()
    multiclass = RandomForestMulticlassClassificationModel(
        train_X=multiclass_X,
        train_Y=multiclass_y,
        n_estimators=8,
        random_state=0,
    ).fit()
    multiclass_posterior = multiclass.posterior(
        torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.double)
    )
    assert multiclass_posterior.mean.shape == torch.Size([3, 3])
    torch.testing.assert_close(
        multiclass_posterior.mean.sum(dim=-1),
        torch.ones(3, dtype=torch.double),
    )


def test_real_ngboost_binary_and_multiclass_smoke() -> None:
    binary_X, binary_y = _binary_data()
    binary = NGBoostBinaryClassificationModel(
        train_X=binary_X,
        train_Y=binary_y,
        n_estimators=5,
        random_state=0,
        verbose=False,
    ).fit()
    binary_posterior = binary.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert binary_posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(binary_posterior.mean).all()

    multiclass_X, multiclass_y = _multiclass_data()
    multiclass = NGBoostMulticlassClassificationModel(
        train_X=multiclass_X,
        train_Y=multiclass_y,
        n_estimators=5,
        random_state=0,
        verbose=False,
    ).fit()
    multiclass_posterior = multiclass.posterior(
        torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.double)
    )
    assert multiclass_posterior.mean.shape == torch.Size([3, 3])
    torch.testing.assert_close(
        multiclass_posterior.mean.sum(dim=-1),
        torch.ones(3, dtype=torch.double),
        atol=1e-8,
        rtol=1e-8,
    )
