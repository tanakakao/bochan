from __future__ import annotations

import pytest
import torch

from bochan.models.ordinal.external import (
    NGBoostMixedOrdinalModel,
    NGBoostOrdinalEnsembleModel,
    NGBoostOrdinalModel,
    RandomForestMixedOrdinalModel,
    RandomForestOrdinalModel,
)

pytest.importorskip("sklearn")
pytest.importorskip("ngboost")


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.1], [0.2], [0.35], [0.5], [0.65], [0.8], [0.9], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [[0], [0], [0], [1], [1], [1], [2], [2], [2]],
        dtype=torch.long,
    )
    return X, Y


def _mixed_data() -> tuple[torch.Tensor, torch.Tensor]:
    X, Y = _data()
    categories = torch.tensor(
        [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0]],
        dtype=torch.double,
    )
    return torch.cat([X, categories], dim=-1), Y


@pytest.mark.parametrize("model", ["random_forest", "ngboost", "ngboost_ensemble"])
def test_real_external_ordinal_models_smoke(model: str) -> None:
    train_X, train_Y = _data()
    if model == "random_forest":
        fitted = RandomForestOrdinalModel(
            train_X=train_X,
            train_Y=train_Y,
            n_estimators=6,
            max_depth=3,
            random_state=0,
        ).fit()
    elif model == "ngboost":
        fitted = NGBoostOrdinalModel(
            train_X=train_X,
            train_Y=train_Y,
            n_estimators=8,
            random_state=0,
            verbose=False,
        ).fit()
    else:
        fitted = NGBoostOrdinalEnsembleModel(
            train_X=train_X,
            train_Y=train_Y,
            ensemble_size=2,
            bootstrap=True,
            random_state=0,
            n_estimators=5,
            verbose=False,
        ).fit()

    X = torch.tensor([[0.25], [0.72]], dtype=torch.double)
    probs = fitted.class_probs(X)
    posterior = fitted.posterior(X)

    assert probs.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.isfinite(probs).all()
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


@pytest.mark.parametrize(
    "model_cls",
    [RandomForestMixedOrdinalModel, NGBoostMixedOrdinalModel],
)
def test_real_mixed_external_ordinal_smoke(model_cls) -> None:
    train_X, train_Y = _mixed_data()
    kwargs = {"random_state": 0}
    if model_cls is RandomForestMixedOrdinalModel:
        kwargs.update({"n_estimators": 5, "max_depth": 3})
    else:
        kwargs.update({"n_estimators": 5, "verbose": False})

    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        **kwargs,
    ).fit()
    probs = model.class_probs(torch.tensor([[0.42, 1.0]], dtype=torch.double))

    assert probs.shape == torch.Size([1, 3])
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(1, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )
