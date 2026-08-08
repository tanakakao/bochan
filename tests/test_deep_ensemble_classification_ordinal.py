from __future__ import annotations

import pytest
import torch

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.binary.neural import (
    DeepEnsembleBinaryClassificationModel,
    DeepEnsembleMixedBinaryClassificationModel,
)
from bochan.models.classification.multiclass.neural import (
    DeepEnsembleMixedMulticlassClassificationModel,
    DeepEnsembleMulticlassClassificationModel,
)
from bochan.models.ordinal.neural import (
    DeepEnsembleMixedOrdinalModel,
    DeepEnsembleOrdinalModel,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.15], [0.3], [0.45], [0.6], [0.75], [0.9], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]], dtype=torch.long)
    return X, Y


def _multiclass_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.1], [0.2], [0.4], [0.5], [0.6], [0.8], [0.9], [1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def _mixed(X: torch.Tensor) -> torch.Tensor:
    categories = torch.tensor(
        [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0]],
        dtype=X.dtype,
    )
    if X.shape[0] == 8:
        categories = categories[:8]
    return torch.cat([X, categories], dim=-1)


def test_binary_deep_ensemble_probability_and_candidate_gradient() -> None:
    train_X, train_Y = _binary_data()
    model = DeepEnsembleBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=False,
        random_state=0,
    ).fit(num_epochs=4, lr=0.02, batch_size=8)

    X = torch.tensor([[[0.5]]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(X)
    assert posterior.values.shape == torch.Size([1, 3, 1, 1])
    assert torch.isfinite(posterior.mean).all()
    posterior.mean.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_multiclass_deep_ensemble_probability_simplex() -> None:
    train_X, train_Y = _multiclass_data()
    model = DeepEnsembleMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=False,
        random_state=0,
    ).fit(num_epochs=4, lr=0.02, batch_size=9)

    probs = model.class_probs(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert probs.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )


def test_ordinal_deep_ensemble_probability_and_utility() -> None:
    train_X, train_Y = _multiclass_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=False,
        random_state=0,
    ).fit(num_epochs=4, lr=0.02, batch_size=9)

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    probs = model.class_probs(X)
    assert probs.shape == torch.Size([2, 3])
    torch.testing.assert_close(
        probs.sum(dim=-1),
        torch.ones(2, dtype=torch.double),
        atol=1e-6,
        rtol=1e-6,
    )
    utilities = torch.tensor([0.0, 1.0, 3.0], dtype=torch.double)
    expected = model.expected_utility(X, utilities)
    assert torch.isfinite(expected).all()


@pytest.mark.parametrize(
    ("task_type", "expected_cls"),
    [
        ("binary", DeepEnsembleBinaryClassificationModel),
        ("multiclass", DeepEnsembleMulticlassClassificationModel),
        ("ordinal", DeepEnsembleOrdinalModel),
    ],
)
def test_deep_ensemble_registry_resolves_task_local_models(task_type, expected_cls) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type=task_type,
            model_type="deep_ensemble",
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


@pytest.mark.parametrize(
    ("task_type", "expected_cls"),
    [
        ("binary", DeepEnsembleMixedBinaryClassificationModel),
        ("multiclass", DeepEnsembleMixedMulticlassClassificationModel),
        ("ordinal", DeepEnsembleMixedOrdinalModel),
    ],
)
def test_mixed_deep_ensemble_registry_resolves_task_local_models(task_type, expected_cls) -> None:
    resolved = resolve_model_cls(
        ModelConfig(
            task_type=task_type,
            model_type="deep_ensemble",
            cat_dims=[1],
            outcome_transform=False,
        )
    )
    assert resolved is expected_cls


def test_mixed_binary_deep_ensemble_rejects_unseen_categories() -> None:
    train_X, train_Y = _binary_data()
    model = DeepEnsembleMixedBinaryClassificationModel(
        train_X=_mixed(train_X),
        train_Y=train_Y,
        cat_dims=[1],
        ensemble_size=2,
        hidden_dims=(8,),
        bootstrap=False,
        random_state=0,
    ).fit(num_epochs=3, lr=0.02, batch_size=8)

    assert model.categorical_values == {1: (0.0, 1.0, 2.0)}
    with pytest.raises(ValueError, match="not observed during training"):
        model.class_probs(torch.tensor([[0.5, 3.0]], dtype=torch.double))


def test_high_level_deep_ensemble_binary_fit() -> None:
    train_X, train_Y = _binary_data()
    config = ModelConfig(
        task_type="binary",
        model_type="deep_ensemble",
        outcome_transform=False,
        model_kwargs={
            "ensemble_size": 2,
            "hidden_dims": (8,),
            "bootstrap": False,
            "random_state": 0,
        },
    )
    fitted = fit_model(
        build_model(train_X, train_Y, config),
        FitConfig(num_epochs=3, lr=0.02, batch_size=8),
    )
    assert isinstance(fitted.model, DeepEnsembleBinaryClassificationModel)
    assert fitted.model.is_fitted
