from __future__ import annotations

import pytest
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import is_ensemble

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.classification.neural import (
    DeepEnsembleBinaryClassificationModel,
    DeepEnsembleMixedBinaryClassificationModel,
    DeepEnsembleMixedMulticlassClassificationModel,
    DeepEnsembleMulticlassClassificationModel,
)
from bochan.models.ordinal.neural import (
    DeepEnsembleMixedOrdinalModel,
    DeepEnsembleOrdinalModel,
)
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior
from bochan.posteriors.ordinal_ensemble import OrdinalEnsemblePosterior


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.15], [0.3], [0.45], [0.6], [0.75], [0.9], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    return train_X, train_Y


def _multiclass_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.1], [0.2], [0.4], [0.5], [0.6], [0.8], [0.9], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    return train_X, train_Y


def _ordinal_data() -> tuple[torch.Tensor, torch.Tensor]:
    return _multiclass_data()


def test_binary_deep_ensemble_probability_posterior_and_gradient() -> None:
    train_X, train_Y = _binary_data()
    model = DeepEnsembleBinaryClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=True,
        random_state=4,
    ).fit(num_epochs=3, lr=0.02)

    candidate = torch.tensor([[0.55]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)

    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 1, 1])
    assert posterior.epistemic_variance.shape == torch.Size([1, 1])
    assert torch.all(posterior.epistemic_variance >= 0)

    probs = model.class_probs(candidate)
    assert probs.shape == torch.Size([1, 2])
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, dtype=torch.double))

    probs[..., 1].sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_multiclass_deep_ensemble_probability_and_latent_bridge() -> None:
    train_X, train_Y = _multiclass_data()
    model = DeepEnsembleMulticlassClassificationModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=True,
        random_state=7,
    ).fit(num_epochs=3, lr=0.02)

    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(X)
    assert isinstance(posterior, ClassificationEnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 3])
    torch.testing.assert_close(
        posterior.values.sum(dim=-1),
        torch.ones(3, 2, dtype=torch.double),
    )
    torch.testing.assert_close(
        model.class_probs(X).sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )

    latent = model.latent_posterior(X)
    assert isinstance(latent, GPyTorchPosterior)
    assert torch.isfinite(latent.mean).all()


def test_classification_high_level_registry_and_fit() -> None:
    train_X, train_Y = _binary_data()
    model_cls = resolve_model_cls(
        ModelConfig(task_type="binary", model_type="deep_ensemble")
    )
    assert model_cls is DeepEnsembleBinaryClassificationModel

    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="binary",
            model_type="deep_ensemble",
            model_kwargs={
                "ensemble_size": 2,
                "hidden_dims": (8,),
                "random_state": 11,
            },
        ),
    )
    fitted = fit_model(bundle, FitConfig(num_epochs=2, lr=0.02, batch_size=4))

    assert fitted.model.is_fitted
    assert fitted.mll is None
    assert len(fitted.model.fit_losses) == 2


def test_mixed_classification_routes_and_preserves_continuous_gradient() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.15, 1.0],
            [0.3, 0.0],
            [0.45, 1.0],
            [0.6, 0.0],
            [0.75, 1.0],
            [0.9, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    config = ModelConfig(
        task_type="binary",
        model_type="deep_ensemble",
        cat_dims=[1],
        model_kwargs={
            "ensemble_size": 2,
            "hidden_dims": (8,),
            "random_state": 13,
        },
    )

    assert resolve_model_cls(config) is DeepEnsembleMixedBinaryClassificationModel
    model = fit_model(
        build_model(train_X, train_Y, config),
        FitConfig(num_epochs=2, lr=0.02),
    ).model
    assert model.categorical_values == {1: (0.0, 1.0)}

    candidate = torch.tensor([[0.5, 1.0]], dtype=torch.double, requires_grad=True)
    p1 = model.class_probs(candidate)[..., 1]
    p1.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad[:, 0]).all()

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.5, 2.0]], dtype=torch.double))


def test_mixed_multiclass_registry() -> None:
    config = ModelConfig(
        task_type="multiclass",
        model_type="deep_ensemble",
        cat_dims=[1],
    )
    assert resolve_model_cls(config) is DeepEnsembleMixedMulticlassClassificationModel


def test_ordinal_deep_ensemble_latent_probability_and_gradient() -> None:
    train_X, train_Y = _ordinal_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=3,
        hidden_dims=(8,),
        bootstrap=True,
        random_state=17,
    ).fit(num_epochs=3, lr=0.02)

    candidate = torch.tensor([[0.55]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)

    assert isinstance(posterior, OrdinalEnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 1, 1])
    assert posterior.distribution.mean.shape == torch.Size([1])
    assert posterior.distribution.covariance_matrix.shape == torch.Size([1, 1])
    samples = posterior.rsample(torch.Size([4]))
    assert samples.shape == torch.Size([4, 1, 1])

    probability_posterior = model.probability_posterior(candidate)
    assert isinstance(probability_posterior, ClassificationEnsemblePosterior)
    assert probability_posterior.values.shape == torch.Size([3, 1, 3])
    probs = model.class_probs(candidate)
    assert probs.shape == torch.Size([1, 3])
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, dtype=torch.double))

    utility = model.expected_utility(
        candidate,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    )
    utility.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_ordinal_deep_ensemble_disables_outer_ensemble_reduction() -> None:
    train_X, train_Y = _ordinal_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=2,
        hidden_dims=(4,),
        random_state=19,
    )

    # Existing ordinal acquisitions already marginalize the finite posterior
    # internally. Keeping BoTorch's outer ensemble reduction enabled would
    # reduce the acquisition output a second time.
    assert not is_ensemble(model)


def test_ordinal_high_level_registry_and_fit() -> None:
    train_X, train_Y = _ordinal_data()
    config = ModelConfig(
        task_type="ordinal",
        model_type="deep_ensemble",
        model_kwargs={
            "ensemble_size": 2,
            "hidden_dims": (8,),
            "random_state": 23,
        },
    )
    assert resolve_model_cls(config) is DeepEnsembleOrdinalModel

    fitted = fit_model(
        build_model(train_X, train_Y, config),
        FitConfig(num_epochs=2, lr=0.02, batch_size=5),
    )
    assert fitted.model.is_fitted
    assert fitted.mll is not None
    assert torch.isfinite(fitted.model.likelihood.cutpoints).all()


def test_mixed_ordinal_registry_gradient_and_unseen_category() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 1.0],
            [0.2, 0.0],
            [0.4, 1.0],
            [0.5, 0.0],
            [0.6, 1.0],
            [0.8, 0.0],
            [0.9, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]])
    config = ModelConfig(
        task_type="ordinal",
        model_type="deep_ensemble",
        cat_dims=[1],
        model_kwargs={
            "ensemble_size": 2,
            "hidden_dims": (8,),
            "random_state": 29,
        },
    )

    assert resolve_model_cls(config) is DeepEnsembleMixedOrdinalModel
    model = fit_model(
        build_model(train_X, train_Y, config),
        FitConfig(num_epochs=2, lr=0.02),
    ).model
    assert model.categorical_values == {1: (0.0, 1.0)}

    candidate = torch.tensor([[0.55, 1.0]], dtype=torch.double, requires_grad=True)
    model.expected_utility(
        candidate,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    ).sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad[:, 0]).all()

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.55, 2.0]], dtype=torch.double))
