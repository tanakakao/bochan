from __future__ import annotations

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.ensemble import EnsembleModel
from botorch.models.transforms.input import InputPerturbation
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import nn

from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model, resolve_model_cls
from bochan.models.regression.neural import (
    DeepEnsembleMixedRegressorModel,
    DeepEnsembleRegressorModel,
)


class _AffineMember(nn.Module):
    def __init__(self, weight: float, bias: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[weight]], dtype=torch.double))
        self.bias = nn.Parameter(torch.tensor([bias], dtype=torch.double))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., :1] @ self.weight + self.bias


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.double)
    train_Y = (2.0 * train_X + 0.5).clone()
    return train_X, train_Y


def _fitted_affine_ensemble() -> DeepEnsembleRegressorModel:
    train_X, train_Y = _training_data()
    members = [
        _AffineMember(1.0, 0.0),
        _AffineMember(1.5, 0.5),
        _AffineMember(2.0, 1.0),
    ]
    model = DeepEnsembleRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        members=members,
        bootstrap=False,
        outcome_transform=None,
    ).fit(num_epochs=1, lr=1e-12)
    with torch.no_grad():
        for member, weight, bias in zip(
            model.members,
            (1.0, 1.5, 2.0),
            (0.0, 0.5, 1.0),
            strict=True,
        ):
            member.weight.fill_(weight)
            member.bias.fill_(bias)
    return model


def test_deep_ensemble_is_botorch_ensemble_with_epistemic_posterior() -> None:
    model = _fitted_affine_ensemble()
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double)

    assert isinstance(model, EnsembleModel)
    posterior = model.posterior(X)

    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    expected_values = torch.tensor(
        [
            [[0.25], [0.75]],
            [[0.875], [1.625]],
            [[1.5], [2.5]],
        ],
        dtype=torch.double,
    )
    torch.testing.assert_close(posterior.values, expected_values)
    torch.testing.assert_close(posterior.mean, expected_values.mean(dim=0))
    assert torch.all(posterior.variance > 0)


def test_deep_ensemble_keeps_acquisition_gradient_to_candidate_x() -> None:
    model = _fitted_affine_ensemble()
    candidate = torch.tensor([[[0.4]]], dtype=torch.double, requires_grad=True)
    acqf = qLogExpectedImprovement(model=model, best_f=torch.tensor(-10.0, dtype=torch.double))

    value = acqf(candidate)
    value.backward()

    assert torch.isfinite(value).all()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0


def test_deep_ensemble_supports_eval_only_input_perturbation() -> None:
    train_X, train_Y = _training_data()
    perturbation = InputPerturbation(
        perturbation_set=torch.tensor([[-0.05], [0.05]], dtype=torch.double),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
    )
    model = DeepEnsembleRegressorModel(
        train_X=train_X,
        train_Y=train_Y,
        ensemble_size=2,
        hidden_dims=(8,),
        random_state=11,
        input_transform=perturbation,
        outcome_transform=None,
    ).fit(num_epochs=2, lr=0.01)

    posterior = model.posterior(torch.tensor([[0.5]], dtype=torch.double))

    assert posterior.values.shape == torch.Size([2, 2, 1])


def test_deep_ensemble_default_registry_and_high_level_fit() -> None:
    train_X, train_Y = _training_data()
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type="deep_ensemble",
            outcome_transform=False,
        )
    )
    assert model_cls is DeepEnsembleRegressorModel

    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="deep_ensemble",
            outcome_transform=False,
            model_kwargs={
                "ensemble_size": 2,
                "hidden_dims": (8,),
                "random_state": 3,
            },
        ),
    )
    fitted = fit_model(bundle, FitConfig(num_epochs=3, lr=0.01, batch_size=3))

    assert fitted.model.is_fitted
    assert fitted.mll is None
    assert len(fitted.model.fit_losses) == 2
    assert all(len(history) == 3 for history in fitted.model.fit_losses)


def test_mixed_deep_ensemble_registry_one_hot_and_continuous_gradient() -> None:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.25, 1.0],
            [0.5, 0.0],
            [0.75, 1.0],
            [1.0, 0.0],
        ],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.25 * train_X[:, 1:2]).clone()
    config = ModelConfig(
        task_type="regression",
        model_type="deep_ensemble",
        cat_dims=[1],
        outcome_transform=False,
        model_kwargs={
            "ensemble_size": 2,
            "hidden_dims": (8,),
            "random_state": 7,
        },
    )

    model_cls = resolve_model_cls(config)
    assert model_cls is DeepEnsembleMixedRegressorModel

    model = fit_model(
        build_model(train_X, train_Y, config),
        FitConfig(num_epochs=3, lr=0.01),
    ).model
    assert model.categorical_values == {1: (0.0, 1.0)}

    candidate = torch.tensor([[0.4, 1.0]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    posterior.mean.sum().backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad[:, 0]).all()
    assert posterior.values.shape == torch.Size([2, 1, 1])

    with pytest.raises(ValueError, match="not observed during training"):
        model.posterior(torch.tensor([[0.4, 2.0]], dtype=torch.double))


def test_deep_ensemble_preserves_default_outcome_transform_contract() -> None:
    train_X, train_Y = _training_data()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="deep_ensemble",
            model_kwargs={
                "ensemble_size": 2,
                "hidden_dims": (8,),
                "random_state": 17,
            },
        ),
    )
    model = fit_model(bundle, FitConfig(num_epochs=2, lr=0.01)).model

    posterior = model.posterior(torch.tensor([[0.4]], dtype=torch.double))

    assert posterior.mean.shape == torch.Size([1, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
