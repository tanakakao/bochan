from __future__ import annotations

import torch
from torch import nn

from bochan.acquisition.ordinal.active_learning.single_output import qOrdinalBALD
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from bochan.models.ordinal.neural.deep_ensemble import DeepEnsembleOrdinalModel
from bochan.posteriors.classification_ensemble import ClassificationEnsemblePosterior


class _ConstantOrdinalMember(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value), dtype=torch.double))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.value.expand(*X.shape[:-1], 1)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[0.0], [0.15], [0.3], [0.45], [0.6], [0.75], [0.9], [1.0], [0.85]],
        dtype=torch.double,
    )
    Y = torch.tensor([[0], [0], [0], [1], [1], [1], [2], [2], [2]], dtype=torch.long)
    return X, Y


def test_ordinal_bald_uses_finite_probability_members_for_deep_ensemble() -> None:
    train_X, train_Y = _training_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        members=[
            _ConstantOrdinalMember(-1.5),
            _ConstantOrdinalMember(0.0),
            _ConstantOrdinalMember(1.5),
        ],
        ensemble_size=3,
        bootstrap=False,
    )
    model._is_fitted = True

    X = torch.tensor([[[0.35]]], dtype=torch.double)
    probability_posterior = model.probability_posterior(X)
    assert isinstance(probability_posterior, ClassificationEnsemblePosterior)

    probs = probability_posterior.values
    weights = probability_posterior.weights.to(dtype=probs.dtype, device=probs.device)
    weight_shape = [1] * probs.ndim
    weight_shape[-3] = weights.numel()
    mean_probs = (weights.view(*weight_shape) * probs).sum(dim=-3)
    predictive_entropy = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=-1)

    member_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_weight_shape = [1] * member_entropy.ndim
    entropy_weight_shape[-2] = weights.numel()
    expected_conditional_entropy = (
        weights.view(*entropy_weight_shape) * member_entropy
    ).sum(dim=-2)
    expected = (predictive_entropy - expected_conditional_entropy).squeeze(-1)

    actual = qOrdinalBALD(
        model=model,
        num_samples=16,
        exclude_observed_duplicates=False,
    )(X)

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual > 0.0)


def test_ordinal_bald_single_finite_member_is_zero() -> None:
    train_X, train_Y = _training_data()
    model = DeepEnsembleOrdinalModel(
        train_X=train_X,
        train_Y=train_Y,
        members=[_ConstantOrdinalMember(0.25)],
        ensemble_size=1,
        bootstrap=False,
    )
    model._is_fitted = True

    value = qOrdinalBALD(
        model=model,
        num_samples=16,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.4]]], dtype=torch.double))

    torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0.0)


def test_ordinal_bald_gp_style_posterior_still_uses_mc_path() -> None:
    class _GaussianOrdinalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.likelihood = OrdinalLogitLikelihood(num_classes=3).double()
            self.register_buffer("train_X", torch.tensor([[0.0], [1.0]], dtype=torch.double))

        @property
        def ordinal_likelihood(self):
            return self.likelihood

        @property
        def batch_shape(self) -> torch.Size:
            return torch.Size()

        def posterior(self, X: torch.Tensor):
            from botorch.posteriors.gpytorch import GPyTorchPosterior
            from gpytorch.distributions import MultivariateNormal

            mean = torch.zeros(*X.shape[:-1], dtype=X.dtype, device=X.device)
            q = X.shape[-2]
            covariance = torch.eye(q, dtype=X.dtype, device=X.device).expand(
                *X.shape[:-2], q, q
            )
            return GPyTorchPosterior(MultivariateNormal(mean, covariance))

    model = _GaussianOrdinalModel()
    value = qOrdinalBALD(
        model=model,
        num_samples=8,
        exclude_observed_duplicates=False,
    )(torch.tensor([[[0.5]]], dtype=torch.double))
    assert torch.isfinite(value).all()
