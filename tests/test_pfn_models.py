from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from bochan.acquisition.regression.pfn import (
    PFNExpectedImprovement,
    PFNProbabilityOfImprovement,
    PFNUpperConfidenceBound,
)
from bochan.api import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model
from bochan.models.regression.foundation import PFNPosterior, PFNRegressorModel


class _FakeCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("centers", torch.tensor([-1.0, 0.0, 1.0]))

    def mean(self, logits: Tensor) -> Tensor:
        return logits.softmax(-1) @ self.centers.to(logits)

    def variance(self, logits: Tensor) -> Tensor:
        probs = logits.softmax(-1)
        centers = self.centers.to(logits)
        mean = probs @ centers
        return probs @ centers.square() - mean.square()

    def ei(self, logits: Tensor, best_f: Tensor | float, maximize: bool = True) -> Tensor:
        assert maximize
        centers = self.centers.to(logits)
        best = torch.as_tensor(best_f, dtype=logits.dtype, device=logits.device)
        improvement = (centers - best).clamp_min(0.0)
        return (logits.softmax(-1) * improvement).sum(-1)

    def pi(self, logits: Tensor, best_f: Tensor | float, maximize: bool = True) -> Tensor:
        assert maximize
        centers = self.centers.to(logits)
        best = torch.as_tensor(best_f, dtype=logits.dtype, device=logits.device)
        return (logits.softmax(-1) * (centers > best).to(logits)).sum(-1)

    def ucb(
        self,
        logits: Tensor,
        best_f: Tensor | float | None,
        rest_prob: float,
        maximize: bool = True,
    ) -> Tensor:
        del best_f, rest_prob
        assert maximize
        return self.mean(logits) + self.variance(logits).sqrt()


class _FakePFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = _FakeCriterion()

    def forward(self, data, single_eval_pos: int):
        _style, X, y = data
        del y, single_eval_pos
        score = X.sum(dim=-1, keepdim=True)
        return torch.cat((-score, torch.zeros_like(score), score), dim=-1)


def _data(dtype=torch.double):
    train_X = torch.tensor(
        [[0.0, 0.0], [0.25, 0.5], [0.75, 0.4], [1.0, 1.0]],
        dtype=dtype,
    )
    train_Y = torch.tensor([[0.1], [0.5], [0.8], [1.2]], dtype=dtype)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype)
    return train_X, train_Y, bounds


def test_pfn_posterior_exposes_marginal_bar_moments_and_gradients():
    train_X, train_Y, bounds = _data()
    model = PFNRegressorModel(
        train_X,
        train_Y,
        bounds=bounds,
        pretrained_model=_FakePFN(),
    ).fit()

    X = torch.tensor([[[0.2, 0.3]], [[0.8, 0.9]]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(X)

    assert isinstance(posterior, PFNPosterior)
    assert posterior.mean.shape == torch.Size([2, 1, 1])
    assert posterior.variance.shape == torch.Size([2, 1, 1])
    assert posterior.probabilities.shape == torch.Size([2, 1, 3])
    assert torch.allclose(posterior.probabilities.sum(dim=-1), torch.ones(2, 1))

    posterior.mean.sum().backward()
    assert X.grad is not None
    assert torch.any(X.grad != 0)

    with pytest.raises(NotImplementedError, match="native"):
        posterior.rsample()


def test_pfn_native_acquisitions_support_q1_and_candidate_gradients():
    train_X, train_Y, bounds = _data()
    model = PFNRegressorModel(
        train_X,
        train_Y,
        bounds=bounds,
        pretrained_model=_FakePFN(),
    ).fit()

    X = torch.tensor([[[0.4, 0.6]], [[0.7, 0.8]]], dtype=torch.double, requires_grad=True)
    acquisitions = [
        PFNExpectedImprovement(model),
        PFNProbabilityOfImprovement(model),
        PFNUpperConfidenceBound(model),
    ]
    for acquisition in acquisitions:
        X.grad = None
        values = acquisition(X)
        assert values.shape == torch.Size([2])
        values.sum().backward()
        assert X.grad is not None
        assert torch.any(X.grad != 0)

    with pytest.raises(AssertionError):
        PFNExpectedImprovement(model)(torch.rand(2, 2, 2, dtype=torch.double))


def test_pfn_minimization_uses_signed_objective_context():
    train_X, train_Y, bounds = _data()
    model = PFNRegressorModel(
        train_X,
        train_Y,
        bounds=bounds,
        pretrained_model=_FakePFN(),
    ).fit()
    X = torch.tensor([[[0.4, 0.6]]], dtype=torch.double)

    maximize_ei = PFNExpectedImprovement(model, maximize=True)(X)
    minimize_ei = PFNExpectedImprovement(model, maximize=False)(X)
    assert torch.isfinite(maximize_ei).all()
    assert torch.isfinite(minimize_ei).all()


def test_pfn_registry_build_and_fit_with_injected_fake_checkpoint():
    train_X, train_Y, bounds = _data()
    bundle = build_model(
        train_X,
        train_Y,
        ModelConfig(
            task_type="regression",
            model_type="pfn",
            model_kwargs={
                "bounds": bounds,
                "pretrained_model": _FakePFN(),
            },
        ),
    )
    bundle = fit_model(bundle, FitConfig())

    assert isinstance(bundle.model, PFNRegressorModel)
    assert bundle.model.is_fitted
    assert bundle.mll is None


def test_pfn_requires_explicit_bounds_when_training_dimension_is_constant():
    train_X = torch.tensor([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="full search-space bounds"):
        PFNRegressorModel(train_X, train_Y, pretrained_model=_FakePFN())
