from __future__ import annotations

import pytest
import torch
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from torch import Tensor

from bochan.acquisition.binary.bayesian_optimization.knowledge_gradient import (
    qBinaryKnowledgeGradient,
)


class _CorrelatedBinaryLatentModel(Model):
    """Small differentiable latent GP-like model for acquisition contract tests."""

    def __init__(self) -> None:
        super().__init__()
        self.likelihood = BernoulliLikelihood()

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def latent_posterior(self, X: Tensor):
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        mean = 2.0 * (X[..., 0] - 0.5)
        distance2 = torch.cdist(X, X).square()
        covariance = torch.exp(-distance2 / 0.08)
        eye = torch.eye(
            X.shape[-2],
            dtype=X.dtype,
            device=X.device,
        )
        covariance = covariance + 1e-4 * eye
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))

    def posterior(self, X: Tensor, **kwargs):
        del kwargs
        latent = self.latent_posterior(X)
        probability = self.likelihood(latent.distribution).mean
        return type("ProbabilityPosterior", (), {"mean": probability.unsqueeze(-1)})()


def _terminal(dtype=torch.double) -> Tensor:
    return torch.linspace(0.0, 1.0, 17, dtype=dtype).unsqueeze(-1)


def test_binary_kg_is_finite_nonnegative_and_differentiable() -> None:
    model = _CorrelatedBinaryLatentModel()
    acq = qBinaryKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        num_samples=128,
        seed=123,
    )
    X = torch.tensor([[[0.25]], [[0.55]], [[0.85]]], dtype=torch.double, requires_grad=True)

    value = acq(X)

    assert value.shape == torch.Size([3])
    assert torch.isfinite(value).all()
    assert torch.all(value >= 0.0)
    value.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_binary_kg_uses_deterministic_saa_samples() -> None:
    model = _CorrelatedBinaryLatentModel()
    acq = qBinaryKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        num_samples=64,
        seed=17,
    )
    X = torch.tensor([[[0.42]], [[0.73]]], dtype=torch.double)

    first = acq(X)
    second = acq(X)

    assert torch.equal(first, second)


def test_binary_kg_can_optimize_class_zero_probability() -> None:
    model = _CorrelatedBinaryLatentModel()
    X = torch.tensor([[[0.35]], [[0.75]]], dtype=torch.double)
    class_one = qBinaryKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        target_class=1,
        num_samples=64,
        seed=5,
    )(X)
    class_zero = qBinaryKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        target_class=0,
        num_samples=64,
        seed=5,
    )(X)

    assert torch.isfinite(class_one).all()
    assert torch.isfinite(class_zero).all()
    assert not torch.equal(class_one, class_zero)


def test_binary_kg_generates_continuous_terminal_set_from_bounds() -> None:
    model = _CorrelatedBinaryLatentModel()
    baseline = torch.tensor([[0.1], [0.9]], dtype=torch.double)
    acq = qBinaryKnowledgeGradient(
        model,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        X_baseline=baseline,
        terminal_size=8,
        num_samples=16,
        seed=9,
    )

    assert acq.terminal_set.shape == torch.Size([10, 1])
    assert torch.equal(acq.terminal_set[:2], baseline)
    assert torch.all((acq.terminal_set >= 0.0) & (acq.terminal_set <= 1.0))


def test_binary_kg_rejects_automatic_mixed_terminal_generation() -> None:
    model = _CorrelatedBinaryLatentModel()
    model.cat_dims = [0]
    with pytest.raises(ValueError, match="mixed/categorical"):
        qBinaryKnowledgeGradient(
            model,
            bounds=torch.tensor([[0.0], [2.0]], dtype=torch.double),
            X_baseline=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        )


def test_binary_kg_allows_explicit_mixed_terminal_set() -> None:
    model = _CorrelatedBinaryLatentModel()
    model.cat_dims = [0]
    acq = qBinaryKnowledgeGradient(
        model,
        terminal_set=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double),
        num_samples=16,
    )
    assert acq.terminal_set.shape == torch.Size([3, 1])


def test_binary_kg_rejects_pending_labels_and_q_greater_than_one() -> None:
    model = _CorrelatedBinaryLatentModel()
    with pytest.raises(NotImplementedError, match="pending-label"):
        qBinaryKnowledgeGradient(
            model,
            terminal_set=_terminal(),
            X_pending=torch.tensor([[0.3]], dtype=torch.double),
        )

    acq = qBinaryKnowledgeGradient(model, terminal_set=_terminal(), num_samples=16)
    with pytest.raises(AssertionError):
        acq(torch.rand(2, 2, 1, dtype=torch.double))


def test_binary_kg_can_use_mc_points_as_explicit_terminal_grid() -> None:
    model = _CorrelatedBinaryLatentModel()
    grid = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    acq = qBinaryKnowledgeGradient(model, mc_points=grid, num_samples=16)
    assert torch.equal(acq.terminal_set, grid)


def test_binary_kg_runs_optimize_acqf_q1() -> None:
    model = _CorrelatedBinaryLatentModel()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = qBinaryKnowledgeGradient(
        model,
        terminal_set=torch.linspace(0.0, 1.0, 9, dtype=torch.double).unsqueeze(-1),
        num_samples=32,
        seed=23,
    )

    candidate, value = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 40},
    )

    assert candidate.shape == torch.Size([1, 1])
    assert value.numel() == 1
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
