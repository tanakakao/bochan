from __future__ import annotations

import pytest
import torch
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from bochan.acquisition.ordinal.bayesian_optimization.knowledge_gradient import (
    qOrdinalKnowledgeGradient,
)
from bochan.models.ordinal.likelihood import OrdinalLogitLikelihood


class _CorrelatedOrdinalLatentModel(Model):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.ordinal_likelihood = OrdinalLogitLikelihood(num_classes=num_classes)
        self.likelihood = self.ordinal_likelihood

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def latent_posterior(self, X: Tensor):
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        mean = 3.0 * (X[..., 0] - 0.5)
        distance2 = torch.cdist(X, X).square()
        covariance = torch.exp(-distance2 / 0.1)
        eye = torch.eye(
            X.shape[-2],
            dtype=X.dtype,
            device=X.device,
        )
        covariance = covariance + 1e-4 * eye
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))

    def posterior(self, X: Tensor, **kwargs):
        del kwargs
        return self.latent_posterior(X)


def _terminal(dtype=torch.double) -> Tensor:
    return torch.linspace(0.0, 1.0, 15, dtype=dtype).unsqueeze(-1)


def test_ordinal_kg_is_finite_nonnegative_and_differentiable() -> None:
    model = _CorrelatedOrdinalLatentModel()
    acq = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        num_samples=128,
        seed=11,
    )
    X = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.double, requires_grad=True)

    value = acq(X)

    assert value.shape == torch.Size([3])
    assert torch.isfinite(value).all()
    assert torch.all(value >= 0.0)
    value.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_ordinal_kg_uses_deterministic_saa_samples() -> None:
    model = _CorrelatedOrdinalLatentModel()
    acq = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        num_samples=64,
        seed=19,
    )
    X = torch.tensor([[[0.37]], [[0.68]]], dtype=torch.double)

    first = acq(X)
    second = acq(X)

    assert torch.equal(first, second)


def test_ordinal_kg_respects_nonuniform_utility_values() -> None:
    model = _CorrelatedOrdinalLatentModel()
    X = torch.tensor([[[0.3]], [[0.7]]], dtype=torch.double)
    linear = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        utility_values=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        num_samples=64,
        seed=7,
    )(X)
    nonlinear = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        utility_values=torch.tensor([0.0, 0.1, 0.4, 5.0]),
        num_samples=64,
        seed=7,
    )(X)

    assert torch.isfinite(linear).all()
    assert torch.isfinite(nonlinear).all()
    assert not torch.equal(linear, nonlinear)


def test_ordinal_kg_can_read_utility_values_from_objective() -> None:
    from bochan.acquisition.objective import OrdinalExpectedUtilityMCObjective

    model = _CorrelatedOrdinalLatentModel()
    objective = OrdinalExpectedUtilityMCObjective(
        ordinal_likelihood=model.ordinal_likelihood,
        utility_values=torch.tensor([0.0, 0.5, 2.0, 10.0]),
    )
    acq = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        objective=objective,
        num_samples=32,
    )
    value = acq(torch.tensor([[[0.55]]], dtype=torch.double))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_ordinal_kg_generates_continuous_terminal_set_from_bounds() -> None:
    model = _CorrelatedOrdinalLatentModel()
    baseline = torch.tensor([[0.15], [0.85]], dtype=torch.double)
    acq = qOrdinalKnowledgeGradient(
        model,
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        X_baseline=baseline,
        terminal_size=8,
        num_samples=16,
        seed=3,
    )

    assert acq.terminal_set.shape == torch.Size([10, 1])
    assert torch.equal(acq.terminal_set[:2], baseline)


def test_ordinal_kg_rejects_automatic_mixed_terminal_generation() -> None:
    model = _CorrelatedOrdinalLatentModel()
    model.cat_dims = [0]
    with pytest.raises(ValueError, match="mixed/categorical"):
        qOrdinalKnowledgeGradient(
            model,
            bounds=torch.tensor([[0.0], [2.0]], dtype=torch.double),
            X_baseline=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        )


def test_ordinal_kg_rejects_bad_utility_length() -> None:
    model = _CorrelatedOrdinalLatentModel()
    acq = qOrdinalKnowledgeGradient(
        model,
        terminal_set=_terminal(),
        utility_values=torch.tensor([0.0, 1.0]),
        num_samples=16,
    )
    with pytest.raises(ValueError, match="utility_values length"):
        acq(torch.tensor([[[0.5]]], dtype=torch.double))


def test_ordinal_kg_rejects_pending_labels_and_q_greater_than_one() -> None:
    model = _CorrelatedOrdinalLatentModel()
    with pytest.raises(NotImplementedError, match="pending-label"):
        qOrdinalKnowledgeGradient(
            model,
            terminal_set=_terminal(),
            X_pending=torch.tensor([[0.3]], dtype=torch.double),
        )

    acq = qOrdinalKnowledgeGradient(model, terminal_set=_terminal(), num_samples=16)
    with pytest.raises(AssertionError):
        acq(torch.rand(2, 2, 1, dtype=torch.double))


def test_ordinal_kg_runs_optimize_acqf_q1() -> None:
    model = _CorrelatedOrdinalLatentModel()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    acq = qOrdinalKnowledgeGradient(
        model,
        terminal_set=torch.linspace(0.0, 1.0, 9, dtype=torch.double).unsqueeze(-1),
        utility_values=torch.tensor([0.0, 0.5, 2.0, 5.0], dtype=torch.double),
        num_samples=32,
        seed=29,
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
