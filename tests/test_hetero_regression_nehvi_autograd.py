from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from bochan.acquisition.regression.bayesian_optimization import (
    qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement,
)
from bochan.acquisition.regression.bayesian_optimization.hetero_multi_output_compat import (
    _AutogradSafeHeteroRegressionMCMultiOutputObjective,
    qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement as _CompatHeteroNEHVI,
)


class _LinearPosteriorModel(nn.Module):
    """Minimal differentiable posterior model for objective autograd tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.5, dtype=torch.double))

    def posterior(self, X: Tensor) -> SimpleNamespace:
        first = X.sum(dim=-1, keepdim=True) * self.weight
        mean = torch.cat([first, -first], dim=-1)
        return SimpleNamespace(mean=mean)


def _posterior_samples(model: _LinearPosteriorModel, X: Tensor) -> Tensor:
    return model.posterior(X).mean.unsqueeze(0)


def test_public_hetero_nehvi_uses_autograd_safe_compat_class() -> None:
    assert (
        qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement
        is _CompatHeteroNEHVI
    )


def test_hetero_nehvi_objective_detaches_baseline_cache() -> None:
    model = _LinearPosteriorModel()
    objective = _AutogradSafeHeteroRegressionMCMultiOutputObjective(
        base_objective=None,
        model=model,
        beta=1.0,
        noise_penalty=0.0,
    )
    X_baseline = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=torch.double,
    )
    samples = _posterior_samples(model, X_baseline)

    cached_objective = objective(samples=samples, X=X_baseline)

    assert samples.requires_grad
    assert not X_baseline.requires_grad
    assert not cached_objective.requires_grad
    assert cached_objective.grad_fn is None


def test_hetero_nehvi_objective_supports_repeated_candidate_backward() -> None:
    model = _LinearPosteriorModel()
    objective = _AutogradSafeHeteroRegressionMCMultiOutputObjective(
        base_objective=None,
        model=model,
        beta=1.0,
        noise_penalty=0.0,
    )
    X_baseline = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=torch.double,
    )
    cached_objective = objective(
        samples=_posterior_samples(model, X_baseline),
        X=X_baseline,
    )
    candidate = torch.tensor(
        [[0.5, 0.6], [0.7, 0.8]],
        dtype=torch.double,
        requires_grad=True,
    )

    for _ in range(2):
        candidate_objective = objective(
            samples=_posterior_samples(model, candidate),
            X=candidate,
        )
        loss = -(candidate_objective.sum() + cached_objective.sum())
        loss.backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()
        candidate.grad = None
        model.weight.grad = None
