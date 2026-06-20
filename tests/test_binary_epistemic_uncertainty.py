from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from botorch.models.model import Model
from torch import Tensor, nn
from torch.distributions import Bernoulli

from bochan.acquisition.binary.active_learning import qBinaryProbabilityVariance
from bochan.acquisition.binary.bayesian_optimization import (
    qBinaryExpectedImprovement,
    qBinaryProbabilityOfImprovement,
)
from bochan.acquisition.binary.epistemic import binary_probability_moments


class _TensorPosterior:
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean
        self.variance = std.square()
        self.std = std

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        sample_shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        n = math.prod(sample_shape) if len(sample_shape) > 0 else 1
        z = torch.linspace(
            -math.sqrt(3.0),
            math.sqrt(3.0),
            n,
            dtype=self.mean.dtype,
            device=self.mean.device,
        )
        z = z.reshape(*sample_shape, *((1,) * self.mean.ndim))
        return self.mean.expand(*sample_shape, *self.mean.shape) + z * self.std


class _ProbabilityPosterior:
    def __init__(self, probability: Tensor) -> None:
        self.mean = probability
        self.variance = probability * (1.0 - probability)


class _IdentityBernoulliLikelihood(nn.Module):
    def forward(self, function_samples: Tensor) -> Bernoulli:
        return Bernoulli(probs=function_samples.clamp(1e-6, 1.0 - 1e-6))


class _EpistemicBinaryModel(Model):
    """Test model whose first feature is probability mean and second is latent std."""

    def __init__(self, default_std: float = 0.02) -> None:
        super().__init__()
        self.likelihood = _IdentityBernoulliLikelihood()
        self.default_std = float(default_std)
        self.train_inputs = (
            torch.tensor([[0.2, default_std], [0.8, default_std]], dtype=torch.double),
        )
    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def _mean_std(self, X: Tensor) -> tuple[Tensor, Tensor]:
        mean = X[..., :1].clamp(0.02, 0.98)
        if X.shape[-1] >= 2:
            std = X[..., 1:2].abs().clamp_min(1e-5)
        else:
            std = torch.full_like(mean, self.default_std)
        return mean, std

    def latent_posterior(self, X: Tensor) -> _TensorPosterior:
        mean, std = self._mean_std(X)
        return _TensorPosterior(mean, std)

    def probability_posterior(self, X: Tensor) -> _ProbabilityPosterior:
        mean, _ = self._mean_std(X)
        return _ProbabilityPosterior(mean)

    def posterior(self, X: Tensor, **kwargs) -> _ProbabilityPosterior:
        return self.probability_posterior(X)


def _sampler(n: int = 257):
    return SimpleNamespace(sample_shape=torch.Size([n]))


def test_ei_and_pi_prefer_higher_probability_for_equal_epistemic_spread() -> None:
    model = _EpistemicBinaryModel(default_std=0.02)
    X = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)

    ei = qBinaryExpectedImprovement(
        model,
        best_f=0.5,
        sampler=_sampler(),
    )
    pi = qBinaryProbabilityOfImprovement(
        model,
        best_f=0.5,
        tau=0.02,
        sampler=_sampler(),
    )

    ei_value = ei(X)
    pi_value = pi(X)

    assert ei_value[1] > ei_value[0]
    assert pi_value[1] > pi_value[0]
    assert ei_value[0] < 1e-6


def test_explicit_binary_best_f_one_is_clamped_inside_probability_domain() -> None:
    model = _EpistemicBinaryModel()
    acquisition = qBinaryExpectedImprovement(
        model,
        best_f=1.0,
        best_f_margin=0.02,
        sampler=_sampler(),
    )

    assert torch.allclose(
        acquisition.best_f,
        torch.tensor(0.98, dtype=torch.double),
    )


def test_probability_variance_uses_epistemic_not_bernoulli_variance() -> None:
    model = _EpistemicBinaryModel()
    X = torch.tensor(
        [
            [[0.10, 0.02]],
            [[0.50, 0.02]],
        ],
        dtype=torch.double,
    )
    acquisition = qBinaryProbabilityVariance(model, num_samples=257)

    value = acquisition(X)

    # Equal latent spread should give nearly equal epistemic variance.  The old
    # p(1-p) score would make the second value roughly 2.8 times larger.
    assert torch.allclose(value[0], value[1], rtol=0.08, atol=1e-5)


def test_probability_variance_increases_with_latent_epistemic_spread() -> None:
    model = _EpistemicBinaryModel()
    X = torch.tensor(
        [
            [[0.50, 0.01]],
            [[0.50, 0.15]],
        ],
        dtype=torch.double,
    )
    acquisition = qBinaryProbabilityVariance(model, num_samples=257)

    value = acquisition(X)

    assert value[1] > 20.0 * value[0]


def test_probability_variance_decomposition_is_consistent() -> None:
    model = _EpistemicBinaryModel()
    X = torch.tensor([[0.5, 0.1]], dtype=torch.double)

    mean, epistemic, aleatoric, total = binary_probability_moments(
        model,
        X,
        num_samples=1001,
    )

    assert torch.allclose(total, mean * (1.0 - mean))
    assert torch.allclose(total, epistemic + aleatoric, atol=2e-3)
