from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.posteriors.posterior import Posterior
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning,
)
from torch import Tensor, nn

from bochan.models.hybrid import (
    HybridMultiOutputModel,
    OutputSpec,
    TaskAwareHybridPosterior,
)


class _DiagonalPosterior(Posterior):
    def __init__(self, mean: Tensor, variance: Tensor) -> None:
        self._mean = mean
        self._variance = variance

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def device(self) -> torch.device:
        return self._mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self._mean.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, max(0, self._mean.ndim - 2))

    def _extended_shape(self, sample_shape: torch.Size | None = None) -> torch.Size:
        sample_shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        return sample_shape + self._mean.shape

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        sample_shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        base = torch.randn(
            sample_shape + self.base_sample_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.rsample_from_base_samples(sample_shape, base)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        target = torch.Size(sample_shape) + self._mean.shape
        return self._mean.expand(target) + self._variance.sqrt().expand(target) * base_samples


class _RegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.train_inputs = (torch.zeros(4, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(4, 1, dtype=torch.double)

    def posterior(self, X: Tensor, **kwargs) -> Posterior:
        del kwargs
        mean = X[..., :1]
        variance = torch.full_like(mean, 0.04)
        return _DiagonalPosterior(mean, variance)


class _BinaryLikelihood(nn.Module):
    def forward(self, latent: Tensor):
        return torch.distributions.Bernoulli(probs=torch.sigmoid(latent))


class _BinaryModel(nn.Module):
    def __init__(self, latent_variance: float = 0.04) -> None:
        super().__init__()
        self.likelihood = _BinaryLikelihood()
        self.latent_variance = float(latent_variance)
        self.train_inputs = (torch.zeros(4, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(4, dtype=torch.double)

    def latent_posterior(self, X: Tensor) -> Posterior:
        mean = torch.zeros_like(X[..., :1])
        variance = torch.full_like(mean, self.latent_variance)
        return _DiagonalPosterior(mean, variance)

    def posterior(self, X: Tensor, **kwargs):
        del kwargs
        probability = torch.full_like(X[..., :1], 0.5)
        return SimpleNamespace(
            mean=probability,
            variance=probability * (1.0 - probability),
        )


class _OrdinalLikelihood(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "cutpoints",
            torch.tensor([-0.5, 0.5], dtype=torch.double),
        )

    def class_probs_from_f(self, latent: Tensor) -> Tensor:
        cuts = self.cutpoints.to(device=latent.device, dtype=latent.dtype)
        cdf = torch.sigmoid(cuts - latent.unsqueeze(-1))
        return torch.cat(
            [cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1.0 - cdf[..., -1:]],
            dim=-1,
        )


class _OrdinalModel(nn.Module):
    def __init__(self, latent_variance: float = 0.04) -> None:
        super().__init__()
        self.ordinal_likelihood = _OrdinalLikelihood()
        self.latent_variance = float(latent_variance)
        self.train_inputs = (torch.zeros(4, 2, dtype=torch.double),)
        self.train_targets = torch.zeros(4, dtype=torch.double)

    def latent_posterior(self, X: Tensor) -> Posterior:
        mean = torch.zeros_like(X[..., :1])
        variance = torch.full_like(mean, self.latent_variance)
        return _DiagonalPosterior(mean, variance)

    def class_probs(self, X: Tensor) -> Tensor:
        latent = torch.zeros_like(X[..., 0])
        return self.ordinal_likelihood.class_probs_from_f(latent)


def _binary_hybrid() -> HybridMultiOutputModel:
    return HybridMultiOutputModel(
        [
            OutputSpec(name="property", task_type="regression", model=_RegressionModel()),
            OutputSpec(name="feasible", task_type="binary", model=_BinaryModel()),
        ]
    )


def test_hybrid_binary_posterior_uses_epistemic_probability_variance() -> None:
    posterior = _binary_hybrid().posterior(torch.zeros(3, 2, dtype=torch.double))

    assert isinstance(posterior, TaskAwareHybridPosterior)
    assert posterior.mean.shape == torch.Size([3, 2])
    assert posterior.variance.shape == torch.Size([3, 2])
    assert torch.all(posterior.variance[..., 1] < 0.01)
    assert torch.all(posterior.variance[..., 1] < 0.25)

    samples = SobolQMCNormalSampler(
        sample_shape=torch.Size([128]),
        seed=0,
    )(posterior)
    assert samples.shape == torch.Size([128, 3, 2])
    assert torch.all(samples[..., 1] >= 0.0)
    assert torch.all(samples[..., 1] <= 1.0)


def test_hybrid_ordinal_posterior_samples_expected_utility_in_range() -> None:
    model = HybridMultiOutputModel(
        [
            OutputSpec(name="property", task_type="regression", model=_RegressionModel()),
            OutputSpec(
                name="grade",
                task_type="ordinal",
                model=_OrdinalModel(),
                utility_values=[0.0, 1.0, 2.0],
            ),
        ]
    )
    posterior = model.posterior(torch.zeros(4, 2, dtype=torch.double))

    assert isinstance(posterior, TaskAwareHybridPosterior)
    assert torch.all(posterior.variance[..., 1] < 0.1)

    samples = posterior.rsample(torch.Size([128]))
    assert samples.shape == torch.Size([128, 4, 2])
    assert torch.all(samples[..., 1] >= 0.0)
    assert torch.all(samples[..., 1] <= 2.0)


def test_hybrid_task_aware_posterior_runs_ehvi() -> None:
    model = _binary_hybrid()
    dtype = torch.double
    ref_point = torch.tensor([-0.5, -0.1], dtype=dtype)
    observed = torch.tensor(
        [[0.0, 0.4], [0.4, 0.55]],
        dtype=dtype,
    )
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=observed,
    )
    acquisition = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=0),
    )
    X = torch.tensor(
        [[[0.2, 0.0]], [[0.7, 0.0]]],
        dtype=dtype,
    )

    values = acquisition(X)

    assert values.shape == torch.Size([2])
    assert torch.isfinite(values).all()
