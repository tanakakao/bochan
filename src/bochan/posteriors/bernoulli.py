import torch
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors import Posterior, GPyTorchPosterior
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.base import MCSampler


import torch
from torch import Tensor
from botorch.posteriors.posterior import Posterior


class SimpleBernoulliPosterior(Posterior):
    """
    single-output classification 用の簡易 posterior。

    想定 shape:
        mean / variance / probs: [..., q, 1]

    初期化方法:
        1. probs を直接渡す
            SimpleBernoulliPosterior(probs=probs)

        2. mean / variance を明示して渡す
            SimpleBernoulliPosterior(mean=mean, variance=variance)

    メモ:
        probs を渡した場合は
            mean = probs
            variance = probs * (1 - probs)
        とする。
    """

    def __init__(
        self,
        mean: Tensor | None = None,
        variance: Tensor | None = None,
        probs: Tensor | None = None,
    ) -> None:
        if probs is not None:
            if mean is not None or variance is not None:
                raise ValueError(
                    "Pass either `probs` or (`mean`, `variance`), not both."
                )
            probs = probs.clamp(1e-9, 1 - 1e-9)
            mean = probs
            variance = probs * (1.0 - probs)

        if mean is None or variance is None:
            raise ValueError(
                "You must pass either `probs`, or both `mean` and `variance`."
            )

        if mean.shape != variance.shape:
            raise ValueError(
                f"mean.shape {tuple(mean.shape)} != variance.shape {tuple(variance.shape)}"
            )
        if mean.ndim < 2:
            raise ValueError(
                f"Expected mean to have at least 2 dims (..., q, 1), got {tuple(mean.shape)}"
            )
        if mean.shape[-1] != 1:
            raise ValueError(
                f"Expected last dim to be 1 for single-output posterior, got {tuple(mean.shape)}"
            )

        self._mean = mean
        self._variance = variance.clamp_min(1e-9)
        self._probs = mean.clamp(1e-9, 1 - 1e-9)
        self._device = mean.device
        self._dtype = mean.dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def variance(self) -> Tensor:
        return self._variance

    @property
    def probs(self) -> Tensor:
        return self._probs

    def _extended_shape(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Size:
        return sample_shape + self._mean.shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def _is_mt(self) -> bool:
        return False

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, max(0, self._mean.dim() - 2))

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()

        shape = self._extended_shape(sample_shape)
        base_samples = torch.randn(
            shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        target_shape = self._extended_shape(sample_shape)
        if base_samples.shape != target_shape:
            raise RuntimeError(
                f"base_samples.shape {tuple(base_samples.shape)} != expected {tuple(target_shape)}"
            )

        mean = self._mean.expand(target_shape)
        std = self._variance.sqrt().expand(target_shape)
        return mean + std * base_samples


@GetSampler.register(SimpleBernoulliPosterior)
def get_sampler_for_simple_bernoulli(
    posterior: SimpleBernoulliPosterior,
    sample_shape: torch.Size,
    seed=None,
):
    class SimpleBernoulliSampler:
        def __call__(self, posterior: Posterior) -> torch.Tensor:
            return posterior.rsample(sample_shape)

    return SimpleBernoulliSampler()