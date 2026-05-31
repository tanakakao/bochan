from __future__ import annotations

from typing import Optional

import torch
from botorch.posteriors.posterior import Posterior
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor


class HybridPosterior(Posterior):
    """Hybrid multi-output 用の軽量 posterior。

    `mean` / `variance` は objective scale の Tensor として扱う。
    出力間の共分散は持たず、`rsample` では各出力を独立な正規 proxy として
    再パラメータ化サンプルする。

    Shape:
        mean:     `batch_shape x q x m`
        variance: `batch_shape x q x m`
    """

    def __init__(
        self,
        mean: Tensor,
        variance: Optional[Tensor] = None,
        *,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()

        if not torch.is_tensor(mean):
            raise TypeError(f"mean must be a Tensor. Got {type(mean)}.")
        if mean.ndim < 2:
            raise ValueError(
                "HybridPosterior expects mean with at least shape (..., q, m). "
                f"Got mean.shape={tuple(mean.shape)}."
            )

        if variance is None:
            variance = torch.zeros_like(mean)
        else:
            if not torch.is_tensor(variance):
                raise TypeError(f"variance must be a Tensor. Got {type(variance)}.")
            variance = variance.to(device=mean.device, dtype=mean.dtype)
            if variance.shape != mean.shape:
                try:
                    variance = variance.expand_as(mean)
                except RuntimeError as e:
                    raise ValueError(
                        "variance must be broadcastable to mean.shape. "
                        f"mean.shape={tuple(mean.shape)}, "
                        f"variance.shape={tuple(variance.shape)}."
                    ) from e

        self._mean = mean
        self._variance = variance.clamp_min(0.0)
        self._eps = float(eps)

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
        # 既存の multi-output posterior 実装に合わせて full tensor shape を返す。
        return self._mean.shape

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._mean.shape

    @property
    def batch_range(self) -> tuple[int, int]:
        # 最後の 2 次元を q, m とみなし、それ以前を t-batch とする。
        return (0, max(0, self._mean.ndim - 2))

    def _extended_shape(
        self,
        sample_shape: Optional[torch.Size] = None,
    ) -> torch.Size:
        if sample_shape is None:
            sample_shape = torch.Size()
        return torch.Size(sample_shape) + self._mean.shape

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        sample_shape = torch.Size(sample_shape)

        if base_samples is None:
            base_samples = torch.randn(
                sample_shape + self._mean.shape,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            base_samples = base_samples.to(device=self.device, dtype=self.dtype)

        return self.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """BoTorch sampler 用の再パラメータ化サンプル。"""

        sample_shape = torch.Size(sample_shape)
        target_shape = sample_shape + self._mean.shape

        base_samples = base_samples.to(device=self.device, dtype=self.dtype)
        if base_samples.shape != target_shape:
            try:
                base_samples = base_samples.expand(target_shape)
            except RuntimeError as e:
                raise RuntimeError(
                    "base_samples must be broadcastable to "
                    f"{tuple(target_shape)}, got {tuple(base_samples.shape)}."
                ) from e

        mean = self._mean.expand(target_shape)
        std = self._variance.clamp_min(self._eps).sqrt().expand(target_shape)
        return mean + std * base_samples

    def sample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)


@GetSampler.register(HybridPosterior)
def _get_sampler_hybrid_posterior(
    posterior: HybridPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    """BoTorch の自動 sampler 解決に HybridPosterior を登録する。

    qNEI などは `prune_baseline=True` の初期処理で `get_sampler(posterior, ...)`
    を呼ぶため、独自 posterior は dispatcher への登録が必要になる。
    """

    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


__all__ = ["HybridPosterior"]
