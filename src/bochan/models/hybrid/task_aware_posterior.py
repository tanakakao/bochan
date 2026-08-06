from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from botorch.posteriors.posterior import Posterior
from botorch.sampling.get_sampler import GetSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from .posterior import HybridPosterior


@dataclass(frozen=True)
class HybridPosteriorComponent:
    """One scalar hybrid output and its task-aware sampling rule."""

    mean: Tensor
    variance: Tensor
    posterior: Posterior | None = None
    sample_transform: Callable[[Tensor], Tensor] | None = None
    name: str = "output"


@dataclass(frozen=True)
class _ComponentLayout:
    component: HybridPosteriorComponent
    source_shape: torch.Size
    event_shape: torch.Size
    event_size: int
    use_source_posterior: bool


class TaskAwareHybridPosterior(HybridPosterior):
    """Hybrid posterior that samples each task on its native decision scale.

    The public mean / variance keep the usual ``batch x q x m`` contract. Base
    samples are flattened only across each output's event dimensions, while the
    public t-batch dimensions remain visible to BoTorch's samplers. Regression
    outputs can therefore retain their source posterior covariance, and binary
    or ordinal outputs can transform latent samples into bounded probability or
    expected-utility samples before EHVI / NEHVI / NParEGO consume them.
    """

    def __init__(
        self,
        mean: Tensor,
        variance: Tensor,
        *,
        components: Sequence[HybridPosteriorComponent],
        eps: float = 1e-9,
    ) -> None:
        super().__init__(mean=mean, variance=variance, eps=eps)
        if len(components) != int(mean.shape[-1]):
            raise ValueError(
                "components must match the hybrid output dimension. "
                f"Got {len(components)} components for mean.shape={tuple(mean.shape)}."
            )

        self.components = tuple(components)
        self._layouts = tuple(self._make_layout(component) for component in components)
        total_event_size = sum(layout.event_size for layout in self._layouts)
        if total_event_size <= 0:
            raise ValueError("TaskAwareHybridPosterior requires at least one base event.")
        self._base_sample_shape = self.batch_shape + torch.Size([total_event_size])

    def _make_layout(self, component: HybridPosteriorComponent) -> _ComponentLayout:
        expected_shape = self.batch_shape + torch.Size([self._mean.shape[-2]])
        if component.mean.shape != expected_shape:
            try:
                component.mean.expand(expected_shape)
                component.variance.expand(expected_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"{component.name}: component moments must be broadcastable to "
                    f"{tuple(expected_shape)}, got mean={tuple(component.mean.shape)}, "
                    f"variance={tuple(component.variance.shape)}."
                ) from exc

        posterior = component.posterior
        use_source = False
        source_shape = torch.Size()
        event_shape = torch.Size()
        if posterior is not None:
            raw_shape = getattr(posterior, "base_sample_shape", None)
            has_base_sampler = callable(
                getattr(posterior, "rsample_from_base_samples", None)
            )
            if raw_shape is not None and has_base_sampler:
                raw_shape = torch.Size(raw_shape)
                batch_ndim = len(self.batch_shape)
                source_batch = raw_shape[:batch_ndim]
                source_event = raw_shape[batch_ndim:]
                public_q = int(self._mean.shape[-2])

                # InputPerturbation can expose q*n_w latent events while the
                # public Hybrid posterior has already aggregated them to q.
                # Such a source posterior cannot share one base-sample layout
                # with the public posterior; use the transformed moments proxy
                # instead. This also avoids silently treating perturbation
                # replicas as additional candidates.
                if (
                    source_batch == self.batch_shape
                    and len(source_event) > 0
                    and int(math.prod(source_event)) == public_q
                ):
                    source_shape = raw_shape
                    event_shape = source_event
                    use_source = True

        if not use_source:
            event_shape = torch.Size([self._mean.shape[-2]])
            source_shape = self.batch_shape + event_shape

        return _ComponentLayout(
            component=component,
            source_shape=source_shape,
            event_shape=event_shape,
            event_size=int(math.prod(event_shape)),
            use_source_posterior=use_source,
        )

    @property
    def base_sample_shape(self) -> torch.Size:
        return self._base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return (0, len(self.batch_shape))

    def _extended_shape(
        self,
        sample_shape: torch.Size | None = None,
    ) -> torch.Size:
        sample_shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        return sample_shape + self._mean.shape

    @staticmethod
    def _normalize_component_samples(
        samples: Tensor,
        *,
        sample_shape: torch.Size,
        component: HybridPosteriorComponent,
    ) -> Tensor:
        target_shape = sample_shape + component.mean.shape
        out = samples

        while out.ndim > len(target_shape) and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.shape == target_shape:
            return out

        # Deep / fully-Bayesian wrappers may expose one extra leading model or
        # MC axis. Their public hybrid moments already average that axis, so the
        # task-aware sampler applies the same reduction while preserving the
        # requested acquisition sample dimensions.
        sample_ndim = len(sample_shape)
        while out.ndim > len(target_shape):
            out = out.mean(dim=sample_ndim)
            if out.shape == target_shape:
                return out

        if out.numel() == math.prod(target_shape):
            return out.reshape(target_shape)
        try:
            return out.expand(target_shape)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{component.name}: transformed samples could not be aligned to "
                f"{tuple(target_shape)}; got {tuple(samples.shape)}."
            ) from exc

    def _draw_component(
        self,
        layout: _ComponentLayout,
        *,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        component = layout.component
        if layout.use_source_posterior:
            source_base = base_samples.reshape(sample_shape + layout.source_shape)
            raw_samples = component.posterior.rsample_from_base_samples(
                sample_shape=sample_shape,
                base_samples=source_base,
            )
        else:
            target = sample_shape + component.mean.shape
            public_events = int(component.mean.shape[-1])
            standard_normal = base_samples[..., :public_events].reshape(target)
            raw_samples = (
                component.mean.expand(target)
                + component.variance.clamp_min(0.0).sqrt().expand(target)
                * standard_normal
            )

        transformed = (
            component.sample_transform(raw_samples)
            if layout.use_source_posterior
            and component.sample_transform is not None
            else raw_samples
        )
        return self._normalize_component_samples(
            transformed,
            sample_shape=sample_shape,
            component=component,
        )

    def rsample(
        self,
        sample_shape: torch.Size | None = None,
        base_samples: Tensor | None = None,
    ) -> Tensor:
        sample_shape = torch.Size() if sample_shape is None else torch.Size(sample_shape)
        target = sample_shape + self.base_sample_shape
        if base_samples is None:
            base_samples = torch.randn(
                target,
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
        sample_shape = torch.Size(sample_shape)
        target = sample_shape + self.base_sample_shape
        base_samples = base_samples.to(device=self.device, dtype=self.dtype)
        if base_samples.shape != target:
            try:
                base_samples = base_samples.expand(target)
            except RuntimeError as exc:
                raise RuntimeError(
                    "base_samples must be broadcastable to "
                    f"{tuple(target)}, got {tuple(base_samples.shape)}."
                ) from exc

        outputs = []
        offset = 0
        for layout in self._layouts:
            next_offset = offset + layout.event_size
            base_i = base_samples[..., offset:next_offset]
            outputs.append(
                self._draw_component(
                    layout,
                    sample_shape=sample_shape,
                    base_samples=base_i,
                )
            )
            offset = next_offset
        return torch.stack(outputs, dim=-1)

    def sample(self, sample_shape: torch.Size | None = None) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)


@GetSampler.register(TaskAwareHybridPosterior)
def _get_sampler_task_aware_hybrid_posterior(
    posterior: TaskAwareHybridPosterior,
    sample_shape: torch.Size,
    seed: int | None = None,
) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


__all__ = [
    "HybridPosteriorComponent",
    "TaskAwareHybridPosterior",
]
