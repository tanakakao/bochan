from __future__ import annotations

import torch

from bochan.models.transforms.posterior.classification import (
    aggregate_perturbed_posterior,
    aggregate_perturbed_posterior_chunked,
)


class _FakePosterior:
    def __init__(self, mean: torch.Tensor, event_ndim: int) -> None:
        self.mean = mean
        self.variance = torch.full_like(mean, 0.25)
        self._event_ndim = int(event_ndim)

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size(self.mean.shape[-self._event_ndim :])


class _ExpandedPosteriorModel:
    def __init__(
        self,
        *,
        n_w: int,
        num_outputs: int = 2,
        num_classes: int = 3,
        singleton_output: bool = False,
    ) -> None:
        self.n_w = int(n_w)
        self.num_outputs = int(num_outputs)
        self.num_classes = int(num_classes)
        self.singleton_output = bool(singleton_output)

    def posterior(self, X: torch.Tensor, **kwargs) -> _FakePosterior:
        del kwargs
        batch_shape = X.shape[:-2]
        q = int(X.shape[-2])
        expanded_q = q * self.n_w

        values = torch.arange(
            int(torch.tensor(batch_shape).prod().item())
            * expanded_q
            * self.num_outputs
            * self.num_classes
            if len(batch_shape) > 0
            else expanded_q * self.num_outputs * self.num_classes,
            dtype=X.dtype,
            device=X.device,
        )

        if self.singleton_output:
            mean = values[: expanded_q * self.num_classes].reshape(
                *batch_shape,
                expanded_q,
                1,
                self.num_classes,
            )
        else:
            mean = values.reshape(
                *batch_shape,
                expanded_q,
                self.num_outputs,
                self.num_classes,
            )
        return _FakePosterior(mean=mean, event_ndim=3)


class _ExpandedOrdinalModel:
    def __init__(self, *, n_w: int, num_outputs: int = 2) -> None:
        self.n_w = int(n_w)
        self.num_outputs = int(num_outputs)

    def posterior(self, X: torch.Tensor, **kwargs) -> _FakePosterior:
        del kwargs
        batch_shape = X.shape[:-2]
        q = int(X.shape[-2])
        expanded_q = q * self.n_w
        mean = torch.arange(
            expanded_q * self.num_outputs,
            dtype=X.dtype,
            device=X.device,
        ).reshape(*batch_shape, expanded_q, self.num_outputs)
        return _FakePosterior(mean=mean, event_ndim=2)


def test_aggregates_kronecker_multiclass_q_output_class_layout() -> None:
    q = 4
    n_w = 3
    model = _ExpandedPosteriorModel(n_w=n_w)
    X = torch.zeros(q, 2, dtype=torch.double)

    posterior = aggregate_perturbed_posterior(
        model=model,
        X=X,
        n_w=n_w,
    )

    assert posterior.q_dim == 0
    assert posterior.mean.shape == torch.Size([q, 2, 3])
    assert posterior.variance.shape == torch.Size([q, 2, 3])
    assert posterior.mean_per_w.shape == torch.Size([q, n_w, 2, 3])
    assert posterior.variance_per_w.shape == torch.Size([q, n_w, 2, 3])


def test_aggregates_t_batch_multiclass_layout() -> None:
    batch_size = 5
    q = 4
    n_w = 2
    model = _ExpandedPosteriorModel(n_w=n_w)
    X = torch.zeros(batch_size, q, 2, dtype=torch.double)

    posterior = aggregate_perturbed_posterior(
        model=model,
        X=X,
        n_w=n_w,
    )

    assert posterior.q_dim == 1
    assert posterior.mean.shape == torch.Size([batch_size, q, 2, 3])
    assert posterior.mean_per_w.shape == torch.Size(
        [batch_size, q, n_w, 2, 3]
    )


def test_singleton_multiclass_output_is_squeezed() -> None:
    q = 4
    n_w = 2
    model = _ExpandedPosteriorModel(
        n_w=n_w,
        num_outputs=1,
        singleton_output=True,
    )
    X = torch.zeros(q, 2, dtype=torch.double)

    posterior = aggregate_perturbed_posterior(
        model=model,
        X=X,
        n_w=n_w,
    )

    assert posterior.q_dim == 0
    assert posterior.mean.shape == torch.Size([q, 3])
    assert posterior.mean_per_w.shape == torch.Size([q, n_w, 3])


def test_existing_q_output_layout_remains_supported() -> None:
    q = 4
    n_w = 3
    model = _ExpandedOrdinalModel(n_w=n_w)
    X = torch.zeros(q, 2, dtype=torch.double)

    posterior = aggregate_perturbed_posterior(
        model=model,
        X=X,
        n_w=n_w,
    )

    assert posterior.q_dim == 0
    assert posterior.mean.shape == torch.Size([q, 2])
    assert posterior.mean_per_w.shape == torch.Size([q, n_w, 2])


def test_chunked_aggregation_concatenates_along_inferred_q_axis() -> None:
    q = 10
    n_w = 2
    model = _ExpandedPosteriorModel(n_w=n_w)
    X = torch.zeros(q, 2, dtype=torch.double)

    posterior = aggregate_perturbed_posterior_chunked(
        model=model,
        X=X,
        n_w=n_w,
        chunk_size=4,
    )

    assert posterior.q_dim == 0
    assert posterior.mean.shape == torch.Size([q, 2, 3])
    assert posterior.variance.shape == torch.Size([q, 2, 3])
