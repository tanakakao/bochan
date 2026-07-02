from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

from bochan.acquisition.multiclass.active_learning import (
    qMultiOutputMulticlassBALD,
)


class _DeterministicMultiOutputMulticlassModel(Module):
    """Return two multiclass probability tensors with preserved t-batch axes."""

    num_outputs = 2

    def class_probs_list(self, X: Tensor) -> list[Tensor]:
        logits_0 = torch.stack(
            [
                X[..., 0],
                X[..., 1],
                -(X[..., 0] + X[..., 1]),
            ],
            dim=-1,
        )
        logits_1 = torch.stack(
            [
                -X[..., 0],
                X[..., 0] - X[..., 1],
                X[..., 1],
            ],
            dim=-1,
        )
        return [
            torch.softmax(logits_0, dim=-1),
            torch.softmax(logits_1, dim=-1),
        ]


def test_bald_preserves_batch_when_num_samples_equals_batch_size() -> None:
    batch_size = 32
    q = 3
    num_samples = 32
    X = torch.rand(batch_size, q, 2, dtype=torch.double)
    acquisition = qMultiOutputMulticlassBALD(
        model=_DeterministicMultiOutputMulticlassModel(),
        num_samples=num_samples,
    )

    samples = acquisition._sample_probs(X, num_samples=num_samples)
    scores = acquisition._pointwise_bald_per_output(X)
    values = acquisition(X)

    assert samples.shape == torch.Size([num_samples, batch_size, q, 2, 3])
    assert scores.shape == torch.Size([batch_size, q, 2])
    assert values.shape == torch.Size([batch_size])
    assert torch.isfinite(values).all()


def test_bald_preserves_distinct_sample_and_batch_sizes() -> None:
    batch_size = 7
    q = 3
    num_samples = 11
    X = torch.rand(batch_size, q, 2, dtype=torch.double)
    acquisition = qMultiOutputMulticlassBALD(
        model=_DeterministicMultiOutputMulticlassModel(),
        num_samples=num_samples,
    )

    samples = acquisition._sample_probs(X, num_samples=num_samples)
    values = acquisition(X)

    assert samples.shape == torch.Size([num_samples, batch_size, q, 2, 3])
    assert values.shape == torch.Size([batch_size])
