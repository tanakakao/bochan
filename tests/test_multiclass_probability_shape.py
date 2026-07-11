from __future__ import annotations

import torch

from bochan.acquisition.multiclass.bayesian_optimization import (
    MulticlassTargetProbabilityObjective,
)
from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation import (
    validate_hypervolume_objective_q,
)
from bochan.models.components.multiclass import MulticlassProbsPosterior


class _FakeLatentPosterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean
        self.base_sample_shape = torch.Size(mean.shape)
        self.batch_range = (0, max(0, mean.ndim - 2))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.mean.expand(*sample_shape, *self.mean.shape)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: torch.Tensor,
    ) -> torch.Tensor:
        del base_samples
        return self.rsample(sample_shape)


def test_multiclass_posterior_removes_gpytorch_singleton_output_axis() -> None:
    batch_size = 4
    num_classes = 2
    q = 3
    latent_mean = torch.randn(batch_size, num_classes, q, 1, dtype=torch.double)
    posterior = MulticlassProbsPosterior(
        latent_posterior=_FakeLatentPosterior(latent_mean),
        num_classes=num_classes,
    )

    assert posterior.mean.shape == torch.Size([batch_size, q, num_classes])
    samples = posterior.rsample(torch.Size([5]))
    assert samples.shape == torch.Size([5, batch_size, q, num_classes])
    torch.testing.assert_close(
        posterior.mean.sum(dim=-1),
        torch.ones(batch_size, q, dtype=torch.double),
    )


def test_multiclass_objective_handles_outputs_equal_to_classes() -> None:
    sample_shape = 5
    batch_size = 4
    q = 3
    num_outputs = 2
    num_classes = 2
    probabilities = torch.softmax(
        torch.randn(
            sample_shape,
            batch_size,
            q,
            num_outputs,
            num_classes,
            dtype=torch.double,
        ),
        dim=-1,
    )
    objective = MulticlassTargetProbabilityObjective(num_outputs=num_outputs)

    values = objective(probabilities)

    assert values.shape == torch.Size(
        [sample_shape, batch_size, q, num_outputs]
    )


def test_multiclass_objective_removes_singleton_between_q_and_outputs() -> None:
    sample_shape = 5
    batch_size = 4
    q = 3
    num_outputs = 2
    num_classes = 2
    probabilities = torch.softmax(
        torch.randn(
            sample_shape,
            batch_size,
            q,
            1,
            num_outputs,
            num_classes,
            dtype=torch.double,
        ),
        dim=-1,
    )
    objective = MulticlassTargetProbabilityObjective(num_outputs=num_outputs)

    values = objective(probabilities)

    assert values.shape == torch.Size(
        [sample_shape, batch_size, q, num_outputs]
    )


def test_multiclass_objective_restores_q1_from_raw_X() -> None:
    sample_shape = 128
    batch_size = 32
    num_outputs = 2
    num_classes = 3
    probabilities_without_q = torch.softmax(
        torch.randn(
            sample_shape,
            batch_size,
            num_outputs,
            num_classes,
            dtype=torch.double,
        ),
        dim=-1,
    )
    X = torch.rand(batch_size, 1, 4, dtype=torch.double)
    objective = MulticlassTargetProbabilityObjective(num_outputs=num_outputs)

    values = objective(probabilities_without_q, X=X)

    assert values.shape == torch.Size(
        [sample_shape, batch_size, 1, num_outputs]
    )
    validate_hypervolume_objective_q(values, X)


def test_multiclass_objective_does_not_duplicate_existing_q1() -> None:
    sample_shape = 8
    batch_size = 6
    num_outputs = 2
    num_classes = 3
    probabilities = torch.softmax(
        torch.randn(
            sample_shape,
            batch_size,
            1,
            num_outputs,
            num_classes,
            dtype=torch.double,
        ),
        dim=-1,
    )
    X = torch.rand(batch_size, 1, 4, dtype=torch.double)
    objective = MulticlassTargetProbabilityObjective(num_outputs=num_outputs)

    values = objective(probabilities, X=X)

    assert values.shape == torch.Size(
        [sample_shape, batch_size, 1, num_outputs]
    )
    validate_hypervolume_objective_q(values, X)
