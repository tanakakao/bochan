from __future__ import annotations

import torch

from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
    MulticlassTargetProbabilityObjective,
    compute_observed_multiclass_utility,
)


def test_multiclass_utility_ranges_are_applied_per_output() -> None:
    utility_values = [
        [0.0, 1.0, 2.0],
        [0.0, 10.0, 100.0],
    ]
    probabilities = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]]],
        dtype=torch.double,
    )
    objective = MulticlassTargetProbabilityObjective(
        num_outputs=2,
        utility_values=utility_values,
    )

    posterior_values = objective(probabilities)
    observed_values = compute_observed_multiclass_utility(
        torch.tensor([[2, 2], [1, 1]], dtype=torch.long),
        utility_values=utility_values,
    )

    torch.testing.assert_close(
        posterior_values,
        torch.tensor([[[2.0, 55.0]]], dtype=torch.double),
    )
    torch.testing.assert_close(
        observed_values,
        torch.tensor([[2.0, 100.0], [1.0, 10.0]]),
    )
