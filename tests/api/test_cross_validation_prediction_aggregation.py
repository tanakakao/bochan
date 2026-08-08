"""Regression tests for one-to-many cross-validation predictions."""

import torch

from bochan.api.cross_validation import (
    _aggregate_expanded_moments,
    _aggregate_expanded_probabilities,
)


def test_aggregate_expanded_moments_uses_total_variance() -> None:
    """Perturbation rows collapse to nominal rows with total variance preserved."""

    mean = torch.tensor([[1.0], [3.0], [2.0], [4.0]], dtype=torch.double)
    variance = torch.ones_like(mean)

    aggregated_mean, aggregated_variance = _aggregate_expanded_moments(
        mean,
        variance,
        n_rows=2,
    )

    assert torch.allclose(
        aggregated_mean,
        torch.tensor([[2.0], [3.0]], dtype=torch.double),
    )
    assert aggregated_variance is not None
    assert torch.allclose(
        aggregated_variance,
        torch.tensor([[2.0], [2.0]], dtype=torch.double),
    )


def test_aggregate_expanded_probabilities_averages_per_nominal_row() -> None:
    """Class probabilities are averaged only across each row's expansion."""

    probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.3, 0.7],
        ],
        dtype=torch.double,
    )

    aggregated = _aggregate_expanded_probabilities(probabilities, n_rows=2)

    assert torch.allclose(
        aggregated,
        torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.double),
    )
