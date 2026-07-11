from __future__ import annotations

import torch
from bochan.acquisition.ordinal.bayesian_optimization import (
    multi_output,
    qMultiOutputOrdinalUtilityObjective,
)
from bochan.likelihoods.ordinal import OrdinalLogitLikelihood
from torch import nn


def test_ordinal_probability_conversion_detaches_fitted_cutpoints() -> None:
    likelihood = OrdinalLogitLikelihood(num_classes=3).double()
    latent = torch.tensor([[-0.4, 0.8]], dtype=torch.double, requires_grad=True)

    with torch.no_grad():
        expected = likelihood.class_probs_from_f(latent.detach())
    actual = multi_output.ordinal_probs_from_latent(
        latent,
        likelihood,
        num_classes=3,
        link="auto",
    )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()

    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert likelihood.raw_gaps.grad is None


def test_cached_ordinal_utility_supports_repeated_backward() -> None:
    likelihoods = [
        OrdinalLogitLikelihood(num_classes=3).double(),
        OrdinalLogitLikelihood(num_classes=3).double(),
    ]
    objective = qMultiOutputOrdinalUtilityObjective(
        model=nn.Module(),
        ordinal_likelihoods=likelihoods,
        utility_values=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
    )

    baseline_samples = torch.zeros(16, 1, 3, 2, dtype=torch.double)
    cached_baseline_utility = objective(baseline_samples)

    assert not cached_baseline_utility.requires_grad

    for _ in range(2):
        candidate_samples = torch.randn(
            16,
            1,
            2,
            2,
            dtype=torch.double,
            requires_grad=True,
        )
        loss = objective(candidate_samples).sum() + cached_baseline_utility.sum()
        loss.backward()

        assert candidate_samples.grad is not None
        assert torch.isfinite(candidate_samples.grad).all()

    assert all(likelihood.raw_gaps.grad is None for likelihood in likelihoods)
