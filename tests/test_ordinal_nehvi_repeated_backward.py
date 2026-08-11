from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import nn

from bochan.acquisition.ordinal.bayesian_optimization import (
    multi_output,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qMultiOutputOrdinalUtilityObjective,
)
from bochan.models.ordinal.likelihood import OrdinalLogitLikelihood
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel
from bochan.models.wide_multitask_variants import WideMultiTaskOrdinalGPModel


def _ordinal_train_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    return train_X, train_Y


def _utility_values() -> list[torch.Tensor]:
    return [
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    ]


def _assert_nehvi_supports_optimizer_style_repeated_backward(
    model: nn.Module,
    train_X: torch.Tensor,
) -> None:
    model.eval()
    model.likelihood.eval()
    acquisition = qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=[-0.1, -0.1],
        X_baseline=train_X,
        utility_values=_utility_values(),
        sampler=SobolQMCNormalSampler(torch.Size([4]), seed=123),
        prune_baseline=False,
        cache_root=False,
    )
    candidate = torch.tensor(
        [[[0.25], [0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )

    for _ in range(2):
        value = acquisition(candidate)
        (-value.sum()).backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()

        with torch.no_grad():
            candidate.add_(0.01).clamp_(0.0, 1.0)
        candidate.grad = None


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


def test_kronecker_ordinal_nehvi_supports_repeated_torch_backward() -> None:
    train_X, train_Y = _ordinal_train_data()
    model = KroneckerMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )

    _assert_nehvi_supports_optimizer_style_repeated_backward(model, train_X)


def test_wide_multitask_ordinal_nehvi_supports_repeated_torch_backward() -> None:
    train_X, train_Y = _ordinal_train_data()
    model = WideMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=2,
        num_inducing=4,
    )

    _assert_nehvi_supports_optimizer_style_repeated_backward(model, train_X)
