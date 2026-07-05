from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean
from botorch.sampling.normal import SobolQMCNormalSampler

from bochan.acquisition.objective import make_outcome_constraints
from bochan.acquisition.regression.bayesian_optimization import (
    qMultiOutputRegressionNParEGO,
)
from bochan.models.wide_multitask_compat import WideMultiTaskGP


def _case() -> tuple[
    WideMultiTaskGP,
    torch.Tensor,
    torch.Tensor,
    list,
]:
    train_X = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        [
            train_X,
            1.0 - (train_X - 0.4).square(),
        ],
        dim=-1,
    )
    model = WideMultiTaskGP(train_X=train_X, train_Y=train_Y)
    model.eval()
    Xq = torch.tensor(
        [[[0.1], [0.3], [0.5], [0.7], [0.9]]],
        dtype=torch.double,
        requires_grad=True,
    )
    constraints = make_outcome_constraints(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.1, 1.0],
    )
    return model, train_X, Xq, constraints


def _assert_scalar_acquisition_with_gradient(
    values: torch.Tensor,
    Xq: torch.Tensor,
) -> None:
    gradient = torch.autograd.grad(values.sum(), Xq)[0]
    assert values.shape == torch.Size([1])
    assert torch.isfinite(values).all()
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()


def test_wide_multitask_qnehvi_preserves_candidate_axis() -> None:
    model, train_X, Xq, constraints = _case()
    acquisition = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=[-0.1, -0.1],
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([8]), seed=123),
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,
    )

    values = acquisition(Xq)

    _assert_scalar_acquisition_with_gradient(values, Xq)


def test_wide_multitask_nparego_preserves_candidate_axis() -> None:
    model, train_X, Xq, constraints = _case()
    acquisition = qMultiOutputRegressionNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        sampler=SobolQMCNormalSampler(torch.Size([8]), seed=123),
        constraints=constraints,
    )

    values = acquisition(Xq)

    _assert_scalar_acquisition_with_gradient(values, Xq)


def test_wide_multitask_nsgaii_posterior_mean_preserves_objective_axis() -> None:
    model, _, Xq, constraints = _case()
    population_X = Xq.detach().squeeze(0).unsqueeze(-2)
    acquisition = MultiOutputPosteriorMean(model=model)

    posterior_mean = model.posterior(population_X).mean
    values = acquisition(population_X)

    assert posterior_mean.shape == torch.Size([5, 1, 2])
    assert values.shape == torch.Size([5, 2])
    assert torch.isfinite(values).all()
    assert [constraint(values).shape for constraint in constraints] == [
        torch.Size([5]),
        torch.Size([5]),
    ]


def test_wide_multitask_output_subset_keeps_q_for_mc_sampling() -> None:
    model, _, Xq, _ = _case()
    posterior = model.posterior(Xq, output_indices=[1])
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=123)

    samples = sampler(posterior)

    assert posterior.mean.shape == torch.Size([1, 5, 1])
    assert samples.shape == torch.Size([8, 1, 5, 1])
