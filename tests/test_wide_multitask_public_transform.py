from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multioutput_acquisition import MultiOutputPosteriorMean
from botorch.models.transforms.input import Normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.objective import (
    MultiOutputRegressionInputPerturbationObjective,
    RegressionLinearMCObjective,
)
from bochan.acquisition.regression.active_learning import qMultiOutputRegressionBALD
from bochan.acquisition.regression.bayesian_optimization import (
    qMultiOutputRegressionNParEGO,
)
from bochan.api import AutoStandardizeOutcomeTransform
from bochan.models.transforms.input import build_input_transform
from bochan.models.multitask.task_feature import (
    PerturbationAwareWidePosterior,
    TaskFeatureInputTransform,
    WideMultiTaskGP,
)
from bochan.optim.nsgaii_outputs import adapt_nsgaii_outputs


def test_task_feature_transform_accepts_public_and_internal_inputs() -> None:
    transform = TaskFeatureInputTransform(
        Normalize(
            d=2,
            bounds=torch.tensor(
                [[0.0, 10.0], [10.0, 30.0]],
                dtype=torch.double,
            ),
        ),
        data_dim=2,
    )
    public_X = torch.tensor(
        [[0.0, 10.0], [10.0, 30.0]],
        dtype=torch.double,
    )
    internal_X = torch.tensor(
        [[0.0, 10.0, 0.0], [10.0, 30.0, 1.0]],
        dtype=torch.double,
    )

    public_transformed = transform(public_X)
    internal_transformed = transform(internal_X)

    expected_public = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    torch.testing.assert_close(public_transformed, expected_public)
    torch.testing.assert_close(internal_transformed[:, :2], expected_public)
    torch.testing.assert_close(internal_transformed[:, -1], internal_X[:, -1])
    torch.testing.assert_close(transform.untransform(public_transformed), public_X)
    torch.testing.assert_close(transform.untransform(internal_transformed), internal_X)


def _wide_regression_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.3],
            [0.6, 0.7],
            [0.8, 0.2],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.stack(
        [
            train_X[:, 0] + 0.2 * train_X[:, 1],
            1.0 - train_X[:, 0] + 0.1 * train_X[:, 1],
        ],
        dim=-1,
    )
    return train_X, train_Y


def test_wide_multitask_bald_uses_public_distance_transform() -> None:
    train_X, train_Y = _wide_regression_data()
    model = WideMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(
            d=2,
            bounds=torch.tensor(
                [[0.0, 0.0], [1.0, 1.0]],
                dtype=torch.double,
            ),
        ),
    )
    model.eval()
    acquisition = qMultiOutputRegressionBALD(
        model=model,
        same_batch_penalty_weight=0.1,
        observed_penalty_weight=0.1,
        X_observed=train_X,
    )
    Xq = torch.tensor(
        [[[0.15, 0.25], [0.50, 0.50], [0.85, 0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(Xq)
    gradient = torch.autograd.grad(value.sum(), Xq)[0]

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()


def test_perturbed_wide_posterior_preserves_task_order() -> None:
    flat_values = torch.tensor(
        [
            [
                [0.0],
                [1.0],
                [2.0],
                [10.0],
                [11.0],
                [12.0],
                [100.0],
                [101.0],
                [102.0],
                [110.0],
                [111.0],
                [112.0],
            ]
        ],
        dtype=torch.double,
    )
    base = SimpleNamespace(mean=flat_values)
    posterior = PerturbationAwareWidePosterior(
        base,
        public_q=2,
        num_tasks=2,
        output_indices=[0, 1],
        input_ndim=3,
    )

    transformed = posterior._transform(flat_values)

    torch.testing.assert_close(
        transformed,
        torch.tensor(
            [
                [
                    [0.0, 10.0],
                    [1.0, 11.0],
                    [2.0, 12.0],
                    [100.0, 110.0],
                    [101.0, 111.0],
                    [102.0, 112.0],
                ]
            ],
            dtype=torch.double,
        ),
    )


def _perturbed_wide_case():
    train_X, train_Y = _wide_regression_data()
    input_transform = build_input_transform(
        train_X=train_X,
        bounds=None,
        perturbation=True,
        n_w=4,
        normalize=True,
    )
    model = WideMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=input_transform,
        outcome_transform=AutoStandardizeOutcomeTransform(),
    )
    model.eval()
    Xq = torch.tensor(
        [[[0.15, 0.25], [0.50, 0.50], [0.85, 0.75]]],
        dtype=torch.double,
        requires_grad=True,
    )
    inner_objective = RegressionLinearMCObjective(
        output_indices=[0, 1],
        weights=[1.0, 1.0],
        signs=[1.0, 1.0],
        dtype=torch.double,
    )
    objective = MultiOutputRegressionInputPerturbationObjective(
        inner_objective=inner_objective,
        n_w=4,
        risk_type=None,
    )
    return model, train_X, train_Y, Xq, objective


def _assert_scalar_with_gradient(value: torch.Tensor, Xq: torch.Tensor) -> None:
    gradient = torch.autograd.grad(value.sum(), Xq, retain_graph=True)[0]
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()


def test_perturbed_wide_posterior_preserves_expanded_q_until_objective() -> None:
    model, _, _, Xq, objective = _perturbed_wide_case()
    posterior = model.posterior(Xq)
    sampler = SobolQMCNormalSampler(torch.Size([8]), seed=123)
    samples = sampler(posterior)
    objective_values = objective(samples, X=Xq)

    assert posterior.mean.shape == torch.Size([1, 12, 2])
    assert samples.shape == torch.Size([8, 1, 12, 2])
    assert objective_values.shape == torch.Size([8, 1, 3, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(objective_values).all()


def test_perturbed_wide_ehvi_nehvi_and_nparego() -> None:
    model, train_X, train_Y, Xq, objective = _perturbed_wide_case()
    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)

    ehvi = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(torch.Size([8]), seed=123),
        objective=objective,
    )
    nehvi = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(torch.Size([8]), seed=124),
        objective=objective,
        prune_baseline=False,
        cache_root=False,
    )
    nparego = qMultiOutputRegressionNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=ref_point,
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        sampler=SobolQMCNormalSampler(torch.Size([8]), seed=125),
        objective=objective,
    )

    _assert_scalar_with_gradient(ehvi(Xq), Xq)
    _assert_scalar_with_gradient(nehvi(Xq), Xq)
    _assert_scalar_with_gradient(nparego(Xq), Xq)


def test_perturbed_wide_bald_skips_multioutput_objective_indexing() -> None:
    model, _, _, Xq, objective = _perturbed_wide_case()
    acquisition = qMultiOutputRegressionBALD(
        model=model,
        objective=objective,
        fallback_to_variance=True,
    )

    _assert_scalar_with_gradient(acquisition(Xq), Xq)


def test_perturbed_wide_nsgaii_objective_aggregates_to_pymoo_shape() -> None:
    model, _, _, Xq, objective = _perturbed_wide_case()
    population_X = Xq.detach().squeeze(0).unsqueeze(-2)
    acquisition = MultiOutputPosteriorMean(model=model)
    acquisition_context, objective_adapter = adapt_nsgaii_outputs(
        acquisition,
        objective,
    )

    raw_values = acquisition_context(population_X)
    values = objective_adapter(raw_values)

    assert raw_values.shape == torch.Size([3, 4, 2])
    assert values.shape == torch.Size([3, 2])
    assert torch.isfinite(values).all()
