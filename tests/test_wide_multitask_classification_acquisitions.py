from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.binary.bayesian_optimization import (
    qMultiOutputBinaryExpectedHypervolumeImprovement,
    qMultiOutputBinaryNoisyExpectedHypervolumeImprovement,
    qMultiOutputBinaryNParEGO,
)
from bochan.acquisition.multiclass.bayesian_optimization import (
    MulticlassTargetProbabilityObjective,
    compute_observed_multiclass_utility,
    qMultiOutputMulticlassExpectedHypervolumeImprovement,
    qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement,
    qMultiOutputMulticlassNParEGO,
)
from bochan.acquisition.objective import make_outcome_constraints
from bochan.acquisition.ordinal.bayesian_optimization import (
    compute_observed_ordinal_utility,
    qMultiOutputOrdinalExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement,
    qMultiOutputOrdinalNParEGO,
    qMultiOutputOrdinalUtilityObjective,
)
from bochan.models.wide_multitask_variants import (
    WideMultiTaskBinaryClassificationGPModel,
    WideMultiTaskMulticlassClassificationGPModel,
    WideMultiTaskOrdinalGPModel,
)


def _train_x() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)


def _candidate_x() -> torch.Tensor:
    return torch.tensor(
        [[[0.1], [0.3], [0.5], [0.7], [0.9]]],
        dtype=torch.double,
        requires_grad=True,
    )


def _sampler() -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(torch.Size([4]), seed=123)


def _assert_value_and_gradient(value: torch.Tensor, X: torch.Tensor) -> None:
    gradient = torch.autograd.grad(value.sum(), X, retain_graph=True)[0]
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
    assert gradient.shape == X.shape
    assert torch.isfinite(gradient).all()


def test_wide_binary_multitask_acquisition_family_preserves_q_and_constraints() -> None:
    train_X = _train_x()
    train_Y = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.double,
    )
    model = WideMultiTaskBinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        rank=1,
        num_inducing_points=6,
    )
    model.eval()
    model.likelihood.eval()

    Xq = _candidate_x()
    latent = model.latent_posterior(Xq)
    assert latent.mean.shape == torch.Size([1, 5, 2])
    assert latent.event_shape == torch.Size([5, 2])
    assert latent.rsample(torch.Size([4])).shape == torch.Size([4, 1, 5, 2])

    constraints = make_outcome_constraints(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.1, 0.9],
    )
    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)

    ehvi = qMultiOutputBinaryExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=_sampler(),
        constraints=constraints,
    )
    _assert_value_and_gradient(ehvi(Xq), Xq)

    nehvi = qMultiOutputBinaryNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        sampler=_sampler(),
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,
    )
    _assert_value_and_gradient(nehvi(Xq), Xq)

    nparego = qMultiOutputBinaryNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=ref_point,
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        sampler=_sampler(),
        constraints=constraints,
    )
    assert len(nparego.constraints) == 2
    _assert_value_and_gradient(nparego(Xq), Xq)


def test_wide_ordinal_multitask_acquisition_family_uses_one_utility_per_task() -> None:
    train_X = _train_x()
    train_Y = torch.tensor(
        [
            [0, 2],
            [0, 2],
            [1, 2],
            [1, 1],
            [2, 1],
            [2, 0],
        ],
        dtype=torch.long,
    )
    model = WideMultiTaskOrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=1,
        inducing_points_num=6,
    )
    model.eval()
    model.likelihood.eval()

    utility_values = [
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    ]
    objective = qMultiOutputOrdinalUtilityObjective(
        model=model,
        utility_values=utility_values,
    )
    assert len(objective.ordinal_likelihoods) == 2
    assert objective.ordinal_likelihoods[0] is objective.ordinal_likelihoods[1]

    Xq = _candidate_x()
    posterior = model.posterior(Xq)
    samples = _sampler()(posterior)
    values = objective(samples, X=Xq)
    assert posterior.mean.shape == torch.Size([1, 5, 2])
    assert samples.shape == torch.Size([4, 1, 5, 2])
    assert values.shape == torch.Size([4, 1, 5, 2])

    Y_baseline = compute_observed_ordinal_utility(
        train_Y=train_Y,
        utility_values=utility_values,
    ).to(dtype=torch.double)
    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=Y_baseline,
    )
    constraints = make_outcome_constraints(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.1, 1.9],
    )

    ehvi = qMultiOutputOrdinalExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        objective=objective,
        sampler=_sampler(),
        constraints=constraints,
    )
    _assert_value_and_gradient(ehvi(Xq), Xq)

    nehvi = qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        objective=objective,
        sampler=_sampler(),
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,
    )
    _assert_value_and_gradient(nehvi(Xq), Xq)

    nparego = qMultiOutputOrdinalNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=ref_point,
        objective=objective,
        Y_baseline=Y_baseline,
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        sampler=_sampler(),
        constraints=constraints,
    )
    assert len(nparego.constraints) == 2
    _assert_value_and_gradient(nparego(Xq), Xq)


def test_wide_multiclass_multitask_acquisition_family_reduces_class_axis() -> None:
    train_X = _train_x()
    train_Y = torch.tensor(
        [
            [0, 2],
            [0, 2],
            [1, 2],
            [1, 1],
            [2, 1],
            [2, 0],
        ],
        dtype=torch.long,
    )
    model = WideMultiTaskMulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y,
        num_classes=3,
        rank=1,
        num_inducing_points=6,
    )
    model.eval()
    model.likelihood.eval()

    Xq = _candidate_x()
    posterior = model.posterior(Xq)
    samples = _sampler()(posterior)
    objective = MulticlassTargetProbabilityObjective(
        num_outputs=2,
        utility_values=torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    )
    values = objective(samples, X=Xq)

    assert posterior.mean.shape == torch.Size([1, 5, 2, 3])
    assert posterior._extended_shape() == torch.Size([1, 5, 2])
    assert posterior.batch_shape == torch.Size([1])
    assert posterior.event_shape == torch.Size([5, 2, 3])
    assert samples.shape == torch.Size([4, 1, 5, 2, 3])
    assert values.shape == torch.Size([4, 1, 5, 2])

    Y_baseline = compute_observed_multiclass_utility(
        train_Y=train_Y,
        utility_values=torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    ).to(dtype=torch.double)
    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point,
        Y=Y_baseline,
    )
    constraints = make_outcome_constraints(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.1, 1.9],
    )

    ehvi = qMultiOutputMulticlassExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        objective=objective,
        sampler=_sampler(),
        constraints=constraints,
    )
    _assert_value_and_gradient(ehvi(Xq), Xq)

    nehvi = qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_X,
        objective=objective,
        sampler=_sampler(),
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,
    )
    _assert_value_and_gradient(nehvi(Xq), Xq)

    nparego = qMultiOutputMulticlassNParEGO(
        model=model,
        X_baseline=train_X,
        ref_point=ref_point,
        objective=objective,
        Y_baseline=Y_baseline,
        weights=torch.tensor([0.4, 0.6], dtype=torch.double),
        sampler=_sampler(),
        constraints=constraints,
    )
    assert len(nparego.constraints) == 2
    _assert_value_and_gradient(nparego(Xq), Xq)


def test_public_nsgaii_applies_generated_constraints_after_multiclass_objective(
    monkeypatch,
) -> None:
    import bochan.optim as optim
    import bochan.optim.nsgaii_constraints as support

    captured: dict[str, object] = {}

    def fake_optimize_acqf_nsgaii(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([[0.2], [0.8]], dtype=torch.double),
            torch.tensor([[0.2, 0.8], [0.8, 0.2]], dtype=torch.double),
        )

    def raw_constraint(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 0, 0] - 1.0

    monkeypatch.setattr(
        support,
        "_base_optimize_acqf_nsgaii",
        fake_optimize_acqf_nsgaii,
    )

    objective = MulticlassTargetProbabilityObjective(
        num_outputs=2,
        utility_values=torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    )
    generated = make_outcome_constraints(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.1, 1.9],
    )

    optim.optimize_acqf_nsgaii(
        acq_function=object(),
        bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        num_objectives=2,
        objective=objective,
        constraints=[*generated, raw_constraint],
    )

    adapted = captured["constraints"]
    assert adapted is not None
    assert adapted[2] is raw_constraint

    logits = torch.randn(3, 1, 2, 3, dtype=torch.double)
    probabilities = torch.softmax(logits, dim=-1)
    objective_values = objective(probabilities)

    torch.testing.assert_close(adapted[0](probabilities), 0.1 - objective_values[..., 0])
    torch.testing.assert_close(adapted[1](probabilities), objective_values[..., 1] - 1.9)
    assert adapted[0](probabilities).shape == torch.Size([3, 1])
    assert adapted[1](probabilities).shape == torch.Size([3, 1])
