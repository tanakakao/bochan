from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import StratifiedStandardize
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from bochan.acquisition.objective import make_outcome_constraints
from bochan.api import AutoStandardizeOutcomeTransform, ModelConfig
from bochan.api.engine_defaults import resolve_multi_output_model_config
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.wide_multitask import wide_to_long
from bochan.models.wide_multitask_variants import (
    TaskFeatureInputTransform,
    WideMultiTaskBinaryClassificationGPModel,
    WideMultiTaskGP,
    WideMultiTaskMulticlassClassificationGPModel,
    WideMultiTaskOrdinalGPModel,
)


def _wide_data():
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[0.0, 1.0], [0.5, float("nan")], [1.0, 0.0]],
        dtype=torch.double,
    )
    return X, Y


def test_wide_to_long_and_registry() -> None:
    X, Y = _wide_data()
    X_long, Y_long, num_tasks = wide_to_long(X, Y)
    assert num_tasks == 2
    assert X_long.shape == torch.Size([5, 2])
    assert Y_long.shape == torch.Size([5, 1])
    assert MODEL_REGISTRY["normal"]["regression"]["multitask"] is WideMultiTaskGP
    assert MODEL_REGISTRY["normal"]["multi_objective"]["multitask"] is WideMultiTaskGP
    assert MODEL_REGISTRY["normal"]["binary"]["multitask"] is WideMultiTaskBinaryClassificationGPModel
    assert MODEL_REGISTRY["normal"]["ordinal"]["multitask"] is WideMultiTaskOrdinalGPModel
    assert MODEL_REGISTRY["normal"]["multiclass"]["multitask"] is WideMultiTaskMulticlassClassificationGPModel


def test_multitask_is_not_converted_to_model_list() -> None:
    config = ModelConfig(
        task_type="multi_objective",
        model_type="multitask",
        input_type="normal",
        outcome_transform=False,
    )
    resolved = resolve_multi_output_model_config(
        config,
        torch.zeros(4, 2, dtype=torch.double),
    )
    assert resolved is config
    assert resolved.multi_output_config is None


def test_task_feature_is_not_normalized() -> None:
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
    X = torch.tensor(
        [[0.0, 10.0, 0.0], [10.0, 30.0, 1.0]],
        dtype=torch.double,
    )
    transformed = transform(X)
    torch.testing.assert_close(
        transformed[:, :2],
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
    )
    torch.testing.assert_close(transformed[:, -1], X[:, -1])


def test_regression_outputs_are_standardized_per_task() -> None:
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[0.0, 100.0], [1.0, 200.0], [2.0, 300.0]],
        dtype=torch.double,
    )
    model = WideMultiTaskGP(
        train_X=X,
        train_Y=Y,
        outcome_transform=AutoStandardizeOutcomeTransform(),
    )
    assert isinstance(model.outcome_transform, StratifiedStandardize)
    torch.testing.assert_close(
        model.outcome_transform.means.squeeze(-1),
        torch.tensor([1.0, 200.0], dtype=torch.double),
    )
    torch.testing.assert_close(
        model.outcome_transform.stdvs.squeeze(-1),
        torch.tensor([1.0, 100.0], dtype=torch.double),
    )


def test_multiclass_outputs_are_tasks_not_classes() -> None:
    config = ModelConfig(
        task_type="multiclass",
        model_type="multitask",
        outcome_transform=True,
    )
    assert config.outcome_transform is None

    X = torch.rand(6, 2, dtype=torch.double)
    Y = torch.tensor(
        [[0, 2], [1, 1], [2, 0], [0, 2], [1, 1], [2, 0]],
        dtype=torch.double,
    )
    model = WideMultiTaskMulticlassClassificationGPModel(
        train_X=X,
        train_Y=Y,
        rank=2,
        num_inducing_points=4,
        input_transform=Normalize(d=2, bounds=torch.stack([X.min(0).values, X.max(0).values])),
    )
    model.eval()
    probabilities = model.class_probs(X[:2])
    probability_list = model.class_probs_list(X[:2])
    assert model.num_tasks == 2
    assert model.num_outputs == 2
    assert model.num_classes == 3
    assert model.num_classes_list == [3, 3]
    assert probabilities.shape == torch.Size([2, 2, 3])
    assert len(probability_list) == 2
    torch.testing.assert_close(probability_list[0], probabilities[..., 0, :])
    torch.testing.assert_close(probability_list[1], probabilities[..., 1, :])


def test_regression_posterior_and_qmc_gradients() -> None:
    X, Y = _wide_data()
    model = WideMultiTaskGP(train_X=X, train_Y=Y)
    model.eval()
    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))
    assert model.num_outputs == 2
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.rsample(torch.Size([3])).shape == torch.Size([3, 2, 2])

    Xq = torch.tensor(
        [[[0.2], [0.5], [0.8]]],
        dtype=torch.double,
        requires_grad=True,
    )
    samples = SobolQMCNormalSampler(torch.Size([16]))(model.posterior(Xq))
    gradient = torch.autograd.grad(samples.sum(), Xq)[0]
    assert samples.shape[-2:] == torch.Size([3, 2])
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()


def test_regression_wide_posterior_auto_sampler_and_qehvi() -> None:
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    Y = torch.tensor(
        [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        dtype=torch.double,
    )
    model = WideMultiTaskGP(train_X=X, train_Y=Y)
    model.eval()

    Xq = torch.tensor(
        [[[0.1], [0.3], [0.5], [0.7], [0.9]]],
        dtype=torch.double,
        requires_grad=True,
    )
    posterior = model.posterior(Xq)
    sampler = get_sampler(
        posterior=posterior,
        sample_shape=torch.Size([16]),
        seed=123,
    )
    samples = sampler(posterior)
    assert samples.shape == torch.Size([16, 1, 5, 2])

    ref_point = torch.tensor([-0.1, -0.1], dtype=torch.double)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y)
    acquisition = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
        constraints=make_outcome_constraints(
            output_indices=[0, 1],
            operators=["ge", "le"],
            thresholds=[0.1, 0.3],
        ),
    )
    values = acquisition(Xq)
    gradient = torch.autograd.grad(values.sum(), Xq)[0]

    assert values.shape == torch.Size([1])
    assert torch.isfinite(values).all()
    assert gradient.shape == Xq.shape
    assert torch.isfinite(gradient).all()
