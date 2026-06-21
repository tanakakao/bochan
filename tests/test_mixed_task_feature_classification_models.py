import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed

from bochan.acquisition.binary.active_learning import qBinaryPredictiveEntropy
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import (
    MultiTaskBinaryClassificationMixedGPModel,
)
from bochan.models.classification.multiclass import (
    MultiTaskMulticlassClassificationGPModel,
    MultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.components.mixed_multitask import (
    remap_dims_without_task_feature,
)
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel


def _make_train_x() -> torch.Tensor:
    """Return [continuous, task_id, category] long-format inputs."""
    return torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.20, 0.0, 1.0],
            [0.45, 0.0, 0.0],
            [0.75, 0.0, 1.0],
            [0.10, 1.0, 1.0],
            [0.35, 1.0, 0.0],
            [0.60, 1.0, 1.0],
            [0.90, 1.0, 0.0],
        ],
        dtype=torch.double,
    )


def _continuous_transform() -> Normalize:
    return Normalize(d=3, indices=[0])


def test_remaps_category_after_task_column_removal() -> None:
    assert remap_dims_without_task_feature([2], task_feature=1, d=3) == [1]


def test_binary_mixed_multitask_posterior_optimizer_and_conditioning() -> None:
    train_X = _make_train_x()
    train_Y = torch.tensor(
        [0, 0, 1, 1, 0, 1, 1, 0],
        dtype=train_X.dtype,
    )
    model = MultiTaskBinaryClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2],
        num_tasks=2,
        task_feature=1,
        rank=2,
        input_transform=_continuous_transform(),
        num_inducing_points=4,
    )
    model.model.mean_module.constant.data.fill_(0.25)
    model.eval()
    model.likelihood.eval()

    X_test = torch.tensor(
        [
            [[0.20, 0.0, 0.0]],
            [[0.80, 1.0, 1.0]],
        ],
        dtype=train_X.dtype,
    )
    posterior = model.posterior(X_test)
    assert posterior.mean.shape[0] == 2
    assert torch.isfinite(posterior.mean).all()
    assert model.task_covar_matrix.shape == torch.Size([2, 2])

    acquisition = qBinaryPredictiveEntropy(model)
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=train_X.dtype,
    )
    candidate, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features_list=[
            {1: 1.0, 2: 0.0},
            {1: 1.0, 2: 1.0},
        ],
        options={"maxiter": 5, "batch_limit": 4},
    )
    assert candidate.shape == torch.Size([1, 3])
    assert candidate[0, 1].item() == 1.0
    assert torch.isfinite(value).all()

    updated = model.condition_on_observations(
        X=torch.tensor([[0.40, 1.0, 1.0]], dtype=train_X.dtype),
        Y=torch.tensor([1.0], dtype=train_X.dtype),
    )
    assert isinstance(updated, MultiTaskBinaryClassificationMixedGPModel)
    assert updated.cat_dims == [2]
    assert updated.train_inputs_raw[0].shape[-2] == train_X.shape[-2] + 1


def test_ordinal_mixed_multitask_probabilities_and_conditioning() -> None:
    train_X = _make_train_x()
    train_Y = torch.tensor([0, 0, 1, 2, 0, 1, 2, 1], dtype=torch.long)
    model = MultiTaskOrdinalMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2],
        num_classes=3,
        num_tasks=2,
        task_feature=1,
        rank=2,
        input_transform=_continuous_transform(),
        inducing_points_num=4,
    )
    model.eval()
    model.likelihood.eval()

    probabilities = model.class_probs(
        torch.tensor([[0.20, 0.0, 0.0]], dtype=train_X.dtype)
    )
    assert probabilities.shape[-1] == 3
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones_like(probabilities[..., 0]),
        atol=1e-6,
    )
    assert model.task_covar_matrix.shape == torch.Size([2, 2])

    updated = model.condition_on_observations(
        X=torch.tensor([[0.55, 1.0, 0.0]], dtype=train_X.dtype),
        Y=torch.tensor([2], dtype=torch.long),
        refit=False,
    )
    assert isinstance(updated, MultiTaskOrdinalMixedGPModel)
    assert updated.cat_dims == [2]
    assert updated.train_inputs_raw[0].shape[-2] == train_X.shape[-2] + 1


def test_multiclass_normal_and_mixed_multitask_models() -> None:
    train_X = _make_train_x()
    train_Y = torch.tensor([0, 0, 1, 2, 0, 1, 2, 1], dtype=torch.long)

    normal = MultiTaskMulticlassClassificationGPModel(
        train_X=train_X[:, :2],
        train_Y=train_Y,
        num_classes=3,
        num_tasks=2,
        task_feature=1,
        rank=2,
        input_transform=Normalize(d=2, indices=[0]),
        num_inducing_points=4,
    )
    normal.eval()
    normal.likelihood.eval()
    assert normal.class_probs(train_X[:2, :2]).shape[-1] == 3

    mixed = MultiTaskMulticlassClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2],
        num_classes=3,
        num_tasks=2,
        task_feature=1,
        rank=2,
        input_transform=_continuous_transform(),
        num_inducing_points=4,
    )
    mixed.eval()
    mixed.likelihood.eval()

    X_test = torch.tensor(
        [
            [[0.20, 0.0, 0.0]],
            [[0.80, 1.0, 1.0]],
        ],
        dtype=train_X.dtype,
    )
    probabilities = mixed.class_probs(X_test)
    assert probabilities.shape[0] == 2
    assert probabilities.shape[-1] == 3
    assert torch.isfinite(probabilities).all()
    assert mixed.task_covar_matrix.shape == torch.Size([3, 2, 2])

    updated = mixed.condition_on_observations(
        X=torch.tensor([[0.55, 1.0, 0.0]], dtype=train_X.dtype),
        Y=torch.tensor([2], dtype=torch.long),
    )
    assert isinstance(updated, MultiTaskMulticlassClassificationMixedGPModel)
    assert updated.cat_dims == [2]


def test_rejects_transforming_task_or_category_columns() -> None:
    train_X = _make_train_x()
    train_Y = torch.tensor(
        [0, 0, 1, 1, 0, 1, 1, 0],
        dtype=train_X.dtype,
    )
    with pytest.raises(ValueError):
        MultiTaskBinaryClassificationMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2],
            num_tasks=2,
            task_feature=1,
            input_transform=Normalize(d=3),
            num_inducing_points=4,
        )


def test_registry_resolves_mixed_multitask_classification_models() -> None:
    assert (
        MODEL_REGISTRY["mixed"]["binary"]["multitask"]
        is MultiTaskBinaryClassificationMixedGPModel
    )
    assert (
        MODEL_REGISTRY["mixed"]["ordinal"]["multitask"]
        is MultiTaskOrdinalMixedGPModel
    )
    assert (
        MODEL_REGISTRY["mixed"]["multiclass"]["multitask"]
        is MultiTaskMulticlassClassificationMixedGPModel
    )
