import pytest
import torch
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed

from bochan.acquisition.binary.active_learning import qBinaryPredictiveEntropy
from bochan.models.classification.binary.base import MultiTaskBinaryClassificationMixedGPModel
from bochan.models.classification.multiclass.base import (
    MultiTaskMulticlassClassificationGPModel,
    MultiTaskMulticlassClassificationMixedGPModel,
)
from bochan.models.components.mixed_multitask import remap_dims_without_task_feature
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel
from bochan.models.regression.gaussian import MixedMultiTaskGP


def make_x():
    return torch.tensor([
        [0.05, 0.0, 0.0], [0.20, 0.0, 1.0],
        [0.45, 0.0, 0.0], [0.75, 0.0, 1.0],
        [0.10, 1.0, 1.0], [0.35, 1.0, 0.0],
        [0.60, 1.0, 1.0], [0.90, 1.0, 0.0],
    ], dtype=torch.double)


def normalize_continuous():
    return Normalize(d=3, indices=[0])


def test_dim_remapping():
    assert remap_dims_without_task_feature([2], task_feature=1, d=3) == [1]


def test_binary_model_and_mixed_optimizer():
    x = make_x()
    y = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0], dtype=x.dtype)
    model = MultiTaskBinaryClassificationMixedGPModel(
        train_X=x, train_Y=y, cat_dims=[2], num_tasks=2, task_feature=1,
        rank=2, input_transform=normalize_continuous(), num_inducing_points=4,
    )
    model.model.mean_module.constant.data.fill_(0.25)
    model.eval()
    model.likelihood.eval()
    xt = torch.tensor([[[0.2, 0.0, 0.0]], [[0.8, 1.0, 1.0]]], dtype=x.dtype)
    assert model.posterior(xt).mean.shape[0] == 2

    acq = qBinaryPredictiveEntropy(model)
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=x.dtype)
    cand, value = optimize_acqf_mixed(
        acq_function=acq, bounds=bounds, q=1, num_restarts=2, raw_samples=8,
        fixed_features_list=[{1: 1.0, 2: 0.0}, {1: 1.0, 2: 1.0}],
        options={"maxiter": 5, "batch_limit": 4},
    )
    assert cand.shape == torch.Size([1, 3])
    assert cand[0, 1].item() == 1.0
    assert torch.isfinite(value).all()

    updated = model.condition_on_observations(
        X=torch.tensor([[0.4, 1.0, 1.0]], dtype=x.dtype),
        Y=torch.tensor([1.0], dtype=x.dtype),
    )
    assert isinstance(updated, MultiTaskBinaryClassificationMixedGPModel)
    assert updated.cat_dims == [2]


def test_ordinal_model():
    x = make_x()
    y = torch.tensor([0, 0, 1, 2, 0, 1, 2, 1], dtype=torch.long)
    model = MultiTaskOrdinalMixedGPModel(
        train_X=x, train_Y=y, cat_dims=[2], num_classes=3, num_tasks=2,
        task_feature=1, rank=2, input_transform=normalize_continuous(),
        inducing_points_num=4,
    )
    model.eval()
    model.likelihood.eval()
    probs = model.class_probs(torch.tensor([[0.2, 0.0, 0.0]], dtype=x.dtype))
    assert probs.shape[-1] == 3
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-6)


def test_multiclass_normal_and_mixed_models():
    x = make_x()
    y = torch.tensor([0, 0, 1, 2, 0, 1, 2, 1], dtype=torch.long)
    normal = MultiTaskMulticlassClassificationGPModel(
        train_X=x[:, :2], train_Y=y, num_classes=3, num_tasks=2,
        task_feature=1, rank=2, input_transform=Normalize(d=2, indices=[0]),
        num_inducing_points=4,
    )
    normal.eval()
    assert normal.class_probs(x[:2, :2]).shape[-1] == 3

    mixed = MultiTaskMulticlassClassificationMixedGPModel(
        train_X=x, train_Y=y, cat_dims=[2], num_classes=3, num_tasks=2,
        task_feature=1, rank=2, input_transform=normalize_continuous(),
        num_inducing_points=4,
    )
    mixed.eval()
    probs = mixed.class_probs(x[:2])
    assert probs.shape[-1] == 3
    assert mixed.task_covar_matrix.shape == torch.Size([3, 2, 2])


def test_gaussian_model():
    x = make_x()
    y = (torch.sin(2 * torch.pi * x[:, 0]) + 0.2 * x[:, 2] + 0.3 * x[:, 1]).unsqueeze(-1)
    model = MixedMultiTaskGP(
        train_X=x, train_Y=y, task_feature=1, cat_dims=[2], rank=2,
        input_transform=normalize_continuous(),
    )
    model.eval()
    post = model.posterior(
        torch.tensor([[0.2, 0.0], [0.8, 1.0]], dtype=x.dtype),
        output_indices=[0, 1],
    )
    assert post.mean.shape[-2:] == torch.Size([2, 2])


def test_rejects_transforming_task_or_category():
    x = make_x()
    y = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0], dtype=x.dtype)
    with pytest.raises(ValueError):
        MultiTaskBinaryClassificationMixedGPModel(
            train_X=x, train_Y=y, cat_dims=[2], num_tasks=2, task_feature=1,
            input_transform=Normalize(d=3), num_inducing_points=4,
        )
