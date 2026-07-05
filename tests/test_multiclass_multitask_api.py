from __future__ import annotations

import torch

from bochan.api import InputTransformConfig, ModelConfig
from bochan.api.factory import build_model
from bochan.models.wide_multitask_compat import (
    TaskFeatureInputTransform,
    WideMultiTaskMulticlassClassificationGPModel,
)


def test_api_builds_multiclass_multitask_with_public_output_contract() -> None:
    train_X = torch.rand(8, 5, dtype=torch.double)
    train_Y = torch.tensor(
        [
            [0, 2],
            [1, 1],
            [2, 0],
            [0, 2],
            [1, 1],
            [2, 0],
            [0, 1],
            [2, 2],
        ],
        dtype=torch.double,
    )
    config = ModelConfig(
        task_type="multiclass",
        model_type="multitask",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=False,
        ),
        outcome_transform=True,
        model_kwargs={
            "rank": 2,
            "num_inducing_points": 6,
        },
    )

    bundle = build_model(train_X, train_Y, config)
    model = bundle.model
    model.eval()

    assert isinstance(model, WideMultiTaskMulticlassClassificationGPModel)
    assert isinstance(model.input_transform, TaskFeatureInputTransform)
    assert model.input_transform.data_dim == train_X.shape[-1]
    assert model.num_tasks == train_Y.shape[-1]
    assert model.num_outputs == train_Y.shape[-1]
    assert model.num_classes == 3
    assert bundle.metadata["multi_output"] if "multi_output" in bundle.metadata else model.num_outputs > 1

    probabilities = model.posterior(train_X[:3]).mean
    assert probabilities.shape == torch.Size([3, 2, 3])
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(3, 2, dtype=torch.double),
    )
