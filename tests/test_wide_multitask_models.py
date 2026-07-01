from __future__ import annotations

import torch

from bochan.api import ModelConfig
from bochan.api.engine_defaults import resolve_multi_output_model_config
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.wide_multitask import (
    WideMultiTaskBinaryClassificationGPModel,
    WideMultiTaskGP,
    WideMultiTaskMulticlassClassificationGPModel,
    WideMultiTaskOrdinalGPModel,
    wide_to_long,
)


def _wide_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.0], [0.5], [1.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor(
        [[0.0, 1.0], [0.5, float("nan")], [1.0, 0.0]],
        dtype=torch.double,
    )
    return train_X, train_Y


def test_wide_to_long_omits_nan_cells() -> None:
    train_X, train_Y = _wide_data()

    X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)

    assert num_tasks == 2
    assert X_long.shape == torch.Size([5, 2])
    assert Y_long.shape == torch.Size([5, 1])
    assert torch.equal(X_long[:, -1], torch.tensor([0, 1, 0, 0, 1], dtype=torch.double))
    assert not torch.isnan(Y_long).any()


def test_wide_to_long_rejects_empty_task() -> None:
    train_X = torch.zeros(3, 1, dtype=torch.double)
    train_Y = torch.tensor(
        [[0.0, float("nan")], [1.0, float("nan")], [2.0, float("nan")]],
        dtype=torch.double,
    )

    try:
        wide_to_long(train_X, train_Y)
    except ValueError as error:
        assert "Missing task ids: [1]" in str(error)
    else:
        raise AssertionError("Expected an empty-task validation error.")


def test_normal_multitask_registry_entries() -> None:
    assert MODEL_REGISTRY["normal"]["regression"]["multitask"] is WideMultiTaskGP
    assert MODEL_REGISTRY["normal"]["multi_objective"]["multitask"] is WideMultiTaskGP
    assert (
        MODEL_REGISTRY["normal"]["binary"]["multitask"]
        is WideMultiTaskBinaryClassificationGPModel
    )
    assert (
        MODEL_REGISTRY["normal"]["ordinal"]["multitask"]
        is WideMultiTaskOrdinalGPModel
    )
    assert (
        MODEL_REGISTRY["normal"]["multiclass"]["multitask"]
        is WideMultiTaskMulticlassClassificationGPModel
    )


def test_multitask_is_not_automatically_converted_to_model_list() -> None:
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


def test_regression_wide_multitask_posterior_has_standard_multioutput_shape() -> None:
    train_X, train_Y = _wide_data()
    model = WideMultiTaskGP(train_X=train_X, train_Y=train_Y)
    model.eval()

    posterior = model.posterior(torch.tensor([[0.25], [0.75]], dtype=torch.double))

    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    samples = posterior.rsample(torch.Size([3]))
    assert samples.shape == torch.Size([3, 2, 2])
