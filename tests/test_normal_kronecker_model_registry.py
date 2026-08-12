from __future__ import annotations

import torch
from botorch.models.multitask import KroneckerMultiTaskGP

from bochan.api import ModelConfig
from bochan.api.acquisition.defaults import resolve_multi_output_model_config
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


def test_normal_regression_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["regression"]["kronecker"]
        is KroneckerMultiTaskGP
    )
    assert (
        MODEL_REGISTRY["normal"]["multi_objective"]["kronecker"]
        is KroneckerMultiTaskGP
    )


def test_normal_binary_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["binary"]["kronecker"]
        is KroneckerMultiTaskBinaryClassificationGPModel
    )


def test_normal_ordinal_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["ordinal"]["kronecker"]
        is KroneckerMultiTaskOrdinalGPModel
    )


def test_normal_multiclass_kronecker_registry() -> None:
    assert (
        MODEL_REGISTRY["normal"]["multiclass"]["kronecker"]
        is KroneckerMultiTaskMulticlassClassificationGPModel
    )


def test_kronecker_targets_are_not_converted_to_model_list() -> None:
    train_Y = torch.zeros(8, 3, dtype=torch.double)

    for task_type in (
        "regression",
        "multi_objective",
        "binary",
        "ordinal",
        "multiclass",
    ):
        config = ModelConfig(
            task_type=task_type,
            model_type="kronecker",
            input_type="normal",
            outcome_transform=False,
        )

        resolved = resolve_multi_output_model_config(config, train_Y)

        assert resolved is config
        assert resolved.multi_output_config is None


def test_non_kronecker_multi_output_still_uses_automatic_wrapper() -> None:
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        input_type="normal",
        outcome_transform=False,
    )

    resolved = resolve_multi_output_model_config(
        config,
        torch.zeros(8, 3, dtype=torch.double),
    )

    assert resolved.multi_output_config is not None
