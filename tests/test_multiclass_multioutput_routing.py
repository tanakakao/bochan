from __future__ import annotations

import torch

from bochan.api import ModelConfig
from bochan.api.engine_defaults import resolve_multi_output_model_config
from bochan.models.classification.multiclass.base import MultiOutputMulticlassClassificationModel


def test_automatic_multiclass_multi_output_uses_dedicated_wrapper() -> None:
    config = ModelConfig(
        task_type="multiclass",
        model_type="base",
        outcome_transform=False,
    )
    resolved = resolve_multi_output_model_config(
        config,
        torch.zeros(5, 2, dtype=torch.long),
    )

    assert resolved.multi_output_config is not None
    assert resolved.multi_output_config.wrapper_cls is MultiOutputMulticlassClassificationModel
