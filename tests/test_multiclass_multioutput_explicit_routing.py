import torch

from bochan.api import ModelConfig, MultiOutputConfig
from bochan.api.acquisition.defaults import resolve_multi_output_model_config
from bochan.models.classification.multiclass.base import MultiOutputMulticlassClassificationModel


def test_explicit_config_uses_dedicated_wrapper() -> None:
    config = ModelConfig(
        task_type="multiclass",
        model_type="base",
        multi_output_config=MultiOutputConfig(),
    )
    resolved = resolve_multi_output_model_config(config, torch.zeros(4, 2))
    assert resolved.multi_output_config.wrapper_cls is MultiOutputMulticlassClassificationModel
