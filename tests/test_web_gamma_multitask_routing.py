from __future__ import annotations

from bochan.api import ModelConfig
from bochan.models.transforms.outcome import PositiveScaleOutcomeTransform
from bochan.serving.webapp import workflows_tabular


def test_gamma_multitask_is_a_direct_web_multitask_model() -> None:
    assert workflows_tabular._is_direct_multitask_model("gamma_multitask")
    assert workflows_tabular._is_direct_multitask_model("multitask")
    assert not workflows_tabular._is_direct_multitask_model("gamma_base")


def test_gamma_multitask_direct_config_preserves_non_gaussian_family() -> None:
    config = ModelConfig(
        **workflows_tabular._direct_multitask_model_config_kwargs(
            model_type="gamma_multitask",
            input_transform_config=None,
            outcome_transform=True,
            model_kwargs={"rank": 1},
        )
    )

    assert config.task_type == "multi_objective"
    assert config.model_type == "gamma_multitask"
    assert config.multi_output_config is None
    assert config.cat_dims is None
    assert config.model_kwargs == {"rank": 1}
    assert isinstance(config.outcome_transform, PositiveScaleOutcomeTransform)
    assert config.pass_outcome_transform is True
