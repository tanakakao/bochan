from bochan.api import ModelConfig, MultiOutputConfig


def test_placeholder() -> None:
    config = ModelConfig(task_type="multiclass", model_type="base")
    assert config.multi_output_config is None
    assert MultiOutputConfig() is not None
