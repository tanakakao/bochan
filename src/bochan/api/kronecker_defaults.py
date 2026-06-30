"""Kronecker-aware defaults for the high-level API."""

from __future__ import annotations

from typing import Any

from . import engine_defaults as _engine_defaults
from .configs import ModelConfig


_original_resolve_multi_output_model_config = (
    _engine_defaults.resolve_multi_output_model_config
)


def resolve_multi_output_model_config(
    model_config: ModelConfig,
    train_Y: Any,
) -> ModelConfig:
    """Keep block-design Kronecker targets in one correlated model.

    Kronecker multi-task models consume ``train_Y`` with shape ``[n, m]``
    directly. They must not be converted to the automatic ModelList-style
    ``MultiOutputConfig``, which would slice the target into ``m`` separate
    one-task Kronecker models.
    """

    model_type = "".join(
        ch for ch in str(model_config.model_type).lower() if ch.isalnum()
    )
    if model_type == "kronecker":
        return model_config
    return _original_resolve_multi_output_model_config(model_config, train_Y)


# ``BayesianOptimizer.fit`` resolves this name from the engine_defaults module
# at runtime. Patch the module-level resolver before exposing the public class.
_engine_defaults.resolve_multi_output_model_config = resolve_multi_output_model_config
BayesianOptimizer = _engine_defaults.BayesianOptimizer


__all__ = ["BayesianOptimizer", "resolve_multi_output_model_config"]
