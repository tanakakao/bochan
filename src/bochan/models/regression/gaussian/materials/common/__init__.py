"""Common contracts for material-aware Gaussian models."""

from .base import MaterialEncoder
from .fusion import (
    ConcatFusion,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from .training import (
    EncoderTrainingMode,
    EncoderTrainingPolicy,
    apply_encoder_train_mode,
    apply_encoder_training_policy,
    configure_encoder_parameters,
    unique_module_parameters,
)

__all__ = [
    "ConcatFusion",
    "EncoderTrainingMode",
    "EncoderTrainingPolicy",
    "MaterialEncoder",
    "MaterialProcessFusion",
    "apply_encoder_train_mode",
    "apply_encoder_training_policy",
    "build_material_process_fusion",
    "configure_encoder_parameters",
    "unique_module_parameters",
]
