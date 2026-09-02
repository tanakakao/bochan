"""Common contracts for material-aware Gaussian models."""

from .base import MaterialEncoder
from .fusion import (
    ConcatFusion,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from .process import (
    MixedProcessLayout,
    resolve_mixed_process_input_transform,
    resolve_mixed_process_layout,
    select_continuous_process_branch,
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
    "MixedProcessLayout",
    "apply_encoder_train_mode",
    "apply_encoder_training_policy",
    "build_material_process_fusion",
    "configure_encoder_parameters",
    "resolve_mixed_process_input_transform",
    "resolve_mixed_process_layout",
    "select_continuous_process_branch",
    "unique_module_parameters",
]
