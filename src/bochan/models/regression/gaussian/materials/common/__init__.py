"""Common contracts for material-aware Gaussian models."""

from .base import MaterialEncoder
from .fusion import (
    ConcatFusion,
    MaterialProcessFusion,
    build_material_process_fusion,
)
from .multitask import (
    MaterialMultiTaskSpec,
    MaterialTaskMode,
    task_covar_module,
    validate_correlated_task_kernel,
    validate_wide_material_targets,
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
    "MaterialMultiTaskSpec",
    "MaterialProcessFusion",
    "MaterialTaskMode",
    "MixedProcessLayout",
    "apply_encoder_train_mode",
    "apply_encoder_training_policy",
    "build_material_process_fusion",
    "configure_encoder_parameters",
    "resolve_mixed_process_input_transform",
    "resolve_mixed_process_layout",
    "select_continuous_process_branch",
    "task_covar_module",
    "unique_module_parameters",
    "validate_correlated_task_kernel",
    "validate_wide_material_targets",
]
