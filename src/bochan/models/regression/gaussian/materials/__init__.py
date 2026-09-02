"""Material-aware Gaussian model infrastructure.

This package owns domain-neutral material contracts and the composition and
structure model namespaces. Concrete model wrappers remain in ``gaussian.deep``
during the staged migration.
"""

from .common import (
    ConcatFusion,
    EncoderTrainingMode,
    EncoderTrainingPolicy,
    MaterialEncoder,
    MaterialProcessFusion,
    MixedProcessLayout,
    apply_encoder_train_mode,
    apply_encoder_training_policy,
    build_material_process_fusion,
    configure_encoder_parameters,
    resolve_mixed_process_input_transform,
    resolve_mixed_process_layout,
    select_continuous_process_branch,
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
