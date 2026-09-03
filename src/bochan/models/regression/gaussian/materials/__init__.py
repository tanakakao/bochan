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
    MaterialMultiTaskSpec,
    MaterialProcessFusion,
    MaterialSurrogateKind,
    MaterialSurrogateSpec,
    MaterialTaskMode,
    MixedProcessLayout,
    apply_encoder_train_mode,
    apply_encoder_training_policy,
    build_material_gaussian_surrogate,
    build_material_process_fusion,
    configure_encoder_parameters,
    resolve_material_latent_dim,
    resolve_mixed_process_input_transform,
    resolve_mixed_process_layout,
    select_continuous_process_branch,
    task_covar_module,
    unique_module_parameters,
    validate_correlated_task_kernel,
    validate_wide_material_targets,
)

__all__ = [
    "ConcatFusion",
    "EncoderTrainingMode",
    "EncoderTrainingPolicy",
    "MaterialEncoder",
    "MaterialMultiTaskSpec",
    "MaterialProcessFusion",
    "MaterialSurrogateKind",
    "MaterialSurrogateSpec",
    "MaterialTaskMode",
    "MixedProcessLayout",
    "apply_encoder_train_mode",
    "apply_encoder_training_policy",
    "build_material_gaussian_surrogate",
    "build_material_process_fusion",
    "configure_encoder_parameters",
    "resolve_material_latent_dim",
    "resolve_mixed_process_input_transform",
    "resolve_mixed_process_layout",
    "select_continuous_process_branch",
    "task_covar_module",
    "unique_module_parameters",
    "validate_correlated_task_kernel",
    "validate_wide_material_targets",
]
