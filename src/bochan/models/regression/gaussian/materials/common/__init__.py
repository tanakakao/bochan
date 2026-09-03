"""Common contracts for material-aware Gaussian models."""

from .base import MaterialEncoder
from .baseline import BaselineAggregation, MaterialBaselineSpec, MaterialPropertyContract
from .compatibility import (
    LEGACY_MATERIAL_MODEL_PATHS,
    MaterialCompatibilityPath,
    canonical_material_model_paths,
    legacy_material_model_paths,
)
from .fusion import ConcatFusion, MaterialProcessFusion, build_material_process_fusion
from .hardening import (
    ResidualProductionReport,
    assert_residual_posterior_equivalent,
    shared_parameter_aliases,
    validate_residual_production_model,
)
from .multi_baseline import MaterialBaselinePlan, MultipleBaselineModelListGP, ResolvedBaselineAssignment
from .multitask import (
    MaterialMultiTaskSpec,
    MaterialTaskMode,
    task_covar_module,
    validate_correlated_task_kernel,
    validate_wide_material_targets,
)
from .pretrained import (
    MaterialDomain,
    PretrainedLoadingMode,
    PretrainedMaterialCapabilities,
    PretrainedMaterialSpec,
    resolve_pretrained_loading_mode,
)
from .process import (
    MixedProcessLayout,
    resolve_mixed_process_input_transform,
    resolve_mixed_process_layout,
    select_continuous_process_branch,
)
from .registry import (
    MATERIAL_FAMILY_REGISTRY,
    MaterialFamilyRegistration,
    MaterialModelVariant,
    get_material_family,
    list_material_families,
)
from .relaxation import MaterialStructureRelaxer, StructureRelaxationResult, validate_structure_relaxer
from .residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
    predict_material_baseline,
    require_residual_gp_capability,
    validate_direct_material_predictions,
)
from .residual_multitask import SingleOutputBaselineAdapter
from .surrogate import (
    MaterialSurrogateKind,
    MaterialSurrogateSpec,
    build_material_gaussian_surrogate,
    resolve_material_latent_dim,
)
from .tensor_target import TensorTargetKind, TensorTargetLayout
from .training import (
    EncoderTrainingMode,
    EncoderTrainingPolicy,
    apply_encoder_train_mode,
    apply_encoder_training_policy,
    configure_encoder_parameters,
    unique_module_parameters,
)

__all__ = [
    "BaselineAggregation", "ConcatFusion", "DirectMaterialPredictor", "EncoderTrainingMode",
    "EncoderTrainingPolicy", "LEGACY_MATERIAL_MODEL_PATHS", "MATERIAL_FAMILY_REGISTRY",
    "MaterialBaselinePlan", "MaterialBaselineSpec", "MaterialCompatibilityPath", "MaterialDomain",
    "MaterialEncoder", "MaterialFamilyRegistration", "MaterialModelVariant", "MaterialMultiTaskSpec",
    "MaterialProcessFusion", "MaterialPropertyContract", "MaterialStructureRelaxer", "MaterialSurrogateKind",
    "MaterialSurrogateSpec", "MaterialTaskMode", "MixedProcessLayout", "MultipleBaselineModelListGP",
    "PretrainedLoadingMode", "PretrainedMaterialCapabilities", "PretrainedMaterialSpec", "ResidualMaterialGPModel",
    "ResidualProductionReport", "ResolvedBaselineAssignment", "SingleOutputBaselineAdapter",
    "StructureRelaxationResult", "TensorTargetKind", "TensorTargetLayout", "apply_encoder_train_mode",
    "apply_encoder_training_policy", "assert_residual_posterior_equivalent", "build_material_gaussian_surrogate",
    "build_material_process_fusion", "canonical_material_model_paths", "compute_material_residual_targets",
    "configure_encoder_parameters", "get_material_family", "legacy_material_model_paths", "list_material_families",
    "predict_material_baseline", "require_residual_gp_capability", "resolve_material_latent_dim",
    "resolve_mixed_process_input_transform", "resolve_mixed_process_layout", "resolve_pretrained_loading_mode",
    "select_continuous_process_branch", "shared_parameter_aliases", "task_covar_module", "unique_module_parameters",
    "validate_correlated_task_kernel", "validate_direct_material_predictions", "validate_residual_production_model",
    "validate_structure_relaxer", "validate_wide_material_targets",
]
