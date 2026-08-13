"""Public high-level API for bochan.

Importing this package is declarative: the module exports the canonical
``BayesianOptimizer`` and configuration / factory helpers without replacing
functions or class methods in sibling modules at runtime.
"""

from __future__ import annotations

from bochan.inspection import (
    FeatureGroup,
    FeatureImportanceConfig,
    compute_feature_importance,
)

from .acquisition import build_acquisition
from .registry.acquisition import available_acqf_names, resolve_acqf_cls
from .configs import (
    AutoStandardizeOutcomeTransform,
    CandidateRepairConfig,
    CandidateResult,
    DataContext,
    InputTransformConfig,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OutputConfig,
    PredictionResult,
)
from .configs.acquisition import AcquisitionConfig, OutcomeConstraintConfig
from .configs.fit import FitConfig
from .configs.optimize import (
    OptimizeConfig,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from .cross_validation import (
    CrossValidationConfig,
    CrossValidationResult,
    CVFoldResult,
    CVPredictionResult,
    MetricSummary,
    OutputCrossValidationResult,
    clone_fit_config_for_evaluation,
    clone_model_config_for_evaluation,
)
from .factory import (
    build_model,
    fit_model,
    infer_input_type,
    prepare_multi_objective_context,
    resolve_model_cls,
)
from .registry.model import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry
from .observation import ExperimentFailureConfig, ObservationData
from .optimizer import BayesianOptimizer
from .optimizer_api import optimize_candidates
from .study import CandidateBatch, StudySnapshot, StudySuggestion, Trial, TrialState
from .study_controls import (
    BochanStudy,
    EarlyStoppingConfig,
    GenerationSchedule,
    GenerationStep,
    StopDecision,
)

__all__ = [
    "AcquisitionConfig",
    "AutoStandardizeOutcomeTransform",
    "BayesianOptimizer",
    "BochanStudy",
    "CandidateBatch",
    "CandidateRepairConfig",
    "CandidateResult",
    "CrossValidationConfig",
    "CrossValidationResult",
    "CVFoldResult",
    "CVPredictionResult",
    "DataContext",
    "DEFAULT_MODEL_REGISTRY",
    "EarlyStoppingConfig",
    "ExperimentFailureConfig",
    "FitConfig",
    "FeatureGroup",
    "FeatureImportanceConfig",
    "GenerationSchedule",
    "GenerationStep",
    "InputTransformConfig",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
    "MetricSummary",
    "ModelBundle",
    "ModelConfig",
    "MultiObjectiveConfig",
    "MultiOutputConfig",
    "ObjectiveConfig",
    "ObservationData",
    "OutcomeConstraintConfig",
    "OutputConfig",
    "OutputCrossValidationResult",
    "OptimizeConfig",
    "PredictionResult",
    "StopDecision",
    "StudySnapshot",
    "StudySuggestion",
    "Trial",
    "TrialState",
    "available_acqf_names",
    "build_acquisition",
    "build_model",
    "clone_fit_config_for_evaluation",
    "clone_model_config_for_evaluation",
    "compute_feature_importance",
    "fit_model",
    "infer_input_type",
    "optimize_candidates",
    "prepare_multi_objective_context",
    "resolve_acqf_cls",
    "resolve_model_cls",
    "resolve_optimizer_from_cat_dims",
    "uses_mixed_fixed_features",
]
