"""Public high-level API for bochan.

Importing this package is declarative: the module exports the canonical
``BayesianOptimizer`` and configuration / factory helpers without replacing
functions or class methods in sibling modules at runtime.
"""

from __future__ import annotations

from typing import Any

from bochan.inspection import (
    FeatureGroup,
    FeatureImportanceConfig,
    compute_feature_importance,
)

from .acquisition import build_acquisition
from .acquisition_config import AcquisitionConfig, OutcomeConstraintConfig
from .acquisition_registry import available_acqf_names, resolve_acqf_cls
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
from .fit_config import FitConfig
from .observation import ExperimentFailureConfig, ObservationData
from .optimizer import BayesianOptimizer
from .optimizer_api import (
    OptimizeConfig,
    optimize_candidates,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from .study import CandidateBatch, StudySnapshot, StudySuggestion, Trial, TrialState
from .study_controls import (
    BochanStudy,
    EarlyStoppingConfig,
    GenerationSchedule,
    GenerationStep,
    StopDecision,
)

_MODEL_REGISTRY_EXPORTS = {
    "DEFAULT_MODEL_REGISTRY",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
}


def __getattr__(name: str) -> Any:
    """Lazy-load the model registry without making normal API import bootstrap it."""

    if name in _MODEL_REGISTRY_EXPORTS:
        from . import model_registry

        return getattr(model_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
