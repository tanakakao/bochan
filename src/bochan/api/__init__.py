"""High-level API for bochan."""

from .acquisition_registry import available_acqf_names, resolve_acqf_cls
from .configs import (
    AcquisitionConfig,
    AutoStandardizeOutcomeTransform,
    CandidateRepairConfig,
    CandidateResult,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
    ObjectiveConfig,
    OptimizeConfig,
    OutputConfig,
    PredictionResult,
)
from .engine import BayesianOptimizer
from .factory import (
    build_acquisition,
    build_model,
    fit_model,
    infer_input_type,
    optimize_candidates,
    prepare_multi_objective_context,
    resolve_model_cls,
)
from .model_registry import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY, LazyModelRegistry
from .study import BochanStudy, CandidateBatch, StudySnapshot, Trial, TrialState

__all__ = [
    "AcquisitionConfig",
    "AutoStandardizeOutcomeTransform",
    "BayesianOptimizer",
    "BochanStudy",
    "CandidateBatch",
    "CandidateRepairConfig",
    "CandidateResult",
    "DEFAULT_MODEL_REGISTRY",
    "DataContext",
    "FitConfig",
    "InputTransformConfig",
    "LazyModelRegistry",
    "MODEL_REGISTRY",
    "ModelBundle",
    "ModelConfig",
    "MultiObjectiveConfig",
    "MultiOutputConfig",
    "ObjectiveConfig",
    "OptimizeConfig",
    "OutputConfig",
    "PredictionResult",
    "StudySnapshot",
    "Trial",
    "TrialState",
    "available_acqf_names",
    "build_acquisition",
    "build_model",
    "fit_model",
    "infer_input_type",
    "optimize_candidates",
    "prepare_multi_objective_context",
    "resolve_acqf_cls",
    "resolve_model_cls",
]
