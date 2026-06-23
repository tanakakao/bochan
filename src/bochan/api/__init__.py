"""High-level API for bochan."""

from . import engine as _engine
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
from .engine_defaults import BayesianOptimizer

# Keep ``from bochan.api.engine import BayesianOptimizer`` aligned with the
# public high-level API before importing modules that depend on it.
_engine.BayesianOptimizer = BayesianOptimizer

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
from .study import CandidateBatch, StudySnapshot, Trial, TrialState
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
    "DEFAULT_MODEL_REGISTRY",
    "DataContext",
    "EarlyStoppingConfig",
    "FitConfig",
    "GenerationSchedule",
    "GenerationStep",
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
    "StopDecision",
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
