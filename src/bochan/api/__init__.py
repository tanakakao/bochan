"""High-level API for bochan."""

from .acquisition_registry import resolve_acqf_cls
from .configs import (
    AcquisitionConfig,
    CandidateResult,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
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

__all__ = [
    "AcquisitionConfig",
    "BayesianOptimizer",
    "CandidateResult",
    "DataContext",
    "FitConfig",
    "InputTransformConfig",
    "ModelBundle",
    "ModelConfig",
    "MultiObjectiveConfig",
    "MultiOutputConfig",
    "OptimizeConfig",
    "OutputConfig",
    "PredictionResult",
    "build_acquisition",
    "build_model",
    "fit_model",
    "infer_input_type",
    "optimize_candidates",
    "prepare_multi_objective_context",
    "resolve_acqf_cls",
    "resolve_model_cls",
]
