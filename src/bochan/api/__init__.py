"""High-level API for bochan."""

from .acquisition_registry import available_acqf_names, resolve_acqf_cls
from .configs import (
    AcquisitionConfig,
    CandidateRepairConfig,
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
    "CandidateRepairConfig",
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
