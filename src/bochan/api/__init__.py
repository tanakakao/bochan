"""High-level API for bochan."""

from .configs import (
    AcquisitionConfig,
    CandidateResult,
    DataContext,
    FitConfig,
    ModelBundle,
    ModelConfig,
    MultiObjectiveConfig,
    MultiOutputConfig,
    OptimizeConfig,
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
    "ModelBundle",
    "ModelConfig",
    "MultiObjectiveConfig",
    "MultiOutputConfig",
    "OptimizeConfig",
    "PredictionResult",
    "build_acquisition",
    "build_model",
    "fit_model",
    "infer_input_type",
    "optimize_candidates",
    "prepare_multi_objective_context",
    "resolve_model_cls",
]
