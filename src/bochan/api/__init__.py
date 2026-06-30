"""High-level API for bochan."""

# ruff: noqa: E402

from . import configs as _configs
from . import engine as _engine
from . import factory as _factory
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
from .fit_config import FitConfig
from .optimizer_api import (
    OptimizeConfig,
    optimize_candidates,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from .kronecker_defaults import BayesianOptimizer

# Keep direct submodule imports aligned with the public high-level API.
_configs.OptimizeConfig = OptimizeConfig
_engine.OptimizeConfig = OptimizeConfig
_engine._resolve_optimizer_from_cat_dims = resolve_optimizer_from_cat_dims
_engine._uses_mixed_fixed_features = uses_mixed_fixed_features
_engine.optimize_candidates = optimize_candidates
_factory.optimize_candidates = optimize_candidates

# Keep ``from bochan.api.engine import BayesianOptimizer`` aligned with the
# public high-level API before importing modules that depend on it.
_engine.BayesianOptimizer = BayesianOptimizer

from .factory import (
    build_acquisition,
    build_model,
    fit_model,
    infer_input_type,
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
    "OutcomeConstraintConfig",
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
