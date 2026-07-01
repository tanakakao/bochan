"""High-level API for bochan."""

# ruff: noqa: E402

from dataclasses import replace

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
from .engine_defaults import BayesianOptimizer


def _infer_bundle_multi_output(bundle) -> bool:
    """Infer multi-output status for wrappers and correlated multitask models.

    ModelList-style wrappers explicitly set ``metadata['multi_output']``. A
    correlated model such as the Kronecker binary classifier is represented by
    one model object, so the model's ``num_outputs`` property must also be used.
    """
    if bool(bundle.metadata.get("multi_output", False)):
        return True
    num_outputs = getattr(bundle.model, "num_outputs", 1)
    try:
        return int(num_outputs) > 1
    except (TypeError, ValueError):
        return False


def _resolve_acquisition_config_with_model_outputs(
    self,
    acq_config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Resolve contextual aliases using the model's actual output count."""
    if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:
        return acq_config
    self._check_fitted()
    acqf_cls = resolve_acqf_cls(
        acq_config.name,
        self.acquisition_registry,
        task_type=self.bundle.task_type,
        model_type=self.bundle.model_type,
        multi_output=_infer_bundle_multi_output(self.bundle),
    )
    return replace(acq_config, acqf_cls=acqf_cls)


# Correlated multitask models are single model objects but still require the
# multi-output acquisition family for contextual short names.
BayesianOptimizer._resolve_acquisition_config = (
    _resolve_acquisition_config_with_model_outputs
)

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
