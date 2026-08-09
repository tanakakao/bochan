"""High-level API for bochan."""

# ruff: noqa: E402

from dataclasses import replace

from . import acquisition_registry as _acquisition_registry
from . import configs as _configs
from . import engine as _engine
from . import engine_defaults as _engine_defaults
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
from .engine_defaults import BayesianOptimizer, _uses_internal_nparego_baseline
from .fit_config import FitConfig
from .optimizer_api import (
    OptimizeConfig,
    optimize_candidates,
    resolve_optimizer_from_cat_dims,
    uses_mixed_fixed_features,
)
from bochan.inspection import FeatureGroup, FeatureImportanceConfig, compute_feature_importance


def _register_contextual_levelset_aliases() -> None:
    """Register support names for hetero ordinal level-set classes.

    The generic contextual resolver follows the binary / multiclass naming
    convention, while the hetero multi-output ordinal implementations expose
    shorter class names. Register only semantically equivalent aliases.
    """
    module_name = "bochan.acquisition.ordinal.levelset_estimation"
    aliases = {
        "qHeteroMultiOutputOrdinalLatentStraddleAcquisition": ("qHeteroMultiOutputOrdinalStraddle"),
        "qHeteroMultiOutputOrdinalBoundaryVarianceAcquisition": ("qHeteroMultiOutputOrdinalBoundaryVariance"),
        "qHeteroMultiOutputOrdinalICUAcquisition": ("qHeteroMultiOutputOrdinalLevelSetUncertainty"),
    }
    for alias, attr_name in aliases.items():
        _acquisition_registry._register_alias(alias, module_name, attr_name)


_register_contextual_levelset_aliases()


def _normalize_strategy_name(value) -> str:
    """Normalize acquisition / candidate-strategy names."""

    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _is_nsgaii_strategy(config: AcquisitionConfig) -> bool:
    """Return whether the acquisition-side strategy is NSGA-II."""

    return _normalize_strategy_name(config.name) in {
        "nsgaii",
        "nsga2",
    }


def _uses_multi_output_sample_objective(config: AcquisitionConfig) -> bool:
    """Return whether a strategy consumes vector posterior samples / means.

    EHVI, NEHVI, NParEGO, and NSGA-II preserve the objective-output dimension
    and therefore need ``q * n_w -> q`` preprocessing when a multi-output model
    uses a one-to-many input transform. Classification active-learning and
    level-set acquisitions instead operate on already computed scalar scores and
    must not receive this Monte Carlo multi-output objective automatically.
    """

    name = _normalize_strategy_name(config.name)
    cls_name = _normalize_strategy_name(getattr(config.acqf_cls, "__name__", ""))
    combined = f"{name}{cls_name}"

    if _is_nsgaii_strategy(config):
        return True
    if "nparego" in combined:
        return True
    if "expectedhypervolumeimprovement" in combined:
        return True
    return name in {
        "ehi",
        "qehi",
        "ehvi",
        "qehvi",
        "nehi",
        "qnehi",
        "nehvi",
        "qnehvi",
    }


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


_original_resolve_objective_config_n_w_from_input_transform = _engine._resolve_objective_config_n_w_from_input_transform


def _resolve_objective_config_n_w_with_default(
    *,
    acq_config: AcquisitionConfig,
    bundle: ModelBundle | None,
) -> AcquisitionConfig:
    """Infer risk-neutral objectives for one-to-many input perturbation.

    Single-output models receive the existing scalar objective. Multi-output
    models receive ``mode='multi_output'`` only for strategies that consume
    vector posterior samples or posterior means: EHVI, NEHVI, NParEGO, and
    NSGA-II. Score-based active-learning and level-set acquisitions are excluded
    because they use classification / utility score objectives instead.
    """

    if (
        acq_config.objective is not None
        or acq_config.objective_factory is not None
        or acq_config.objective_config is not None
    ):
        return _original_resolve_objective_config_n_w_from_input_transform(
            acq_config=acq_config,
            bundle=bundle,
        )

    if bundle is None:
        return acq_config

    task_type = str(bundle.task_type)
    if task_type not in {
        "regression",
        "multi_objective",
        "binary",
        "ordinal",
        "hybrid",
    }:
        return acq_config

    inferred_n_w = _engine._input_transform_n_w_from_bundle(bundle)
    if inferred_n_w is None:
        return acq_config

    if _infer_bundle_multi_output(bundle):
        if not _uses_multi_output_sample_objective(acq_config):
            return acq_config
        return replace(
            acq_config,
            objective_config=ObjectiveConfig(
                mode="multi_output",
                n_w=inferred_n_w,
                risk_type=None,
            ),
        )

    if acq_config.acqf_factory is not None:
        return acq_config

    return replace(
        acq_config,
        objective_config=ObjectiveConfig(
            n_w=inferred_n_w,
            risk_type=None,
        ),
    )


_engine._resolve_objective_config_n_w_from_input_transform = _resolve_objective_config_n_w_with_default
_engine_defaults._resolve_objective_config_n_w_from_input_transform = _resolve_objective_config_n_w_with_default

# Extend the same defaults to ordinal and multiclass vector objectives after
# the binary / regression resolver above is installed.
from .classification_perturbation_defaults import (
    apply_classification_perturbation_defaults,
)

apply_classification_perturbation_defaults()


def _resolve_acquisition_config_with_model_outputs(
    self,
    acq_config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Resolve contextual aliases using the model's actual output count."""
    if acq_config.acqf_cls is not None or acq_config.acqf_factory is not None:
        return acq_config
    if _is_nsgaii_strategy(acq_config):
        from bochan.optim.nsgaii_strategy import build_nsgaii_strategy

        return replace(acq_config, acqf_factory=build_nsgaii_strategy)
    self._check_fitted()
    task_type, model_type, multi_output = self._acquisition_routing_context()
    if task_type == str(self.bundle.task_type):
        multi_output = _infer_bundle_multi_output(self.bundle)
    acqf_cls = resolve_acqf_cls(
        acq_config.name,
        self.acquisition_registry,
        task_type=task_type,
        model_type=model_type,
        multi_output=multi_output,
    )
    return replace(acq_config, acqf_cls=acqf_cls)


# Correlated multitask models are single model objects but still require the
# multi-output acquisition family for contextual short names.
BayesianOptimizer._resolve_acquisition_config = _resolve_acquisition_config_with_model_outputs


_original_build_acquisition = _factory.build_acquisition


def _build_acquisition_with_thompson_sampling_target(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext | None = None,
):
    """Attach the public model used by high-level Thompson sampling dispatch.

    Some acquisition classes expose an internal latent GP through ``.model``.
    Thompson sampling needs a posterior-bearing public model instead, especially
    for binary probability-space models.
    """

    acqf = _original_build_acquisition(bundle, config, data_context)
    model = getattr(bundle, "model", None)
    if model is not None:
        try:
            object.__setattr__(acqf, "_bochan_thompson_model", model)
        except Exception:
            pass
    return acqf


_factory.build_acquisition = _build_acquisition_with_thompson_sampling_target
_engine.build_acquisition = _build_acquisition_with_thompson_sampling_target
if hasattr(_engine_defaults, "build_acquisition"):
    _engine_defaults.build_acquisition = _build_acquisition_with_thompson_sampling_target


_original_candidate = BayesianOptimizer.candidate


def _candidate_with_acquisition_side_nsgaii(
    self,
    acq_config: AcquisitionConfig,
    opt_config: OptimizeConfig,
    *,
    data_context: DataContext | None = None,
    bounds=None,
    return_result: bool = False,
):
    """Force the NSGA-II backend when ``AcquisitionConfig.name`` selects it.

    The optimizer field is intentionally ignored for this strategy. Other
    optimizer options, candidate repair settings, constraints, and ``q`` remain
    available through ``OptimizeConfig``.
    """

    if _is_nsgaii_strategy(acq_config):
        opt_config = replace(opt_config, optimizer="nsgaii")
    return _original_candidate(
        self,
        acq_config,
        opt_config,
        data_context=data_context,
        bounds=bounds,
        return_result=return_result,
    )


BayesianOptimizer.candidate = _candidate_with_acquisition_side_nsgaii

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
from .observation import ExperimentFailureConfig, ObservationData
from .observation_engine import BayesianOptimizer as ObservationBayesianOptimizer

# Observation-aware fitting is the canonical public optimizer.  The class uses
# normal inheritance from the automatic-default engine; no runtime method or
# factory replacement is required for observation handling.
BayesianOptimizer = ObservationBayesianOptimizer

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
    "ModelBundle",
    "ModelConfig",
    "MetricSummary",
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
