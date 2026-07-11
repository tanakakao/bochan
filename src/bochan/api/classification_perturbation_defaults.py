"""Input-perturbation defaults for ordinal and multiclass vector objectives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import engine as _engine
from . import engine_defaults as _engine_defaults
from . import factory as _factory
from .automatic_default_utils import _num_outputs as _bundle_num_outputs
from .configs import AcquisitionConfig, ModelBundle, ObjectiveConfig

_APPLIED = False
_BASE_BUILD_ACQUISITION = _factory.build_acquisition
_BASE_BUILD_OBJECTIVE = _factory.build_objective
_BASE_BUILD_ORDINAL = _factory._build_ordinal_objective
_BASE_RESOLVE_OBJECTIVE = _engine._resolve_objective_config_n_w_from_input_transform
_BASE_RESOLVE_NPAREGO_CLASS = _engine_defaults._resolve_default_regression_nparego_class


def _normalize(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _is_vector_strategy(config: AcquisitionConfig) -> bool:
    name = _normalize(config.name)
    cls_name = _normalize(getattr(config.acqf_cls, "__name__", ""))
    combined = f"{name}{cls_name}"
    return (
        name in {
            "nsgaii",
            "nsga2",
            "ehi",
            "qehi",
            "ehvi",
            "qehvi",
            "nehi",
            "qnehi",
            "nehvi",
            "qnehvi",
        }
        or "nparego" in combined
        or "expectedhypervolumeimprovement" in combined
    )


def _is_multi_output(bundle: ModelBundle | None) -> bool:
    if bundle is None:
        return False
    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _num_outputs(bundle: ModelBundle) -> int:
    try:
        return max(1, int(bundle.model.num_outputs))
    except (AttributeError, TypeError, ValueError):
        shape = getattr(bundle.train_Y, "shape", None)
        return 1 if shape is None or len(shape) <= 1 else int(shape[-1])


def _output_names(bundle: ModelBundle | None) -> list[str] | None:
    if bundle is None:
        return None
    names = getattr(bundle.model, "output_names", None)
    if callable(names):
        names = names()
    return None if names is None else list(names)


def _resolve_outcome_constraint_config(
    *,
    bundle: ModelBundle | None,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Resolve deferred high-level outcome constraints once model outputs exist."""

    constraint_config = getattr(config, "outcome_constraint_config", None)
    if constraint_config is None:
        return config
    if isinstance(constraint_config, dict):
        from .acquisition_config import OutcomeConstraintConfig

        constraint_config = OutcomeConstraintConfig(**constraint_config)
        config.outcome_constraint_config = constraint_config

    # Model-dependent class / rank constraints are applied by wrapping the base
    # acquisition, not by BoTorch's sample-only constraints argument.
    if constraint_config.wrapper_constraints():
        kwargs = dict(config.acqf_kwargs)
        kwargs.pop("constraints", None)
        config.constraints = None
        config.acqf_kwargs = kwargs
        return config

    if config.constraints is None:
        built_constraints = constraint_config.build(output_names=_output_names(bundle))
        if built_constraints:
            kwargs = dict(config.acqf_kwargs)
            kwargs["constraints"] = built_constraints
            config.constraints = built_constraints
            config.acqf_kwargs = kwargs
    return config


def _resolve_hybrid_nparego_class(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Route hybrid multi-output NParEGO to bochan's vector implementation."""

    resolved = _BASE_RESOLVE_NPAREGO_CLASS(bundle, config)
    if resolved.acqf_cls is not config.acqf_cls:
        return resolved
    if _normalize(config.name) not in {"nparego", "qnparego"}:
        return config
    if str(bundle.task_type) != "hybrid" or _bundle_num_outputs(bundle.train_Y) < 2:
        return config

    from bochan.acquisition.regression.bayesian_optimization import (
        qMultiOutputRegressionNParEGO,
    )

    return replace(config, acqf_cls=qMultiOutputRegressionNParEGO)


def _signs(bundle: ModelBundle, config: ObjectiveConfig, kwargs: dict[str, Any]):
    explicit = kwargs.pop("objective_signs", None)
    if explicit is not None:
        return explicit
    if config.directions is not None:
        return [_factory._direction_to_sign(value) for value in config.directions]
    if config.maximize:
        return None
    return [-1.0] * _num_outputs(bundle)


def _build_ordinal(bundle: ModelBundle, config: ObjectiveConfig):
    if _factory._objective_mode(config) != "multi_output":
        return _BASE_BUILD_ORDINAL(bundle, config)

    from bochan.acquisition.ordinal.bayesian_optimization import (
        qMultiOutputOrdinalUtilityObjective,
    )
    from bochan.acquisition.ordinal.bayesian_optimization._utility_defaults import (
        infer_multioutput_ordinal_utility_values,
    )

    utility_values = config.utility_values
    if utility_values is None:
        utility_values = infer_multioutput_ordinal_utility_values(bundle.model)

    kwargs = dict(config.objective_kwargs)
    ordinal_likelihoods = kwargs.pop("ordinal_likelihoods", None)
    if ordinal_likelihoods is None:
        ordinal_likelihoods = config.ordinal_likelihood
    resolved = {
        "model": bundle.model,
        "utility_values": utility_values,
        "ordinal_likelihoods": ordinal_likelihoods,
        "objective_signs": _signs(bundle, config, kwargs),
        "link": kwargs.pop("link", "auto"),
        "input_perturbation_n_w": kwargs.pop(
            "input_perturbation_n_w",
            kwargs.pop("n_w", config.n_w),
        ),
        "risk_type": kwargs.pop("risk_type", config.risk_type),
        "risk_alpha": kwargs.pop(
            "risk_alpha",
            kwargs.pop("alpha", config.alpha),
        ),
    }
    resolved.update(kwargs)
    resolved = _factory._filter_kwargs_for_callable(
        qMultiOutputOrdinalUtilityObjective,
        resolved,
    )
    return qMultiOutputOrdinalUtilityObjective(**resolved)


def _build_multiclass(bundle: ModelBundle, config: AcquisitionConfig):
    from bochan.acquisition.multiclass.bayesian_optimization.input_perturbation import (
        InputPerturbationMultiOutputObjectiveAdapter,
    )
    from bochan.acquisition.multiclass.bayesian_optimization.multi_output import (
        MulticlassTargetProbabilityObjective,
    )

    objective_config = config.objective_config
    if objective_config is None:
        return None
    mode = _factory._objective_mode(objective_config)
    if mode == "none":
        return None
    if mode != "multi_output":
        raise ValueError(
            "Automatic multiclass objectives currently support mode='multi_output' "
            "for EHVI, NEHVI, NParEGO, and NSGA-II."
        )

    kwargs = dict(objective_config.objective_kwargs)
    acq_kwargs = dict(config.acqf_kwargs)
    target_class = kwargs.pop("target_class", acq_kwargs.get("target_class"))
    output_target_classes = kwargs.pop(
        "output_target_classes",
        acq_kwargs.get("output_target_classes"),
    )
    utility_values = kwargs.pop("utility_values", objective_config.utility_values)
    if utility_values is None:
        utility_values = acq_kwargs.get("utility_values")
    objective_signs = _signs(bundle, objective_config, kwargs)
    if objective_signs is None:
        objective_signs = acq_kwargs.get("objective_signs")

    base = MulticlassTargetProbabilityObjective(
        target_class=target_class,
        output_target_classes=output_target_classes,
        num_outputs=_num_outputs(bundle),
        class_reduction=kwargs.pop(
            "class_reduction",
            acq_kwargs.get("class_reduction", "mean"),
        ),
        utility_values=utility_values,
        objective_signs=objective_signs,
        eps=float(kwargs.pop("eps", acq_kwargs.get("eps", 1e-8))),
    )

    n_w = kwargs.pop("n_w", objective_config.n_w)
    if n_w is None or int(n_w) <= 1:
        return base
    return InputPerturbationMultiOutputObjectiveAdapter(
        base,
        n_w=int(n_w),
        risk_type=kwargs.pop("risk_type", objective_config.risk_type),
        alpha=float(kwargs.pop("alpha", objective_config.alpha)),
    )


def _objective_keeps_perturbation_expanded(config: AcquisitionConfig) -> bool:
    objective_config = config.objective_config
    return bool(
        objective_config is not None
        and objective_config.n_w is not None
        and int(objective_config.n_w) > 1
        and objective_config.risk_type is None
        and objective_config.aggregate_mean_when_no_risk is False
    )


def _maybe_disable_objective_shape_check(objective: Any, config: AcquisitionConfig) -> Any:
    """Allow one-to-many objective values to stay on q*n_w before constraints."""

    if objective is not None and _objective_keeps_perturbation_expanded(config):
        objective._verify_output_shape = False
    return objective


def _build_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: Any | None = None,
):
    if (
        str(bundle.task_type) != "multiclass"
        or config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is None
    ):
        objective = _BASE_BUILD_OBJECTIVE(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )
        return _maybe_disable_objective_shape_check(objective, config)
    return _build_multiclass(bundle, config)


def _build_acquisition(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: Any | None = None,
) -> Any:
    """Build acquisition and wrap model-dependent outcome constraints."""

    config = _resolve_outcome_constraint_config(bundle=bundle, config=config)
    constraint_config = getattr(config, "outcome_constraint_config", None)
    wrapper_constraints = [] if constraint_config is None else constraint_config.wrapper_constraints()
    if not wrapper_constraints:
        return _BASE_BUILD_ACQUISITION(bundle=bundle, config=config, data_context=data_context)

    base_kwargs = dict(config.acqf_kwargs)
    base_kwargs.pop("constraints", None)
    base_config = replace(
        config,
        constraints=None,
        outcome_constraint_config=None,
        acqf_kwargs=base_kwargs,
    )
    base_acqf = _BASE_BUILD_ACQUISITION(
        bundle=bundle,
        config=base_config,
        data_context=data_context,
    )

    from bochan.acquisition.feasible import FeasibilityWeightedAcquisition

    return FeasibilityWeightedAcquisition(
        acqf=base_acqf,
        model=bundle.model,
        constraints=wrapper_constraints,
        eta=constraint_config.eta,
        posterior_mode=constraint_config.posterior_mode,
        reduce_constraints=constraint_config.reduce_constraints,
        reduce_q=constraint_config.reduce_q,
        min_feasibility=constraint_config.min_feasibility,
        detach_feasibility=constraint_config.detach_feasibility,
    )


def _keep_constrained_perturbation_q_expanded(
    *,
    bundle: ModelBundle | None,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    """Keep constrained MC acquisition objective and constraints shape-aligned."""

    config = _resolve_outcome_constraint_config(bundle=bundle, config=config)
    if bundle is None or config.constraints is None or config.objective_config is None:
        return config
    n_w = _engine._input_transform_n_w_from_bundle(bundle, output=config.objective_config.output)
    if n_w is None or int(n_w) <= 1:
        return config
    if config.objective_config.risk_type is not None:
        return config
    if config.objective_config.aggregate_mean_when_no_risk is False:
        return config
    if "aggregate_mean_when_no_risk" in config.objective_config.objective_kwargs:
        return config

    return replace(
        config,
        objective_config=replace(
            config.objective_config,
            aggregate_mean_when_no_risk=False,
        ),
    )


def _resolve_objective(
    *,
    acq_config: AcquisitionConfig,
    bundle: ModelBundle | None,
) -> AcquisitionConfig:
    resolved = _BASE_RESOLVE_OBJECTIVE(
        acq_config=acq_config,
        bundle=bundle,
    )
    resolved = _keep_constrained_perturbation_q_expanded(bundle=bundle, config=resolved)
    if (
        bundle is None
        or str(bundle.task_type) != "multiclass"
        or resolved.objective is not None
        or resolved.objective_factory is not None
        or resolved.objective_config is not None
    ):
        return resolved

    if not _is_multi_output(bundle) or not _is_vector_strategy(resolved):
        return resolved

    n_w = _engine._input_transform_n_w_from_bundle(bundle)
    return replace(
        resolved,
        objective_config=ObjectiveConfig(
            mode="multi_output",
            n_w=n_w,
            risk_type=None,
        ),
    )


def apply_classification_perturbation_defaults() -> None:
    """Register the support routes once."""

    global _APPLIED
    if _APPLIED:
        return

    from .hetero_ordinal_perturbation import (
        apply_hetero_ordinal_perturbation,
    )

    _factory._build_ordinal_objective = _build_ordinal
    _factory.build_acquisition = _build_acquisition
    _factory.build_objective = _build_objective
    _engine.build_acquisition = _build_acquisition
    _engine._resolve_objective_config_n_w_from_input_transform = _resolve_objective
    _engine_defaults._resolve_objective_config_n_w_from_input_transform = _resolve_objective
    _engine_defaults._resolve_default_regression_nparego_class = (
        _resolve_hybrid_nparego_class
    )
    apply_hetero_ordinal_perturbation()
    _APPLIED = True


__all__ = ["apply_classification_perturbation_defaults"]
