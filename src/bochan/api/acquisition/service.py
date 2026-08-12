"""Acquisition orchestration for the public :class:`BayesianOptimizer`."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
from typing import Any

from .. import factory as _factory
from ..acquisition_registry import resolve_acqf_cls
from ..classification_perturbation_defaults import (
    _build_multiclass,
    _build_ordinal,
    _maybe_disable_objective_shape_check,
)
from ..configs import AcquisitionConfig, DataContext, ModelBundle, ObjectiveConfig
from ..engine import (
    _filter_context_fields_for_acqf,
    _input_transform_n_w_from_bundle,
    _resolve_objective_config_n_w_from_input_transform,
)
from ..feasibility_defaults import resolve_outcome_constraint_config
from ..llm_selected_acquisition import (
    is_llm_selected_acquisition,
    resolve_llm_selected_acquisition,
)
from .defaults import resolve_acquisition_defaults


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_nsgaii_strategy(config: AcquisitionConfig) -> bool:
    return _normalize_name(config.name) in {"nsgaii", "nsga2"}


def infer_bundle_multi_output(bundle: ModelBundle) -> bool:
    if bool(bundle.metadata.get("multi_output", False)):
        return True
    try:
        return int(getattr(bundle.model, "num_outputs", 1)) > 1
    except (TypeError, ValueError):
        return False


def _is_vector_strategy(config: AcquisitionConfig) -> bool:
    name = _normalize_name(config.name)
    cls_name = _normalize_name(getattr(config.acqf_cls, "__name__", ""))
    combined = f"{name}{cls_name}"
    return (
        is_nsgaii_strategy(config)
        or "nparego" in combined
        or "expectedhypervolumeimprovement" in combined
        or name
        in {
            "ehi",
            "qehi",
            "ehvi",
            "qehvi",
            "nehi",
            "qnehi",
            "nehvi",
            "qnehvi",
        }
    )


def resolve_acquisition_class(
    optimizer: Any,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    if is_llm_selected_acquisition(config):
        config = resolve_llm_selected_acquisition(optimizer, config)
    if config.acqf_cls is not None or config.acqf_factory is not None:
        return config
    if is_nsgaii_strategy(config):
        from bochan.optim.nsgaii.strategy import build_nsgaii_strategy

        return replace(config, acqf_factory=build_nsgaii_strategy)

    optimizer._check_fitted()
    task_type, model_type, multi_output = optimizer._acquisition_routing_context()
    if task_type == str(optimizer.bundle.task_type):
        multi_output = infer_bundle_multi_output(optimizer.bundle)

    if (
        _normalize_name(config.name) in {"nparego", "qnparego"}
        and task_type == "hybrid"
        and multi_output
    ):
        from bochan.acquisition.regression.bayesian_optimization import qRegressionNParEGO

        return replace(config, acqf_cls=qRegressionNParEGO)

    acqf_cls = resolve_acqf_cls(
        config.name,
        optimizer.acquisition_registry,
        task_type=task_type,
        model_type=model_type,
        multi_output=multi_output,
    )
    return replace(config, acqf_cls=acqf_cls)


def resolve_input_perturbation_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
) -> AcquisitionConfig:
    if (
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
    ):
        resolved = _resolve_objective_config_n_w_from_input_transform(
            acq_config=config,
            bundle=bundle,
        )
    else:
        resolved = config
        task_type = str(bundle.task_type)
        if task_type in {
            "regression",
            "multi_objective",
            "binary",
            "ordinal",
            "multiclass",
            "hybrid",
        }:
            n_w = _input_transform_n_w_from_bundle(bundle)
            if n_w is not None:
                if infer_bundle_multi_output(bundle):
                    if _is_vector_strategy(config):
                        resolved = replace(
                            config,
                            objective_config=ObjectiveConfig(
                                mode="multi_output",
                                n_w=n_w,
                                risk_type=None,
                            ),
                        )
                elif config.acqf_factory is None:
                    resolved = replace(
                        config,
                        objective_config=ObjectiveConfig(
                            n_w=n_w,
                            risk_type=None,
                        ),
                    )

    resolved = resolve_outcome_constraint_config(bundle=bundle, config=resolved)
    objective_config = resolved.objective_config
    if (
        resolved.constraints is not None
        and objective_config is not None
        and objective_config.risk_type is None
        and objective_config.aggregate_mean_when_no_risk
        and "aggregate_mean_when_no_risk"
        not in objective_config.objective_kwargs
    ):
        n_w = _input_transform_n_w_from_bundle(
            bundle,
            output=objective_config.output,
        )
        if n_w is not None and int(n_w) > 1:
            resolved = replace(
                resolved,
                objective_config=replace(
                    objective_config,
                    aggregate_mean_when_no_risk=False,
                ),
            )
    return resolved


def resolve_acquisition(
    optimizer: Any,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    resolved = resolve_acquisition_class(optimizer, config)
    resolved = resolve_input_perturbation_objective(optimizer.bundle, resolved)
    resolved, context = resolve_acquisition_defaults(
        optimizer.bundle,
        resolved,
        context,
    )
    resolved = resolve_outcome_constraint_config(
        bundle=optimizer.bundle,
        config=resolved,
    )
    resolved = _filter_context_fields_for_acqf(resolved)
    return resolved, context


def build_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext | None = None,
) -> Any | None:
    if (
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is None
    ):
        objective = _factory.build_objective(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )
        return _maybe_disable_objective_shape_check(objective, config)

    task_type = str(bundle.task_type)
    if task_type == "multiclass":
        objective = _build_multiclass(bundle, config)
    elif (
        task_type == "ordinal"
        and _factory._objective_mode(config.objective_config) == "multi_output"
    ):
        objective = _build_ordinal(bundle, config.objective_config)
    else:
        objective = _factory.build_objective(
            bundle=bundle,
            config=config,
            data_context=data_context,
        )
    return _maybe_disable_objective_shape_check(objective, config)


def build_acquisition(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    data_context: DataContext | None = None,
) -> Any:
    context = data_context or DataContext()
    context = _factory.prepare_multi_objective_context(bundle, context, config)
    if config.acqf_factory is not None:
        acqf = config.acqf_factory(bundle=bundle, config=config, data_context=context)
    else:
        if config.acqf_cls is None:
            raise ValueError(
                "acqf_cls is None. Provide AcquisitionConfig.acqf_cls or acqf_factory."
            )
        kwargs = {"model": bundle.model}
        kwargs.update(config.acqf_kwargs)
        objective = build_objective(
            bundle=bundle,
            config=config,
            data_context=context,
        )
        if objective is not None:
            kwargs["objective"] = objective
        if config.sampler is not None:
            kwargs["sampler"] = config.sampler
        for field_name in config.context_fields:
            value = getattr(context, field_name, None)
            if value is not None:
                kwargs[field_name] = value
        for key, value in context.extra.items():
            if value is not None:
                kwargs[key] = value
        if config.filter_kwargs_by_signature:
            kwargs = _factory._filter_kwargs_for_callable(config.acqf_cls, kwargs)
        acqf = config.acqf_cls(**kwargs)

    model = getattr(bundle, "model", None)
    if model is not None:
        with suppress(Exception):
            object.__setattr__(acqf, "_bochan_thompson_model", model)
    return acqf


__all__ = [
    "build_acquisition",
    "build_objective",
    "infer_bundle_multi_output",
    "is_nsgaii_strategy",
    "resolve_acquisition",
    "resolve_acquisition_class",
    "resolve_input_perturbation_objective",
]
