"""Automatic defaults for information-theoretic and look-ahead BO.

This module connects BoTorch KG, MES, JES, and HVKG to bochan's high-level API
without duplicating the acquisition algorithms themselves. Required auxiliary
inputs are generated through BoTorch's registered acquisition input
constructors so the high-level API follows the same contracts as BoTorch.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .automatic_default_utils import _num_outputs
from .automatic_multiobjective import (
    make_default_ref_point,
    observed_multiobjective_values,
)
from .configs import AcquisitionConfig, DataContext, ModelBundle, OptimizeConfig


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def information_acquisition_kind(config: AcquisitionConfig) -> str | None:
    """Return ``kg``, ``mes``, ``jes``, or ``hvkg`` for supported acquisitions."""

    name = _normalize_name(config.name)
    cls_name = _normalize_name(getattr(config.acqf_cls, "__name__", ""))
    if cls_name == "qhypervolumeknowledgegradient" or name in {"hvkg", "qhvkg"}:
        return "hvkg"
    if cls_name == "qknowledgegradient" or name in {
        "kg",
        "qkg",
        "knowledgegradient",
        "qknowledgegradient",
    }:
        return "kg"
    if cls_name == "qjointentropysearch" or name in {"jes", "qjes"}:
        return "jes"
    if cls_name == "qmaxvalueentropy" or name in {"mes", "qmes"}:
        return "mes"
    return None


def is_information_acquisition(config: AcquisitionConfig) -> bool:
    """Return whether ``config`` selects KG, MES, JES, or HVKG."""

    return information_acquisition_kind(config) is not None


def _require_supported_task(bundle: ModelBundle, kind: str) -> None:
    task = str(bundle.task_type)
    if task not in {"regression", "multi_objective", "hybrid"}:
        raise ValueError(
            f"{kind.upper()} is exposed by the high-level API only for regression / "
            f"multi-objective / hybrid models. Current task_type={task!r}."
        )


def _has_configured_objective(config: AcquisitionConfig) -> bool:
    return bool(
        config.objective is not None
        or config.objective_factory is not None
        or config.objective_config is not None
    )


def _require_posterior_transform_for_multi_output(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    kind: str,
) -> None:
    if _num_outputs(bundle.train_Y) < 2:
        return
    if config.acqf_kwargs.get("posterior_transform") is None:
        raise ValueError(
            f"{kind.upper()} requires a scalar posterior. For a multi-output model, "
            "pass an explicit posterior_transform in AcquisitionConfig.acqf_kwargs."
        )


def _require_scalar_kg_objective(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    objective: Any,
) -> None:
    """Require an explicit scalarization route for multi-output KG."""

    if _num_outputs(bundle.train_Y) < 2:
        return
    posterior_transform = config.acqf_kwargs.get("posterior_transform")
    if objective is None and posterior_transform is None:
        raise ValueError(
            "KG requires a scalar terminal objective. For a multi-output model, "
            "configure AcquisitionConfig.objective/objective_config or pass an "
            "explicit posterior_transform in AcquisitionConfig.acqf_kwargs."
        )


def _bounds_as_pairs(bounds: Any) -> list[tuple[float, float]]:
    """Convert bochan ``2 x d`` / ``d x 2`` bounds to BoTorch pair format."""

    import torch

    value = torch.as_tensor(bounds).detach().cpu()
    if value.ndim != 2:
        raise ValueError(
            f"bounds must be a 2D array/tensor. Got shape={tuple(value.shape)}."
        )
    if value.shape[0] == 2:
        value = value.transpose(0, 1)
    elif value.shape[1] != 2:
        raise ValueError(
            "bounds must have shape [2, d] or [d, 2]. "
            f"Got shape={tuple(value.shape)}."
        )
    return [(float(row[0]), float(row[1])) for row in value]


def _training_dataset(bundle: ModelBundle) -> Any:
    """Adapt bochan training tensors to BoTorch's metadata-aware dataset API."""

    from botorch.utils.datasets import SupervisedDataset

    n_features = int(bundle.train_X.shape[-1])
    train_y = bundle.train_Y
    n_outputs = int(train_y.shape[-1]) if train_y.ndim > 1 else 1
    return SupervisedDataset(
        X=bundle.train_X,
        Y=train_y,
        feature_names=[f"x_{idx}" for idx in range(n_features)],
        outcome_names=[f"y_{idx}" for idx in range(n_outputs)],
    )


def _get_botorch_input_constructor(acqf_cls: type) -> Any:
    """Small test seam around BoTorch's public input-constructor registry."""

    from botorch.acquisition.input_constructors import get_acqf_input_constructor

    return get_acqf_input_constructor(acqf_cls)


def _merge_generated_inputs(
    config: AcquisitionConfig,
    generated: dict[str, Any],
) -> AcquisitionConfig:
    """Merge generated constructor inputs while preserving explicit user values."""

    kwargs = {key: value for key, value in generated.items() if key != "model"}
    kwargs.update(config.acqf_kwargs)
    return replace(config, acqf_kwargs=kwargs)


def _resolve_kg(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve qKnowledgeGradient terminal value and scalar objective semantics."""

    _require_supported_task(bundle, "kg")

    from .factory import build_objective

    objective = build_objective(bundle=bundle, config=config, data_context=context)
    _require_scalar_kg_objective(bundle, config, objective)
    if objective is not None and config.objective is None:
        config = replace(config, objective=objective)

    kwargs = dict(config.acqf_kwargs)
    kwargs.setdefault(
        "num_fantasies",
        int(context.extra.get("kg_num_fantasies", 64)),
    )
    config = replace(config, acqf_kwargs=kwargs)

    if config.acqf_kwargs.get("current_value") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "KG requires bounds when current_value is not supplied explicitly."
        )
    if context.X_pending is not None:
        raise ValueError(
            "Automatic KG current_value does not condition on X_pending. When pending "
            "points are present, supply AcquisitionConfig.acqf_kwargs['current_value'] "
            "explicitly using the pending-conditioned terminal value."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        objective=objective,
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        num_fantasies=int(config.acqf_kwargs["num_fantasies"]),
        with_current_value=True,
    )
    return _merge_generated_inputs(config, generated), context


def _resolve_mes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "mes")
    if _has_configured_objective(config):
        raise ValueError(
            "MES does not consume AcquisitionConfig.objective/objective_config. "
            "Use maximize for direction and posterior_transform for output scalarization."
        )
    _require_posterior_transform_for_multi_output(bundle, config, "mes")

    if config.acqf_kwargs.get("candidate_set") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "MES requires bounds when candidate_set is not supplied explicitly."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        candidate_size=int(context.extra.get("mes_candidate_size", 1000)),
        maximize=bool(config.acqf_kwargs.get("maximize", True)),
    )
    return _merge_generated_inputs(config, generated), context


def _resolve_jes(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "jes")
    if _has_configured_objective(config):
        raise ValueError(
            "JES does not consume AcquisitionConfig.objective/objective_config. "
            "Use posterior_transform when a scalarized posterior is required."
        )
    _require_posterior_transform_for_multi_output(bundle, config, "jes")

    optimal_inputs = config.acqf_kwargs.get("optimal_inputs")
    optimal_outputs = config.acqf_kwargs.get("optimal_outputs")
    if (optimal_inputs is None) != (optimal_outputs is None):
        raise ValueError(
            "JES requires optimal_inputs and optimal_outputs to be supplied together."
        )
    if optimal_inputs is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "JES requires bounds when optimal samples are not supplied explicitly."
        )

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        bounds=_bounds_as_pairs(context.bounds),
        num_optima=int(context.extra.get("jes_num_optima", 64)),
        condition_noiseless=bool(
            config.acqf_kwargs.get("condition_noiseless", True)
        ),
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        X_pending=context.X_pending,
        estimation_type=str(config.acqf_kwargs.get("estimation_type", "LB")),
        num_samples=int(config.acqf_kwargs.get("num_samples", 64)),
    )
    return _merge_generated_inputs(config, generated), context


def _prepare_hvkg_multiobjective_context(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Prepare MO context without applying generic scalarization to HVKG."""

    from .factory import prepare_multi_objective_context

    mo_config = context.multi_objective
    if mo_config is not None and mo_config.auto_scalarization:
        context = replace(
            context,
            multi_objective=replace(mo_config, auto_scalarization=False),
        )
    context = prepare_multi_objective_context(bundle, context, config)
    return config, context


def _resolve_hvkg_ref_point(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext, Any]:
    explicit = config.acqf_kwargs.get("ref_point")
    if explicit is not None:
        context.ref_point = None
        return config, context, explicit

    ref_point = context.ref_point
    if ref_point is None:
        values = observed_multiobjective_values(bundle, config, context)
        margin = float(context.extra.get("ref_point_margin", 0.1))
        ref_point = make_default_ref_point(values, margin=margin)
        context.ref_point = ref_point
    return config, context, ref_point


def _resolve_hvkg(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    _require_supported_task(bundle, "hvkg")
    if _num_outputs(bundle.train_Y) < 2:
        raise ValueError(
            "HVKG requires a multi-output model with at least two objectives."
        )

    config, context = _prepare_hvkg_multiobjective_context(bundle, config, context)
    config, context, ref_point = _resolve_hvkg_ref_point(bundle, config, context)

    if config.acqf_kwargs.get("current_value") is not None:
        return config, context
    if context.bounds is None:
        raise ValueError(
            "HVKG requires bounds when current_value is not supplied explicitly."
        )

    from .factory import build_objective

    objective = build_objective(bundle=bundle, config=config, data_context=context)
    if objective is not None and config.objective is None:
        config = replace(config, objective=objective)

    constructor = _get_botorch_input_constructor(config.acqf_cls)
    generated = constructor(
        model=bundle.model,
        training_data=_training_dataset(bundle),
        bounds=_bounds_as_pairs(context.bounds),
        objective_thresholds=None,
        objective=objective,
        posterior_transform=config.acqf_kwargs.get("posterior_transform"),
        num_fantasies=int(config.acqf_kwargs.get("num_fantasies", 8)),
        num_pareto=int(config.acqf_kwargs.get("num_pareto", 10)),
        ref_point=ref_point,
    )
    return _merge_generated_inputs(config, generated), context


def resolve_information_acquisition_defaults(
    bundle: ModelBundle,
    config: AcquisitionConfig,
    context: DataContext,
) -> tuple[AcquisitionConfig, DataContext]:
    """Resolve auxiliary inputs for KG, MES, JES, and HVKG."""

    kind = information_acquisition_kind(config)
    if kind == "kg":
        return _resolve_kg(bundle, config, context)
    if kind == "mes":
        return _resolve_mes(bundle, config, context)
    if kind == "jes":
        return _resolve_jes(bundle, config, context)
    if kind == "hvkg":
        return _resolve_hvkg(bundle, config, context)
    return config, context


def resolve_information_optimizer_defaults(
    config: AcquisitionConfig,
    opt_config: OptimizeConfig,
) -> OptimizeConfig:
    """Apply optimizer settings required by MES and one-shot KG/HVKG.

    BoTorch qMES uses sequential or cyclic optimization for q > 1. The bochan
    high-level API uses sequential optimization automatically in that case.
    KG and HVKG are one-shot acquisitions and therefore must remain joint;
    an explicit sequential setting is normalized back to ``False``.
    BoTorch's ``optimize_acqf`` automatically selects the specialized one-shot
    initializer for these acquisition classes.
    """

    kind = information_acquisition_kind(config)
    if kind == "mes" and opt_config.q > 1 and not opt_config.sequential:
        return replace(opt_config, sequential=True)
    if kind in {"kg", "hvkg"} and opt_config.sequential:
        return replace(opt_config, sequential=False)
    return opt_config


__all__ = [
    "information_acquisition_kind",
    "is_information_acquisition",
    "resolve_information_acquisition_defaults",
    "resolve_information_optimizer_defaults",
]
