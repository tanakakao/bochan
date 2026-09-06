"""Candidate-generation application service for FastAPI adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..converters import to_data_context, to_optimize_config, to_tensor
from ..target_categories import to_acquisition_config
from .material_models import apply_material_target_task


def _schema_to_dict(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _inject_llm_options(opt_config: Any, request: Any) -> Any:
    """Move transport-level LLM convenience fields into optimizer kwargs."""

    updates: dict[str, Any] = {}
    if getattr(request, "goal", None) is not None:
        updates["goal"] = request.goal
    if getattr(request, "llm_config", None) is not None:
        updates["llm_config"] = _schema_to_dict(request.llm_config)
    if getattr(request, "llm_context", None) is not None:
        updates["llm_context"] = _schema_to_dict(request.llm_context)
    if not updates:
        return opt_config

    optimizer_kwargs = dict(getattr(opt_config, "optimizer_kwargs", {}) or {})
    for key, value in updates.items():
        optimizer_kwargs.setdefault(key, value)
    return replace(opt_config, optimizer_kwargs=optimizer_kwargs)


def _normalize_transport_fidelity_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            int(index): tuple(float(item) for item in values)
            for index, values in value.items()
        }
    return tuple(float(item) for item in value)


def _normalize_transport_fidelity_assignments(value: Any) -> tuple[dict[int, float], ...]:
    return tuple(
        {int(index): float(item) for index, item in assignment.items()}
        for assignment in value
    )


def _normalize_transport_cost_config(value: Any) -> dict[str, Any]:
    config = dict(value)
    kind = str(config.get("kind", "affine")).strip().lower()
    if kind == "callable" or "cost_callable" in config:
        raise ValueError(
            "FidelityCostConfig(kind='callable') is available only in the Python API; "
            "FastAPI cost_config supports serializable cost modes."
        )
    if "cost_model" in config:
        raise ValueError(
            "A pre-built learned cost_model is available only in the Python API. "
            "For FastAPI kind='learned_gp', provide serializable train_X and train_cost."
        )
    return config


def _inject_multifidelity_options(acq_config: Any, opt_config: Any, request: Any) -> tuple[Any, Any]:
    """Merge transport conveniences into canonical core multi-fidelity configs."""

    acqf_kwargs = dict(getattr(acq_config, "acqf_kwargs", {}) or {})
    target_fidelity = getattr(request, "target_fidelity", None)
    cost_config = getattr(request, "cost_config", None)
    if target_fidelity is not None:
        if "target_fidelity" in acqf_kwargs and float(acqf_kwargs["target_fidelity"]) != float(target_fidelity):
            raise ValueError("target_fidelity conflicts with acquisition_config.acqf_kwargs.")
        acqf_kwargs["target_fidelity"] = float(target_fidelity)
    if cost_config is not None:
        transport_cost = _normalize_transport_cost_config(cost_config)
        if "cost_config" in acqf_kwargs and acqf_kwargs["cost_config"] != transport_cost:
            raise ValueError("cost_config conflicts with acquisition_config.acqf_kwargs.")
        acqf_kwargs["cost_config"] = transport_cost
    if acqf_kwargs != dict(getattr(acq_config, "acqf_kwargs", {}) or {}):
        acq_config = replace(acq_config, acqf_kwargs=acqf_kwargs)

    fidelity_values = getattr(request, "fidelity_values", None)
    fidelity_assignments = getattr(request, "fidelity_assignments", None)
    optimize_fidelity = getattr(request, "optimize_fidelity", None)
    if fidelity_values is not None:
        existing = getattr(opt_config, "fidelity_values", None)
        values = _normalize_transport_fidelity_values(fidelity_values)
        if existing is not None and existing != values:
            raise ValueError("fidelity_values conflicts with optimize_config.fidelity_values.")
        opt_config = replace(opt_config, fidelity_values=values)
    if fidelity_assignments is not None:
        existing = getattr(opt_config, "fidelity_assignments", None)
        assignments = _normalize_transport_fidelity_assignments(fidelity_assignments)
        if existing is not None and existing != assignments:
            raise ValueError(
                "fidelity_assignments conflicts with optimize_config.fidelity_assignments."
            )
        opt_config = replace(opt_config, fidelity_assignments=assignments)
    if optimize_fidelity is not None:
        existing = bool(getattr(opt_config, "optimize_fidelity", False))
        if existing and not bool(optimize_fidelity):
            raise ValueError("optimize_fidelity conflicts with optimize_config.optimize_fidelity.")
        opt_config = replace(opt_config, optimize_fidelity=bool(optimize_fidelity))

    active_modes = sum(
        (
            getattr(opt_config, "fidelity_values", None) is not None,
            getattr(opt_config, "fidelity_assignments", None) is not None,
            bool(getattr(opt_config, "optimize_fidelity", False)),
        )
    )
    if active_modes > 1:
        raise ValueError(
            "Specify only one of fidelity_values, fidelity_assignments, or optimize_fidelity=True."
        )
    return acq_config, opt_config


def _candidate_call_args(optimizer: Any, request: Any) -> dict[str, Any]:
    """Convert one HTTP candidate request into canonical optimizer arguments."""

    options = request.tensor_options
    opt_config = _inject_llm_options(to_optimize_config(request.opt_config, options), request)
    opt_config = apply_material_target_task(optimizer, opt_config, getattr(request, "target_task", None))
    acq_config = to_acquisition_config(request.acq_config, options, optimizer=optimizer)
    acq_config, opt_config = _inject_multifidelity_options(acq_config, opt_config, request)
    return {
        "acq_config": acq_config,
        "opt_config": opt_config,
        "data_context": to_data_context(request.data_context, options) if request.data_context is not None else None,
        "bounds": to_tensor(request.bounds, options) if request.bounds is not None else None,
    }


def generate_candidate_result(optimizer: Any, request: Any, *, use_ask: bool = False) -> tuple[Any, Any]:
    """Generate candidates through the canonical optimizer API."""

    method = optimizer.ask if use_ask else optimizer.candidate
    return method(**_candidate_call_args(optimizer, request))


def compare_candidate_results(optimizer: Any, request: Any) -> dict[str, Any]:
    """Evaluate several acquisition configurations using one converted context."""

    options = request.tensor_options
    opt_config = apply_material_target_task(
        optimizer,
        to_optimize_config(request.opt_config, options),
        getattr(request, "target_task", None),
    )
    acq_configs = []
    for config in request.acq_configs:
        acq_config = to_acquisition_config(config, options, optimizer=optimizer)
        acq_config, resolved_opt = _inject_multifidelity_options(acq_config, opt_config, request)
        acq_configs.append(acq_config)
        opt_config = resolved_opt
    data_context = to_data_context(request.data_context, options) if request.data_context is not None else None
    bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
    return optimizer.compare_acquisitions(
        acq_configs=acq_configs,
        opt_config=opt_config,
        data_context=data_context,
        bounds=bounds,
    )


__all__ = ["compare_candidate_results", "generate_candidate_result"]
