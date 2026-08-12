"""Candidate-generation application service for FastAPI adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..converters import to_data_context, to_optimize_config, to_tensor
from ..tabular_compat import to_acquisition_config


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


def _candidate_call_args(optimizer: Any, request: Any) -> dict[str, Any]:
    """Convert one HTTP candidate request into canonical optimizer arguments."""

    options = request.tensor_options
    return {
        "acq_config": to_acquisition_config(
            request.acq_config,
            options,
            optimizer=optimizer,
        ),
        "opt_config": _inject_llm_options(
            to_optimize_config(request.opt_config, options),
            request,
        ),
        "data_context": (
            to_data_context(request.data_context, options)
            if request.data_context is not None
            else None
        ),
        "bounds": (
            to_tensor(request.bounds, options)
            if request.bounds is not None
            else None
        ),
    }


def generate_candidate_result(
    optimizer: Any,
    request: Any,
    *,
    use_ask: bool = False,
) -> tuple[Any, Any]:
    """Generate candidates through the canonical optimizer API."""

    method = optimizer.ask if use_ask else optimizer.candidate
    return method(**_candidate_call_args(optimizer, request))


def compare_candidate_results(
    optimizer: Any,
    request: Any,
) -> dict[str, Any]:
    """Evaluate several acquisition configurations using one converted context."""

    options = request.tensor_options
    acq_configs = [
        to_acquisition_config(
            config,
            options,
            optimizer=optimizer,
        )
        for config in request.acq_configs
    ]
    opt_config = to_optimize_config(request.opt_config, options)
    data_context = (
        to_data_context(request.data_context, options)
        if request.data_context is not None
        else None
    )
    bounds = to_tensor(request.bounds, options) if request.bounds is not None else None
    return optimizer.compare_acquisitions(
        acq_configs=acq_configs,
        opt_config=opt_config,
        data_context=data_context,
        bounds=bounds,
    )


__all__ = ["compare_candidate_results", "generate_candidate_result"]
