"""Compatibility exports and lifecycle wrapper for the Web target workflow."""

from __future__ import annotations

import logging
from typing import Any

from .logging import current_request_id, get_logger, log_event
from .visualization_sessions import (
    begin_visualization_run,
    discard_visualization_run,
    finalize_visualization_run,
    model_details,
    visualization_options,
)
from .workflows_tabular import (
    _build_outcome_constraint_config,
    _figure_payload,
    _resolve_target_settings,
    _resolve_targets,
    run_regression_web_workflow as _run_regression_web_workflow,
)

LOGGER = get_logger("workflow.details")


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Run the Tabular workflow and retain objects needed by Results and Logs."""

    run_id = current_request_id()
    if not run_id:
        return _run_regression_web_workflow(request, store)

    begin_visualization_run(run_id, request)
    try:
        result = _run_regression_web_workflow(request, store)
        session = finalize_visualization_run(run_id, result)
        details = model_details(session, result)
        result["visualization_run_id"] = run_id
        result["visualization_options"] = visualization_options(session)
        metadata = dict(result.get("metadata") or {})
        metadata["model_details"] = details
        metadata["visualization_session"] = "in_memory"
        result["metadata"] = metadata
        log_event(
            LOGGER,
            logging.INFO,
            "model_details",
            "Actual fitted model and acquisition details",
            model_details=details,
        )
        return result
    except Exception:
        discard_visualization_run(run_id)
        raise


__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_target_settings",
    "_resolve_targets",
    "run_regression_web_workflow",
]
