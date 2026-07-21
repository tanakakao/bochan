"""Compatibility exports and lifecycle wrapper for the Web target workflow."""

from __future__ import annotations

import logging
from typing import Any

from . import workflows_tabular as _workflows_tabular
from .logging import current_request_id, get_logger, log_event
from .target_missing_policy import (
    install_workflow_adapters,
    model_variant,
    target_missing_run,
)
from .visualization_sessions import (
    begin_visualization_run,
    discard_visualization_run,
    finalize_visualization_run,
    model_details,
    visualization_options,
)

install_workflow_adapters(_workflows_tabular)

_build_outcome_constraint_config = _workflows_tabular._build_outcome_constraint_config
_figure_payload = _workflows_tabular._figure_payload
_resolve_target_settings = _workflows_tabular._resolve_target_settings
_resolve_targets = _workflows_tabular._resolve_targets
_run_regression_web_workflow = _workflows_tabular.run_regression_web_workflow

LOGGER = get_logger("workflow.details")


def _attach_missing_metadata(
    result: dict[str, Any],
    report: dict[str, Any],
    *,
    variant: str | None = None,
    effective_model_type: str | None = None,
) -> dict[str, Any]:
    """Attach target-missing decisions to the normal workflow metadata."""

    metadata = dict(result.get("metadata") or {})
    resolved_variant = variant or report.get("multitask_variant")
    resolved_model_type = effective_model_type or report.get("effective_model_type")
    metadata.update(
        {
            "target_missing_policy": report.get("policy"),
            "target_missing_detected": bool(report.get("target_missing_detected")),
            "target_missing_counts": dict(report.get("target_missing_counts") or {}),
            "dropped_feature_rows": int(report.get("dropped_feature_rows") or 0),
            "dropped_target_rows": int(report.get("dropped_target_rows") or 0),
            "dropped_all_target_missing_rows": int(
                report.get("dropped_all_target_missing_rows") or 0
            ),
            "acquisition_baseline_completed": bool(
                report.get("acquisition_baseline_completed")
            ),
            "multitask_variant": resolved_variant,
        }
    )
    if report.get("requested_model_type") == "multitask" and resolved_model_type:
        metadata["internal_model_type"] = resolved_model_type
    result["metadata"] = metadata
    return metadata


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Run the Tabular workflow and retain objects needed by Results and Logs."""

    run_id = current_request_id()
    with target_missing_run(request) as missing_report:
        if not run_id:
            result = _run_regression_web_workflow(request, store)
            _attach_missing_metadata(result, missing_report)
            return result

        begin_visualization_run(run_id, request)
        try:
            result = _run_regression_web_workflow(request, store)
            session = finalize_visualization_run(run_id, result)
            variant, effective_model_type = model_variant(session.optimizer.model)
            metadata = _attach_missing_metadata(
                result,
                missing_report,
                variant=variant,
                effective_model_type=effective_model_type,
            )
            details = model_details(session, result)
            details["target_missing_policy"] = metadata.get("target_missing_policy")
            details["target_missing_detected"] = metadata.get(
                "target_missing_detected"
            )
            details["target_missing_counts"] = metadata.get("target_missing_counts")
            details["multitask_variant"] = metadata.get("multitask_variant")
            details["acquisition_baseline_completed"] = metadata.get(
                "acquisition_baseline_completed"
            )
            result["visualization_run_id"] = run_id
            result["visualization_options"] = visualization_options(session)
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
