"""Compatibility exports and lifecycle wrapper for the Web target workflow."""

from __future__ import annotations

import copy
import logging
from importlib import import_module
from typing import Any

from . import target_results as _target_results
from . import target_settings as _target_settings
from .logging import current_request_id, get_logger, log_event
from .model_reuse import model_reuse_run, prepare_model_reuse_request
from .prediction_shapes import normalize_prediction_rows
from .risk_settings import (
    attach_web_risk_metadata,
    install_web_risk_adapters,
    web_risk_run,
)
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

# ``app.py`` imports visualization helpers before this compatibility module, so
# replace both already-bound helper references before loading the tabular workflow.
_target_settings._as_2d = normalize_prediction_rows
_target_results._as_2d = normalize_prediction_rows
_workflows_tabular = import_module(".workflows_tabular", package=__package__)
_workflows_tabular._as_2d = normalize_prediction_rows

install_workflow_adapters(_workflows_tabular)
install_web_risk_adapters(_workflows_tabular)

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
    """Attach feature / target missing-value decisions to workflow metadata."""

    metadata = dict(result.get("metadata") or {})
    resolved_variant = variant or report.get("multitask_variant")
    resolved_model_type = effective_model_type or report.get("effective_model_type")
    metadata.update(
        {
            "feature_missing_strategy": report.get("feature_missing_strategy"),
            "feature_missing_detected": bool(report.get("feature_missing_detected")),
            "feature_missing_counts": dict(report.get("feature_missing_counts") or {}),
            "feature_impute_values": dict(report.get("feature_impute_values") or {}),
            "continuous_impute_strategy": report.get("continuous_impute_strategy"),
            "categorical_impute_strategy": report.get("categorical_impute_strategy"),
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


def _attach_reuse_metadata(
    result: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Attach whether the fitted model stage was skipped."""

    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "model_reuse_requested": bool(report.get("requested")),
            "model_reused": bool(report.get("model_reused")),
            "fit_skipped": bool(report.get("fit_skipped")),
            "model_reuse_source_run_id": report.get("source_run_id"),
        }
    )
    result["metadata"] = metadata
    return metadata


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Run the Tabular workflow and retain objects needed by Results and Logs."""

    processing_request, source_run_id = prepare_model_reuse_request(request)
    run_id = current_request_id()
    with (
        target_missing_run(processing_request) as missing_report,
        model_reuse_run(processing_request, source_run_id) as reuse_report,
        web_risk_run(processing_request) as risk_report,
    ):
        if not run_id:
            result = _run_regression_web_workflow(processing_request, store)
            _attach_missing_metadata(result, missing_report)
            _attach_reuse_metadata(result, reuse_report)
            attach_web_risk_metadata(result, risk_report)
            return result

        begin_visualization_run(run_id, processing_request)
        try:
            result = _run_regression_web_workflow(processing_request, store)
            session = finalize_visualization_run(run_id, result)
            session.request_details["request_payload"] = (
                processing_request.model_dump(exclude_none=False)
                if hasattr(processing_request, "model_dump")
                else dict(vars(processing_request))
            )
            variant, effective_model_type = model_variant(session.optimizer.model)
            metadata = _attach_missing_metadata(
                result,
                missing_report,
                variant=variant,
                effective_model_type=effective_model_type,
            )
            metadata = _attach_reuse_metadata(result, reuse_report)
            metadata = attach_web_risk_metadata(result, risk_report)
            details = model_details(session, result)
            details["feature_missing_strategy"] = metadata.get(
                "feature_missing_strategy"
            )
            details["feature_missing_detected"] = metadata.get(
                "feature_missing_detected"
            )
            details["feature_missing_counts"] = metadata.get(
                "feature_missing_counts"
            )
            details["feature_impute_values"] = metadata.get("feature_impute_values")
            details["target_missing_policy"] = metadata.get("target_missing_policy")
            details["target_missing_detected"] = metadata.get(
                "target_missing_detected"
            )
            details["target_missing_counts"] = metadata.get("target_missing_counts")
            details["multitask_variant"] = metadata.get("multitask_variant")
            details["acquisition_baseline_completed"] = metadata.get(
                "acquisition_baseline_completed"
            )
            details["model_reused"] = metadata.get("model_reused")
            details["fit_skipped"] = metadata.get("fit_skipped")
            details["model_reuse_source_run_id"] = metadata.get(
                "model_reuse_source_run_id"
            )
            details["input_perturbation_risk_type"] = metadata.get(
                "input_perturbation_risk_type"
            )
            details["input_perturbation_risk_alpha"] = metadata.get(
                "input_perturbation_risk_alpha"
            )
            details["input_perturbation_risk_enabled"] = metadata.get(
                "input_perturbation_risk_enabled"
            )
            result["visualization_run_id"] = run_id
            result["visualization_options"] = visualization_options(session)
            metadata["model_details"] = details
            metadata["visualization_session"] = "in_memory"
            result["metadata"] = metadata
            session.result = copy.deepcopy(result)
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
