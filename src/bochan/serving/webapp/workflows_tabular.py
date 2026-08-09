"""Observation-aware entrypoint for the React Web tabular workflow."""

from __future__ import annotations

from typing import Any

from .workflows_tabular_core import (
    _build_outcome_constraint_config,
    _figure_payload,
    _resolve_target_settings,
    _resolve_targets,
    run_regression_web_workflow as _run_core_workflow,
)


def _normalized_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _requested_acquisition_name(request: Any) -> str:
    acquisition = getattr(request, "acquisition", None)
    for value in (
        getattr(acquisition, "name", None),
        getattr(acquisition, "acq_name", None),
        getattr(request, "acquisition_name", None),
    ):
        if value:
            return str(value)
    return ""


def _validate_partial_multiobjective_acquisition(request: Any, store: Any) -> None:
    """Reject acquisitions that require a fully observed empirical Pareto set."""

    if _normalized_name(getattr(request, "model_type", "")) != "multitask":
        return
    target_columns, _ = _resolve_targets(request)
    if len(target_columns) < 2:
        return
    record = store.get(request.dataset_id)
    if not bool(record.data[target_columns].isna().any().any()):
        return

    acquisition = _normalized_name(_requested_acquisition_name(request))
    if acquisition in {
        "ehvi",
        "qehvi",
        "nparego",
        "qnparego",
        "nsgaii",
        "nsga2",
    }:
        raise ValueError(
            "Partially observed multi-objective Web optimization requires NEHVI. "
            "EHVI, NParEGO, and NSGA-II would otherwise need a fully observed "
            "empirical objective matrix. Missing objectives are not imputed from "
            "posterior means."
        )


def run_regression_web_workflow(request: Any, store: Any) -> dict[str, Any]:
    """Run the Web workflow after validating partial-observation semantics."""

    _validate_partial_multiobjective_acquisition(request, store)
    return _run_core_workflow(request, store)


__all__ = [
    "_build_outcome_constraint_config",
    "_figure_payload",
    "_resolve_target_settings",
    "_resolve_targets",
    "_validate_partial_multiobjective_acquisition",
    "run_regression_web_workflow",
]
