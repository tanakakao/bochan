"""Presentation helpers for observation-aware acquisition diagnostics."""

from __future__ import annotations

from typing import Any


def build_acquisition_diagnostics_view(
    diagnostics: dict[str, Any] | None,
    observation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, presentation-ready Web diagnostics payload.

    The function is intentionally read-only. It converts Phase 10/11 diagnostic
    fields into stable summary cards and warning messages that a Web client can
    render without having to understand acquisition internals.
    """

    if diagnostics is None:
        return {
            "available": False,
            "cards": [],
            "warnings": [],
            "details": None,
            "observation_report": observation_report,
        }

    def value(name: str, default: Any = None) -> Any:
        if name in diagnostics:
            return diagnostics.get(name)
        if observation_report is not None:
            return observation_report.get(name, default)
        return default

    cards = [
        {
            "key": "training_rows",
            "label": "Training rows",
            "value": value("training_rows"),
        },
        {
            "key": "baseline_rows",
            "label": "Acquisition baseline",
            "value": value("baseline_rows"),
        },
        {
            "key": "failed_rows",
            "label": "Failed",
            "value": value("failed_rows", 0),
        },
        {
            "key": "pending_rows",
            "label": "Pending",
            "value": value("pending_rows", 0),
        },
        {
            "key": "known_observation_variance",
            "label": "Known Yvar",
            "value": bool(value("known_observation_variance", False)),
        },
    ]

    warnings: list[str] = []
    if bool(diagnostics.get("baseline_filtered")):
        warnings.append(
            "The automatic acquisition baseline excludes rows without the selected objective observation."
        )
    if bool(diagnostics.get("partial_observation")):
        warnings.append("Some target outputs are only partially observed.")
    if int(value("failed_rows", 0) or 0) > 0 and bool(
        diagnostics.get("failed_excluded_from_objective_training")
    ):
        warnings.append("Failed experiments are excluded from objective-model training.")
    if int(value("pending_rows", 0) or 0) > 0 and bool(
        diagnostics.get("pending_excluded_from_objective_training")
    ):
        warnings.append("Pending experiments are excluded from objective-model training.")

    details = {
        "baseline_source": diagnostics.get("baseline_source"),
        "baseline_filtered": bool(diagnostics.get("baseline_filtered", False)),
        "partial_observation": bool(diagnostics.get("partial_observation", False)),
        "observed_per_output": list(diagnostics.get("observed_per_output") or []),
        "objective_output_indices": list(
            diagnostics.get("objective_output_indices") or []
        ),
        "known_observation_variance": bool(
            diagnostics.get("known_observation_variance", False)
        ),
    }

    return {
        "available": True,
        "cards": cards,
        "warnings": warnings,
        "details": details,
        "observation_report": observation_report,
    }


def acquisition_diagnostics_view_for_optimizer(optimizer: Any) -> dict[str, Any]:
    """Build the Web view from an optimizer without mutating its state."""

    diagnostics = getattr(optimizer, "last_acquisition_diagnostics", None)
    observations = getattr(optimizer, "observations", None)
    observation_report = None
    if observations is not None and callable(getattr(observations, "report", None)):
        observation_report = observations.report()
    return build_acquisition_diagnostics_view(diagnostics, observation_report)


__all__ = [
    "acquisition_diagnostics_view_for_optimizer",
    "build_acquisition_diagnostics_view",
]
