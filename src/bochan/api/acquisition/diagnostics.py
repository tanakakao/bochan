"""Diagnostics for observation-aware acquisition resolution."""

from __future__ import annotations

from typing import Any

from ..configs import AcquisitionConfig, DataContext, ModelBundle
from .defaults.observations import _as_output_matrix, _scalar_objective_indices


def _row_count(values: Any | None) -> int | None:
    if values is None:
        return None
    try:
        return int(values.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        try:
            return len(values)
        except (TypeError, ValueError):
            return None


def build_acquisition_observation_diagnostics(
    *,
    bundle: ModelBundle,
    config: AcquisitionConfig,
    before_context: DataContext,
    after_context: DataContext,
    observations: Any | None = None,
) -> dict[str, Any]:
    """Describe the observation state used for one acquisition call.

    This function is intentionally read-only. It reports how many objective rows
    are available, whether the automatic acquisition baseline was reduced by
    partial observation semantics, and whether failed / pending experiments were
    excluded before objective-model fitting.
    """

    import torch

    train_X = getattr(bundle, "train_X", None)
    train_Y = _as_output_matrix(getattr(bundle, "train_Y", None))
    training_rows = _row_count(train_X)
    baseline_rows = _row_count(after_context.X_baseline)

    observed_per_output: list[int] = []
    objective_output_indices: list[int] | None = None
    partial_observation = False
    if train_Y is not None:
        finite = torch.isfinite(train_Y)
        observed_per_output = [
            int(value) for value in finite.sum(dim=0).detach().cpu().tolist()
        ]
        partial_observation = not bool(finite.all())
        objective_output_indices = _scalar_objective_indices(
            bundle,
            config,
            n_outputs=int(train_Y.shape[-1]),
        )

    report: dict[str, Any] = {}
    if observations is not None and hasattr(observations, "report"):
        report = dict(observations.report())
        values = report.get("observed_per_output")
        if values is not None:
            observed_per_output = [int(value) for value in values]
        partial_observation = partial_observation or any(
            count < int(report.get("n_success", count))
            for count in observed_per_output
        )

    automatic_baseline = (
        before_context.X_baseline is None or before_context.X_baseline is train_X
    )
    baseline_filtered = bool(
        automatic_baseline
        and training_rows is not None
        and baseline_rows is not None
        and baseline_rows < training_rows
    )

    diagnostics: dict[str, Any] = {
        "training_rows": training_rows,
        "baseline_rows": baseline_rows,
        "baseline_source": "automatic" if automatic_baseline else "explicit",
        "baseline_filtered": baseline_filtered,
        "partial_observation": partial_observation,
        "observed_per_output": observed_per_output,
        "objective_output_indices": objective_output_indices,
        "known_observation_variance": getattr(bundle, "train_Yvar", None) is not None,
    }

    if report:
        diagnostics.update(
            {
                "observation_rows": int(report.get("n_rows", 0)),
                "completed_rows": int(report.get("n_completed", 0)),
                "success_rows": int(report.get("n_success", 0)),
                "failed_rows": int(report.get("n_failed", 0)),
                "pending_rows": int(report.get("n_pending", 0)),
                "failed_excluded_from_objective_training": True,
                "pending_excluded_from_objective_training": True,
            }
        )
    else:
        diagnostics.update(
            {
                "observation_rows": training_rows,
                "completed_rows": training_rows,
                "success_rows": training_rows,
                "failed_rows": 0,
                "pending_rows": 0,
                "failed_excluded_from_objective_training": False,
                "pending_excluded_from_objective_training": False,
            }
        )

    return diagnostics


__all__ = ["build_acquisition_observation_diagnostics"]
