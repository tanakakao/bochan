"""Serialize composition importance without importing visualization extras."""

from __future__ import annotations

from typing import Any

_GROUP_NAME = "組成全体"


def _records_by_scope(
    importance: Any,
    targets: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read core PI objects directly and normalize element rows internally."""

    overall: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    for target in targets:
        output = importance.outputs[target]
        method = output.predictive_methods["permutation"]
        records: list[dict[str, Any]] = []
        for entry in method.entries.values():
            summary = entry.importance
            records.append(
                {
                    "output_name": target,
                    "task_type": output.task_type,
                    "importance_kind": "predictive",
                    "method": "permutation",
                    "class_label": None,
                    "feature": entry.name,
                    "rank": summary.rank,
                    "indices": list(entry.indices),
                    "feature_type": entry.feature_type,
                    "role": entry.role,
                    "mean": summary.mean,
                    "std": summary.std,
                    "minimum": summary.minimum,
                    "maximum": summary.maximum,
                    "median": summary.median,
                    "normalized_mean": summary.normalized_mean,
                    "normalized_std": entry.metadata.get("normalized_std"),
                    "baseline_metric": entry.baseline_metric,
                    "metric_name": entry.metric_name,
                    "metric_direction": entry.metric_direction,
                    "n_repeats": importance.n_repeats,
                    "evaluation_source": "training",
                }
            )

        element_records = [
            record for record in records if str(record.get("feature")) != _GROUP_NAME
        ]
        positive_total = sum(
            max(float(record.get("mean") or 0.0), 0.0)
            for record in element_records
        )
        for record in records:
            if str(record.get("feature")) == _GROUP_NAME:
                record["normalized_mean"] = None
                overall.append(record)
                continue
            mean = float(record.get("mean") or 0.0)
            record["normalized_mean"] = (
                max(mean, 0.0) / positive_total if positive_total > 0.0 else 0.0
            )
            record["label"] = f"{record['feature']} 比率"
            elements.append(record)
    return overall, elements


def install_composition_importance_records_compat() -> None:
    """Replace the Plotly-coupled record conversion before workflow execution."""

    from . import composition_feature_importance as module

    if getattr(module, "_records_compat_installed", False):
        return
    module._records_by_scope = _records_by_scope
    module._records_compat_installed = True


__all__ = ["install_composition_importance_records_compat"]
