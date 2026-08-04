"""Keep coordinate PI for outputs without composition-element importance."""

from __future__ import annotations

from typing import Any


def install_composition_importance_output_compat() -> None:
    """Limit coordinate-to-group replacement to supported regression outputs."""

    from . import composition_feature_importance_views as views

    if getattr(views, "_composition_output_compat_installed", False):
        return
    original = views._replace_predictive_summary

    def adapted(
        result: dict[str, Any],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        coordinate_features = {
            str(value) for value in payload.get("coordinate_features") or ()
        }
        supported_outputs = {
            str(row.get("output_name")) for row in payload.get("overall") or ()
        }
        preserved = [
            dict(row)
            for row in list(result.get("feature_importance_summary") or ())
            if str(row.get("importance_kind")) == "predictive"
            and str(row.get("feature")) in coordinate_features
            and str(row.get("output_name")) not in supported_outputs
        ]
        rows = original(result, payload)
        existing = {
            (
                str(row.get("output_name")),
                str(row.get("importance_kind")),
                str(row.get("method")),
                str(row.get("feature")),
            )
            for row in rows
        }
        rows.extend(
            row
            for row in preserved
            if (
                str(row.get("output_name")),
                str(row.get("importance_kind")),
                str(row.get("method")),
                str(row.get("feature")),
            )
            not in existing
        )
        result["feature_importance_summary"] = rows
        return rows

    views._replace_predictive_summary = adapted
    views._composition_output_compat_installed = True


__all__ = ["install_composition_importance_output_compat"]
