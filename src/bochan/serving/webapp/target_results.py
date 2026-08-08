"""Candidate serialization and visualization helpers for web target workflows."""

from __future__ import annotations

import json
import re
from typing import Any

from .target_settings import _as_2d


def _figure_payload(
    figure: Any,
    *,
    figure_id: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    """Convert a Plotly figure into a JSON-safe web response payload."""

    figure.update_layout(
        title=title,
        autosize=True,
        width=None,
        margin=dict(l=60, r=30, t=70, b=60),
    )
    return {
        "id": figure_id,
        "title": title,
        "description": description,
        "figure": json.loads(figure.to_json()),
    }


def _build_feature_importance_visualizations(
    result: Any,
    *,
    visualization_config: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Adapt visualization-layer feature-importance figures for the Web API.

    Args:
        result: Existing core or cross-validated importance result.
        visualization_config: Presentation-only request settings.

    Returns:
        Figure payloads and non-fatal per-view warnings.
    """
    from bochan.visualization import build_feature_importance_figures

    def getter(name: str, default: Any) -> Any:
        """Read one presentation setting from the request object."""
        return getattr(visualization_config, name, default)

    try:
        figures = build_feature_importance_figures(
            result,
            include_predictive=getter("include_predictive", True),
            include_noise=getter("include_noise", True),
            include_classwise=getter("include_classwise", False),
            normalized=getter("normalized", False),
            top_k=getter("top_k", 15),
            rank_by=getter("rank_by", "value"),
        )
    except Exception as exc:
        return [], [f"Feature-importance visualization failed: {exc}"]
    payloads, warnings = [], []
    for key, figure in figures.items():
        try:
            payloads.append(
                _figure_payload(
                    figure,
                    figure_id=f"feature-importance-{_safe_figure_id(key)}",
                    title=str(figure.layout.title.text or "Feature importance"),
                    description="Permutation importance; error bars are repeat standard deviation, or between-fold standard deviation for CV.",
                )
            )
        except Exception as exc:
            warnings.append(f"Feature-importance figure {key!r} failed: {exc}")
    return payloads, warnings


def _safe_figure_id(value: str) -> str:
    """Return a stable HTML-friendly identifier fragment."""

    normalized = re.sub(r"[^0-9A-Za-z_-]+", "-", str(value)).strip("-")
    return normalized or "target"


def _flatten_acq_values(acq_value: Any) -> list[float]:
    """Return acquisition values as a flat Python list."""

    try:
        values = acq_value.detach().cpu().reshape(-1).tolist()
    except Exception:
        values = [acq_value]
    return [float(value) for value in values if value is not None]


def _batch_acq_value(acq_value: Any, n: int) -> float | None:
    """Return a scalar value only when it represents the complete q-batch."""

    values = _flatten_acq_values(acq_value)
    return values[0] if n > 1 and len(values) == 1 else None


def _broadcast_acq_values(acq_value: Any, n: int) -> list[float | None]:
    """Return only genuine per-candidate acquisition values."""

    values = _flatten_acq_values(acq_value)
    if len(values) == n:
        return values
    if n == 1 and values:
        return [values[0]]
    return [None for _ in range(n)]


def _display_predictions(
    optimizer: Any,
    X: Any,
    *,
    target_columns: list[str],
    target_metadata: dict[str, dict[str, Any]],
    hybrid_model: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return display-scale means/stds and class probabilities by target."""

    import torch

    from bochan.acquisition.binary.epistemic import binary_probability_moments

    n_rows = int(X.shape[0])
    posterior = optimizer.model.posterior(X, output_mode="mean") if hybrid_model else optimizer.model.posterior(X)
    means = _as_2d(posterior.mean, n_rows=n_rows)
    variances = _as_2d(posterior.variance, n_rows=n_rows).clamp_min(0)

    output: dict[str, dict[str, Any]] = {}
    class_probabilities: dict[str, Any] = {}
    for index, target in enumerate(target_columns):
        meta = target_metadata[target]
        task = str(meta["internal_task"])
        if task == "binary":
            if hybrid_model:
                probability_posterior = optimizer.model.posterior(
                    X,
                    output_indices=[index],
                    output_mode="probability",
                )
                probability_mean = _as_2d(
                    probability_posterior.mean,
                    n_rows=n_rows,
                )
                epistemic_variance = _as_2d(
                    probability_posterior.variance,
                    n_rows=n_rows,
                ).clamp_min(0)
                mean = probability_mean[:, 0]
                variance = epistemic_variance[:, 0]
                class_probabilities[target] = optimizer.model.class_probs_list(
                    X,
                    output_indices=[target],
                )[0]
            else:
                probability_mean, epistemic_variance, _, _ = binary_probability_moments(
                    optimizer.model,
                    X,
                )
                mean = _as_2d(probability_mean, n_rows=n_rows)[:, 0]
                variance = _as_2d(
                    epistemic_variance,
                    n_rows=n_rows,
                ).clamp_min(0)[:, 0]
            prediction_space = "probability"
        elif task == "ordinal" and hybrid_model:
            probs = optimizer.model.class_probs_list(
                X,
                output_indices=[target],
            )[0]
            ranks = torch.arange(
                int(meta["num_classes"]),
                dtype=probs.dtype,
                device=probs.device,
            )
            mean = (probs * ranks).sum(dim=-1)
            variance = (probs * (ranks - mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
            class_probabilities[target] = probs
            prediction_space = "expected_rank"
        else:
            mean = means[:, index]
            variance = variances[:, index]
            prediction_space = "probability" if task in {"binary", "multiclass"} else "outcome"
            if task in {"binary", "multiclass"} and hybrid_model:
                class_probabilities[target] = optimizer.model.class_probs_list(
                    X,
                    output_indices=[target],
                )[0]
        output[target] = {
            "mean": mean,
            "std": variance.sqrt(),
            "prediction_space": prediction_space,
        }
    return output, class_probabilities


def _setting_constraint_result(
    setting: dict[str, Any],
    meta: dict[str, Any],
    *,
    predicted_mean: float,
    row_index: int,
    class_probabilities: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one displayed candidate against an optional target constraint."""

    del row_index, class_probabilities
    goal = str(setting["goal"])
    target = str(setting["target"])
    task = str(meta["internal_task"])
    if goal in {"none", "target"}:
        return {
            "target": target,
            "goal": goal,
            "value": meta.get("configured_value"),
            "predicted_mean": predicted_mean,
            "ok": True,
            "violation": 0.0,
        }

    threshold = float(meta["class_index"]) if task == "ordinal" else float(meta["configured_value"])
    if goal == "above":
        ok = predicted_mean >= threshold - 1e-8
        violation = max(threshold - predicted_mean, 0.0)
    else:
        ok = predicted_mean <= threshold + 1e-8
        violation = max(predicted_mean - threshold, 0.0)
    return {
        "target": target,
        "goal": goal,
        "value": meta["configured_value"] if task == "ordinal" else threshold,
        "threshold_rank": threshold if task == "ordinal" else None,
        "target_classes": meta.get("target_classes", []),
        "predicted_mean": predicted_mean,
        "ok": bool(ok),
        "violation": float(violation),
    }


def _candidate_rows(
    *,
    optimizer: Any,
    candidates: Any,
    acq_value: Any,
    encoded: dict[str, Any],
    target_columns: list[str],
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
    hybrid_model: bool,
) -> list[dict[str, Any]]:
    """Build candidate rows with one prediction channel per target."""

    from bochan.serving.fastapi.converters import to_serializable

    prediction_tensors, class_probabilities = _display_predictions(
        optimizer,
        candidates,
        target_columns=target_columns,
        target_metadata=target_metadata,
        hybrid_model=hybrid_model,
    )
    n_candidates = int(candidates.shape[0])
    candidate_values = candidates.detach().cpu().tolist()
    acq_values = _broadcast_acq_values(acq_value, n_candidates)
    inverse_maps = encoded["inverse_category_maps"]
    feature_columns = encoded["feature_columns"]
    setting_by_target = {setting["target"]: setting for setting in target_settings}

    rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(candidate_values):
        decoded: dict[str, Any] = {}
        encoded_values: dict[str, float] = {}
        for feature_index, column in enumerate(feature_columns):
            value = float(values[feature_index])
            encoded_values[column] = value
            if column in inverse_maps:
                decoded[column] = inverse_maps[column].get(int(round(value)), str(int(round(value))))
            else:
                decoded[column] = value

        predictions: dict[str, dict[str, Any]] = {}
        constraint_results: list[dict[str, Any]] = []
        for target in target_columns:
            values_by_target = prediction_tensors[target]
            mean = float(values_by_target["mean"][row_index].detach().cpu())
            std = float(values_by_target["std"][row_index].detach().cpu())
            predictions[target] = {
                "mean": mean,
                "std": std,
                "prediction_space": values_by_target["prediction_space"],
            }
            setting = setting_by_target[target]
            if not setting.get("legacy") and setting.get("goal") in {"above", "below"}:
                constraint_results.append(
                    _setting_constraint_result(
                        setting,
                        target_metadata[target],
                        predicted_mean=mean,
                        row_index=row_index,
                        class_probabilities=class_probabilities,
                    )
                )

        first_prediction = predictions[target_columns[0]]
        rows.append(
            {
                "rank": row_index + 1,
                "values": decoded,
                "encoded_values": encoded_values,
                "acq_value": acq_values[row_index],
                "predictions": predictions,
                "predicted_target_mean": first_prediction["mean"],
                "predicted_target_std": first_prediction["std"],
                "constraints_ok": all(item["ok"] for item in constraint_results),
                "constraints": constraint_results,
                "raw": {"candidate": to_serializable(candidates[row_index])},
            }
        )
    return rows


def _build_visualizations(
    *,
    optimizer: Any,
    train_x: Any,
    original_targets: Any,
    target_columns: list[str],
    target_metadata: dict[str, dict[str, Any]],
    hybrid_model: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build regression YY plots; retain explicit warnings for discrete outputs."""

    figures: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        display, _ = _display_predictions(
            optimizer,
            train_x,
            target_columns=target_columns,
            target_metadata=target_metadata,
            hybrid_model=hybrid_model,
        )
    except Exception as exc:
        return [], [f"可視化用予測を生成できませんでした: {exc}"]

    for target in target_columns:
        meta = target_metadata[target]
        if meta["internal_task"] != "regression":
            warnings.append(f"{target}: 分類・順序回帰の専用可視化は未接続のため、候補テーブルで確認してください。")
            continue
        try:
            import plotly.graph_objects as go

            observed = original_targets[target].to_numpy(dtype=float)
            predicted = display[target]["mean"].detach().cpu().numpy()
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=observed,
                    y=predicted,
                    mode="markers",
                    name="prediction",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    name="ideal",
                )
            )
            figure.update_xaxes(title="実測値")
            figure.update_yaxes(title="予測値")
            figures.append(
                _figure_payload(
                    figure,
                    figure_id=f"{_safe_figure_id(target)}-yyplot",
                    title=f"{target}: 実測値と予測値",
                    description="学習データに対する予測平均を実測値と比較します。",
                )
            )
        except Exception as exc:
            warnings.append(f"{target}のYY plotを生成できませんでした: {exc}")
    return figures, warnings


def _best_observed(
    original_targets: Any,
    encoded_targets: Any,
    target_settings: list[dict[str, Any]],
    target_metadata: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Return a compact target-wise observed summary in the display scale."""

    values: dict[str, float] = {}
    for setting in target_settings:
        target = str(setting["target"])
        meta = target_metadata[target]
        goal = str(setting["goal"])
        if meta["internal_task"] == "regression":
            series = original_targets[target]
            if goal == "below":
                values[target] = float(series.min())
            elif goal == "target" and not setting.get("legacy"):
                target_value = float(meta["configured_value"])
                index = (series - target_value).abs().idxmin()
                values[target] = float(series.loc[index])
            else:
                values[target] = float(series.max())
        elif meta["internal_task"] in {"binary", "multiclass"}:
            class_indices = [int(index) for index in meta.get("class_indices", [])]
            values[target] = float(encoded_targets[target].isin(class_indices).mean())
        else:
            series = encoded_targets[target]
            if goal == "below":
                values[target] = float(series.min())
            elif goal == "target":
                target_indices = [int(index) for index in meta.get("class_indices", [])]
                values[target] = float(target_indices[0]) if target_indices else float(series.max())
            else:
                values[target] = float(series.max())
    return values


__all__ = [
    "_best_observed",
    "_build_visualizations",
    "_candidate_rows",
    "_figure_payload",
]
