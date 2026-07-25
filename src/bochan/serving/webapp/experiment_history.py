"""Experiment-cycle history storage and objective-progress visualizations."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from statistics import fmean
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCycleRequest(BaseModel):
    """One completed experiment cycle appended to a Web dataset."""

    model_config = ConfigDict(extra="forbid")

    parent_dataset_id: str
    dataset_id: str
    dataset_name: str
    source_run_id: str | None = None
    append_mode: Literal["manual", "import"]
    n_rows_before: int = Field(ge=0)
    n_rows_after: int = Field(ge=0)
    rows: list[dict[str, Any]] = Field(min_length=1)
    feature_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(min_length=1)
    target_settings: list[dict[str, Any]] = Field(default_factory=list)
    model: dict[str, Any] = Field(default_factory=dict)
    acquisition: dict[str, Any] = Field(default_factory=dict)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    best_observed_before: dict[str, float | None] = Field(default_factory=dict)
    candidate_count: int = Field(default=0, ge=0)
    notes: str | None = None


def _safe_float(value: Any) -> float | None:
    """Convert a finite scalar to float without treating booleans as outcomes."""

    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted != converted or converted in {float("inf"), float("-inf")}:
        return None
    return converted


def _same_value(left: Any, right: Any) -> bool:
    """Compare categorical values robustly across JSON number/string encodings."""

    return str(left) == str(right)


def _target_metric_values(
    rows: list[dict[str, Any]],
    target: str,
    setting: dict[str, Any],
) -> tuple[list[float], str, list[float]]:
    """Convert observed outcomes to a graphable metric and target references."""

    task_type = str(setting.get("task_type") or "regression")
    goal = str(setting.get("goal") or "none")

    if task_type == "classification":
        desired = list(setting.get("target_classes") or [])
        if not desired and setting.get("target_class") is not None:
            desired = [setting["target_class"]]
        values = [
            1.0 if any(_same_value(row.get(target), candidate) for candidate in desired) else 0.0
            for row in rows
            if row.get(target) is not None
        ]
        return values, "desired_class_rate", [1.0]

    if task_type == "ordinal":
        order = list(setting.get("class_order") or [])
        rank_by_value = {str(value): float(index) for index, value in enumerate(order)}
        values = [
            rank_by_value[str(row[target])]
            for row in rows
            if row.get(target) is not None and str(row[target]) in rank_by_value
        ]
        references = [
            rank_by_value[str(value)]
            for value in list(setting.get("target_values") or [])
            if str(value) in rank_by_value
        ]
        if not references and goal in {"above", "below"} and str(setting.get("value")) in rank_by_value:
            references = [rank_by_value[str(setting["value"])]]
        return values, "ordinal_rank", references

    values = [
        converted
        for row in rows
        if (converted := _safe_float(row.get(target))) is not None
    ]
    references: list[float] = []
    if goal == "target":
        configured = _safe_float(setting.get("value"))
        if configured is not None:
            references = [configured]
    return values, "outcome", references


def _best_value(
    values: list[float],
    *,
    direction: str,
    goal: str,
    references: list[float],
) -> float | None:
    """Select a cycle best value using direction or target-distance semantics."""

    if not values:
        return None
    if goal == "target" and references:
        return min(values, key=lambda value: min(abs(value - reference) for reference in references))
    return min(values) if direction == "minimize" else max(values)


def _target_summaries(request: ExperimentCycleRequest) -> dict[str, dict[str, Any]]:
    """Summarize measured objective values for one experiment cycle."""

    setting_by_target = {
        str(setting.get("target")): setting
        for setting in request.target_settings
        if setting.get("target") is not None
    }
    summaries: dict[str, dict[str, Any]] = {}
    for target in request.target_columns:
        setting = setting_by_target.get(target, {})
        direction = str(setting.get("direction") or "maximize")
        goal = str(setting.get("goal") or "none")
        values, metric_kind, references = _target_metric_values(request.rows, target, setting)
        summaries[target] = {
            "task_type": str(setting.get("task_type") or "regression"),
            "metric_kind": metric_kind,
            "direction": direction,
            "goal": goal,
            "references": references,
            "count": len(values),
            "mean": fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "best": _best_value(
                values,
                direction=direction,
                goal=goal,
                references=references,
            ),
        }
    return summaries


def _is_better(candidate: float, current: float, summary: dict[str, Any]) -> bool:
    """Return whether candidate improves the current cumulative objective value."""

    references = [float(value) for value in summary.get("references") or []]
    if summary.get("goal") == "target" and references:
        candidate_distance = min(abs(candidate - reference) for reference in references)
        current_distance = min(abs(current - reference) for reference in references)
        return candidate_distance < current_distance
    if summary.get("direction") == "minimize":
        return candidate < current
    return candidate > current


def _cycle_hover_text(cycle: dict[str, Any]) -> str:
    """Build compact Plotly hover text for one cycle."""

    model_name = str(cycle.get("model", {}).get("type") or "—")
    acquisition_name = str(cycle.get("acquisition", {}).get("name") or "—")
    created_at = str(cycle.get("created_at") or "")
    return (
        f"サイクル {cycle['cycle_number']}<br>"
        f"{created_at}<br>"
        f"モデル: {model_name}<br>"
        f"獲得関数: {acquisition_name}<br>"
        f"追加件数: {cycle.get('appended_rows', 0)}"
    )


def _history_visualizations(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one Plotly objective-progress figure per target."""

    if not cycles:
        return []

    import plotly.graph_objects as go

    targets: list[str] = []
    for cycle in cycles:
        for target in cycle.get("target_columns", []):
            if target not in targets:
                targets.append(target)

    visualizations: list[dict[str, Any]] = []
    for target in targets:
        usable = [
            cycle
            for cycle in cycles
            if cycle.get("target_summary", {}).get(target, {}).get("count", 0) > 0
        ]
        if not usable:
            continue

        cycle_numbers = [int(cycle["cycle_number"]) for cycle in usable]
        cycle_best = [float(cycle["target_summary"][target]["best"]) for cycle in usable]
        cycle_mean = [float(cycle["target_summary"][target]["mean"]) for cycle in usable]
        hover = [_cycle_hover_text(cycle) for cycle in usable]

        first_summary = usable[0]["target_summary"][target]
        has_initial_cycle = any(int(cycle["cycle_number"]) == 0 for cycle in usable)
        baseline = None if has_initial_cycle else _safe_float(
            usable[0].get("best_observed_before", {}).get(target)
        )
        cumulative_x: list[int] = []
        cumulative_y: list[float] = []
        cumulative_hover: list[str] = []
        current = baseline
        if current is not None:
            cumulative_x.append(0)
            cumulative_y.append(current)
            cumulative_hover.append("初期データのベスト")
        for cycle, value in zip(usable, cycle_best, strict=True):
            if current is None or _is_better(value, current, first_summary):
                current = value
            cumulative_x.append(int(cycle["cycle_number"]))
            cumulative_y.append(float(current))
            cumulative_hover.append(_cycle_hover_text(cycle))

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=cycle_numbers,
                y=cycle_best,
                mode="lines+markers",
                name="サイクル内ベスト",
                text=hover,
                hovertemplate="%{text}<br>値: %{y}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=cycle_numbers,
                y=cycle_mean,
                mode="lines+markers",
                name="サイクル平均",
                text=hover,
                hovertemplate="%{text}<br>平均: %{y}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=cumulative_x,
                y=cumulative_y,
                mode="lines+markers",
                name="累積ベスト",
                text=cumulative_hover,
                hovertemplate="%{text}<br>累積ベスト: %{y}<extra></extra>",
            )
        )
        figure.update_layout(
            title=f"{target}: 実験サイクルごとの目的変数推移",
            autosize=True,
            width=None,
            margin=dict(l=64, r=30, t=70, b=58),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        figure.update_xaxes(title="実験サイクル", dtick=1)
        figure.update_yaxes(title=target)
        visualizations.append(
            {
                "id": f"experiment-history-{target}",
                "target": target,
                "title": f"{target}のサイクル推移",
                "description": "各サイクルの実測ベスト、平均、初期データを含む累積ベストを表示します。",
                "figure": json.loads(figure.to_json()),
            }
        )
    return visualizations


class ExperimentHistoryStore:
    """In-memory experiment history linked through parent and updated dataset IDs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_dataset: dict[str, dict[str, Any]] = {}
        self._by_cycle_id: dict[str, dict[str, Any]] = {}

    def list_for_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return the ordered lineage ending at ``dataset_id``."""

        with self._lock:
            current = dataset_id
            reversed_cycles: list[dict[str, Any]] = []
            visited: set[str] = set()
            while current in self._by_dataset and current not in visited:
                visited.add(current)
                cycle = self._by_dataset[current]
                reversed_cycles.append(cycle)
                current = str(cycle["parent_dataset_id"])
            reversed_cycles.reverse()
            return deepcopy(reversed_cycles)

    def add(
        self,
        request: ExperimentCycleRequest,
        *,
        initial_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record a completed cycle and assign its lineage cycle number.

        Args:
            request: Completed experiment-cycle payload.
            initial_rows: Original dataset rows used to create cycle zero when the
                lineage has not yet been recorded.

        Returns:
            A copy of the newly recorded experiment cycle.
        """

        if request.dataset_id == request.parent_dataset_id:
            raise ValueError("The updated dataset_id must differ from parent_dataset_id.")
        if request.n_rows_after < request.n_rows_before:
            raise ValueError("n_rows_after must be greater than or equal to n_rows_before.")

        with self._lock:
            previous = self.list_for_dataset(request.parent_dataset_id)
            if not previous and initial_rows is not None:
                initial_request = request.model_copy(
                    update={
                        "dataset_id": request.parent_dataset_id,
                        "n_rows_after": request.n_rows_before,
                        "rows": initial_rows,
                    }
                )
                initial_cycle = {
                    "cycle_id": uuid4().hex,
                    "cycle_number": 0,
                    "created_at": datetime.now(UTC).isoformat(),
                    **initial_request.model_dump(mode="json"),
                    "parent_dataset_id": request.parent_dataset_id,
                    "append_mode": "initial",
                    "appended_rows": len(initial_rows),
                    "model": {"type": "initial_data", "n_train": len(initial_rows)},
                    "acquisition": {},
                    "optimizer": {},
                    "source_run_id": None,
                    "target_summary": _target_summaries(initial_request),
                }
                self._by_dataset[request.parent_dataset_id] = initial_cycle
                self._by_cycle_id[initial_cycle["cycle_id"]] = initial_cycle
                previous = [initial_cycle]
            cycle = {
                "cycle_id": uuid4().hex,
                "cycle_number": len(previous),
                "created_at": datetime.now(UTC).isoformat(),
                **request.model_dump(mode="json"),
                "appended_rows": len(request.rows),
                "target_summary": _target_summaries(request),
            }
            self._by_dataset[request.dataset_id] = cycle
            self._by_cycle_id[cycle["cycle_id"]] = cycle
            return deepcopy(cycle)

    def response_for_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Return cycle records and generated target-progress figures."""

        cycles = self.list_for_dataset(dataset_id)
        targets: list[str] = []
        for cycle in cycles:
            for target in cycle.get("target_columns", []):
                if target not in targets:
                    targets.append(target)
        return {
            "dataset_id": dataset_id,
            "count": len(cycles),
            "targets": targets,
            "cycles": cycles,
            "visualizations": _history_visualizations(cycles),
        }


__all__ = ["ExperimentCycleRequest", "ExperimentHistoryStore"]
