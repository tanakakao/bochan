"""Application helpers for stateful BochanStudy FastAPI endpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from bochan.api import BochanStudy, Trial

from .converters import (
    to_acquisition_config,
    to_data_context,
    to_fit_config,
    to_model_config,
    to_optimize_config,
    to_serializable,
    to_tensor,
)
from .schemas.study import StudyCreateRequest, StudySummaryResponse


def _dump(value: Any) -> dict[str, Any]:
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else dict(value)


def acquisition_config(value: Any, options: Any) -> Any:
    """Convert only the canonical HTTP acquisition representation."""
    if value is None or isinstance(value, str):
        return value
    return to_acquisition_config(value, options)


def generation_schedule(value: Any, options: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = {"steps": value} if isinstance(value, list) else _dump(value)
    steps = []
    for raw_step in payload.get("steps", []):
        step = _dump(raw_step)
        acq = step.pop("acquisition_config", None)
        if acq is not None:
            step["acq_config"] = acquisition_config(acq, options)
        opt = step.pop("optimize_config", None)
        if opt is not None:
            step["opt_config"] = to_optimize_config(opt, options)
        if step.get("data_context") is not None:
            step["data_context"] = to_data_context(step["data_context"], options)
        steps.append(step)
    payload["steps"] = steps
    return payload


def build_study(
    request: StudyCreateRequest,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> BochanStudy:
    options = request.tensor_options
    study = BochanStudy(
        model_config=(
            to_model_config(request.bo_model_config, options)
            if request.bo_model_config is not None
            else None
        ),
        fit_config=(to_fit_config(request.fit_config) if request.fit_config is not None else None),
        acq_config=acquisition_config(request.acquisition_config, options),
        opt_config=(
            to_optimize_config(request.optimize_config, options)
            if request.optimize_config is not None
            else None
        ),
        data_context=(
            to_data_context(request.data_context, options)
            if request.data_context is not None
            else None
        ),
        bounds=to_tensor(request.bounds, options) if request.bounds is not None else None,
        n_initial_random=request.n_initial_random,
        early_stopping_config=request.early_stopping_config,
        generation_schedule=generation_schedule(request.generation_schedule, options),
        metadata=dict(metadata if metadata is not None else request.metadata),
    )
    if (request.initial_X is None) != (request.initial_Y is None):
        raise ValueError("initial_X and initial_Y must be provided together.")
    if request.initial_X is not None:
        study.add_observations(
            to_tensor(request.initial_X, options),
            to_tensor(request.initial_Y, options),
            metadata=request.initial_metadata,
        )
    return study


def restore_trials(study: BochanStudy, snapshot: Mapping[str, Any]) -> None:
    study.trials = [Trial.from_dict(item) for item in snapshot.get("trials", [])]
    study.next_trial_id = int(snapshot.get("next_trial_id", 0))
    if study.trials:
        study.next_trial_id = max(study.next_trial_id, max(t.trial_id for t in study.trials) + 1)
    state = (snapshot.get("metadata") or {}).get("early_stopping_state")
    if isinstance(state, Mapping):
        study._early_stopping_state.update(dict(state))


def summary(study_id: str, study: BochanStudy) -> StudySummaryResponse:
    config = study.model_config
    return StudySummaryResponse(
        study_id=study_id,
        task_type=str(getattr(config, "task_type", "regression")),
        model_type=str(getattr(config, "model_type", "base")),
        n_trials=len(study.trials),
        n_completed=study.n_completed,
        n_pending=study.n_pending,
        n_failed=sum(t.state.value == "FAILED" for t in study.trials),
        metadata=to_serializable(study.metadata),
        current_generation_step=to_serializable(study.current_generation_step()),
        stop_decision=to_serializable(study.stop_decision),
    )


def _direction(study: BochanStudy, output_index: int, direction: Any | None) -> str:
    if direction is not None:
        return str(direction)
    objective = getattr(getattr(study, "acq_config", None), "objective_config", None)
    directions = getattr(objective, "directions", None) if objective is not None else None
    if directions is not None:
        values = list(directions)
        index = output_index if output_index >= 0 else len(values) + output_index
        if 0 <= index < len(values):
            return str(values[index])
    configured = getattr(objective, "direction", None) if objective is not None else None
    if configured is not None:
        return str(configured)
    maximize = getattr(objective, "maximize", None) if objective is not None else None
    return "maximize" if maximize is None or bool(maximize) else "minimize"


def _values(value: Any) -> list[Any]:
    value = to_serializable(value)
    if isinstance(value, Mapping):
        return list(value.values())
    if not isinstance(value, list):
        return [value]
    out: list[Any] = []
    stack = list(reversed(value))
    while stack:
        item = stack.pop()
        stack.extend(reversed(item)) if isinstance(item, list) else out.append(item)
    return out


def _trial_value(trial: Trial, output_index: int) -> float | None:
    values = _values(trial.y)
    index = output_index if output_index >= 0 else len(values) + output_index
    if index < 0 or index >= len(values):
        raise IndexError(f"output_index={output_index} is out of range for {len(values)} outputs.")
    try:
        value = float(values[index])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def history_records(
    study: BochanStudy,
    *,
    output_index: int,
    direction: Any | None,
) -> tuple[str, list[dict[str, Any]]]:
    resolved = _direction(study, output_index, direction)
    records = []
    best: float | None = None
    for order, trial in enumerate(sorted(study.completed_trials(), key=lambda t: t.trial_id)):
        value = _trial_value(trial, output_index)
        if value is None:
            continue
        is_best = best is None or (resolved == "maximize" and value > best) or (resolved == "minimize" and value < best)
        if is_best:
            best = value
        records.append({"trial_id": trial.trial_id, "order": order, "cycle": trial.metadata.get("cycle", order), "value": value, "best_value": best, "is_best": is_best})
    return resolved, records


def pareto_records(
    study: BochanStudy,
    *,
    output_indices: list[int] | None,
    directions: list[str] | None,
) -> tuple[list[int], list[str], list[Trial], list[dict[str, Any]]]:
    completed = [trial for trial in study.completed_trials() if trial.y is not None]
    indices = list(output_indices) if output_indices is not None else list(range(max((len(_values(t.y)) for t in completed), default=0)))
    resolved = [_direction(study, index, None) for index in indices] if directions is None else list(directions)
    pareto = study.pareto_trials(output_indices=indices, directions=resolved)
    pareto_ids = {trial.trial_id for trial in pareto}
    records = []
    for trial in completed:
        values = [_trial_value(trial, index) for index in indices]
        if any(value is None for value in values):
            continue
        item = trial.to_dict()
        item["selected_values"] = values
        item["is_pareto"] = trial.trial_id in pareto_ids
        records.append(to_serializable(item))
    return indices, resolved, pareto, records


__all__ = [
    "acquisition_config",
    "build_study",
    "generation_schedule",
    "history_records",
    "pareto_records",
    "restore_trials",
    "summary",
]
