"""Conversion and response helpers for BochanStudy serving."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bochan.api import BochanStudy, Trial
from bochan.api.study_results import _resolve_direction, _row_values, _trial_value

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


def acquisition_config(value: Any, options: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    return to_acquisition_config(value, options)


def generation_schedule(value: Any, options: Any) -> Any:
    if value is None:
        return None
    payload = {"steps": value} if isinstance(value, list) else dict(value)
    steps: list[dict[str, Any]] = []
    for raw_step in payload.get("steps", []):
        step = dict(raw_step)
        if "acquisition_config" in step and "acq_config" not in step:
            step["acq_config"] = step.pop("acquisition_config")
        if "optimize_config" in step and "opt_config" not in step:
            step["opt_config"] = step.pop("optimize_config")
        if step.get("acq_config") is not None:
            step["acq_config"] = acquisition_config(step["acq_config"], options)
        if step.get("opt_config") is not None:
            step["opt_config"] = to_optimize_config(step["opt_config"], options)
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
        fit_config=(
            to_fit_config(request.fit_config)
            if request.fit_config is not None
            else None
        ),
        acq_config=acquisition_config(request.acq_config, options),
        opt_config=(
            to_optimize_config(request.opt_config, options)
            if request.opt_config is not None
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
        study.next_trial_id = max(
            study.next_trial_id,
            max(trial.trial_id for trial in study.trials) + 1,
        )
    state = (snapshot.get("metadata") or {}).get("early_stopping_state")
    if isinstance(state, Mapping):
        study._early_stopping_state.update(dict(state))


def summary(study_id: str, study: BochanStudy) -> StudySummaryResponse:
    model_config = study.model_config
    return StudySummaryResponse(
        study_id=study_id,
        task_type=str(getattr(model_config, "task_type", "regression")),
        model_type=str(getattr(model_config, "model_type", "base")),
        n_trials=len(study.trials),
        n_completed=study.n_completed,
        n_pending=study.n_pending,
        n_failed=sum(trial.state.value == "FAILED" for trial in study.trials),
        metadata=to_serializable(study.metadata),
        current_generation_step=to_serializable(study.current_generation_step()),
        stop_decision=to_serializable(study.stop_decision),
    )


def history_records(
    study: BochanStudy,
    *,
    output_index: int,
    direction: Any | None,
) -> tuple[str, list[dict[str, Any]]]:
    resolved = _resolve_direction(study, output_index, direction)
    records: list[dict[str, Any]] = []
    best_value: float | None = None
    for order, trial in enumerate(sorted(study.completed_trials(), key=lambda item: item.trial_id)):
        value = _trial_value(trial, output_index)
        if value is None:
            continue
        is_best = (
            best_value is None
            or (resolved == "maximize" and value > best_value)
            or (resolved == "minimize" and value < best_value)
        )
        if is_best:
            best_value = value
        records.append(
            {
                "trial_id": trial.trial_id,
                "order": order,
                "cycle": trial.metadata.get("cycle", order),
                "value": value,
                "best_value": best_value,
                "is_best": is_best,
            }
        )
    return resolved, records


def pareto_records(
    study: BochanStudy,
    *,
    output_indices: list[int] | None,
    directions: list[str] | None,
) -> tuple[list[int], list[str], list[Trial], list[dict[str, Any]]]:
    completed = [trial for trial in study.completed_trials() if trial.y is not None]
    if output_indices is None:
        output_count = max((len(_row_values(trial.y)) for trial in completed), default=0)
        indices = list(range(output_count))
    else:
        indices = list(output_indices)
    resolved_directions = (
        [_resolve_direction(study, index, None) for index in indices]
        if directions is None
        else list(directions)
    )
    pareto = study.pareto_trials(
        output_indices=indices,
        directions=resolved_directions,
    )
    pareto_ids = {trial.trial_id for trial in pareto}
    records: list[dict[str, Any]] = []
    for trial in completed:
        values = [_trial_value(trial, index) for index in indices]
        if any(value is None for value in values):
            continue
        item = trial.to_dict()
        item["selected_values"] = values
        item["is_pareto"] = trial.trial_id in pareto_ids
        records.append(to_serializable(item))
    return indices, resolved_directions, pareto, records


__all__ = [
    "acquisition_config",
    "build_study",
    "history_records",
    "pareto_records",
    "restore_trials",
    "summary",
]
