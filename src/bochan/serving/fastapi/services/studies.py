"""Application helpers for stateful BochanStudy FastAPI endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bochan.api import BochanStudy

from ..converters import (
    to_acquisition_config,
    to_data_context,
    to_fit_config,
    to_model_config,
    to_optimize_config,
    to_serializable,
    to_tensor,
)
from ..schemas.study import StudyCreateRequest, StudySummaryResponse


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
        model_config=(to_model_config(request.bo_model_config, options) if request.bo_model_config is not None else None),
        fit_config=(to_fit_config(request.fit_config) if request.fit_config is not None else None),
        acq_config=acquisition_config(request.acquisition_config, options),
        opt_config=(to_optimize_config(request.optimize_config, options) if request.optimize_config is not None else None),
        data_context=(to_data_context(request.data_context, options) if request.data_context is not None else None),
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
    from bochan.api import Trial

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


__all__ = ["acquisition_config", "build_study", "generation_schedule", "restore_trials", "summary"]
