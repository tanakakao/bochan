"""Optuna-like result helpers for :class:`bochan.api.BochanStudy`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .study import Trial, TrialState

Direction = Literal["maximize", "minimize"]


def _normalize_direction(value: Any) -> Direction:
    """Normalize user-facing direction aliases."""
    if isinstance(value, bool):
        return "maximize" if value else "minimize"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "maximize" if float(value) >= 0.0 else "minimize"
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"maximize", "max", "greater", "higher", "high"}:
        return "maximize"
    if normalized in {"minimize", "min", "less", "lower", "low"}:
        return "minimize"
    raise ValueError(
        "direction must be 'maximize' or 'minimize'. "
        f"Got {value!r}."
    )


def _row_values(value: Any) -> list[Any]:
    """Flatten one stored trial value or candidate row to a Python list."""
    if isinstance(value, Mapping):
        return list(value.values())

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "reshape") and hasattr(current, "tolist"):
        try:
            flattened = current.reshape(-1).tolist()
            return flattened if isinstance(flattened, list) else [flattened]
        except Exception:
            pass
    if hasattr(current, "tolist"):
        try:
            converted = current.tolist()
            if isinstance(converted, list):
                result: list[Any] = []
                stack = list(reversed(converted))
                while stack:
                    item = stack.pop()
                    if isinstance(item, list):
                        stack.extend(reversed(item))
                    else:
                        result.append(item)
                return result
            return [converted]
        except Exception:
            pass
    if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        result = []
        for item in current:
            result.extend(_row_values(item))
        return result
    return [current]


def _finite_value(value: Any) -> float | None:
    """Return a finite scalar or ``None``."""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _trial_value(trial: Trial, output_index: int) -> float | None:
    """Extract one finite objective value from a completed trial."""
    if trial.state != TrialState.COMPLETED or trial.y is None:
        return None
    values = _row_values(trial.y)
    index = int(output_index)
    if index < 0:
        index += len(values)
    if index < 0 or index >= len(values):
        raise IndexError(
            f"output_index={output_index} is out of range for {len(values)} outputs."
        )
    return _finite_value(values[index])


def _objective_config(study: Any) -> Any | None:
    config = getattr(study, "acq_config", None)
    return getattr(config, "objective_config", None)


def _resolve_direction(
    study: Any,
    output_index: int = 0,
    direction: Any | None = None,
) -> Direction:
    """Resolve an explicit or configured optimization direction."""
    if direction is not None:
        return _normalize_direction(direction)

    config = _objective_config(study)
    if config is None:
        return "maximize"

    directions = getattr(config, "directions", None)
    if directions is not None:
        values = list(directions)
        index = int(output_index)
        if index < 0:
            index += len(values)
        if 0 <= index < len(values):
            return _normalize_direction(values[index])

    configured = getattr(config, "direction", None)
    if configured is not None:
        return _normalize_direction(configured)

    maximize = getattr(config, "maximize", None)
    if maximize is not None:
        return _normalize_direction(bool(maximize))
    return "maximize"


def _output_count(study: Any) -> int:
    """Infer the number of stored outputs from the first valid completed trial."""
    for trial in study.completed_trials():
        if trial.y is not None:
            return len(_row_values(trial.y))
    return 0


def _require_single_objective(study: Any) -> None:
    """Reject ambiguous Optuna-like properties for multi-output studies."""
    task_type = str(getattr(getattr(study, "model_config", None), "task_type", ""))
    output_count = _output_count(study)
    if task_type == "multi_objective" or output_count > 1:
        raise RuntimeError(
            "A multi-objective study has no single best trial. Use "
            "get_best_trial(output_index=..., direction=...) for one output or "
            "pareto_trials(...) for the Pareto set."
        )


def best_trials(
    self: Any,
    *,
    top_k: int = 1,
    output_index: int = 0,
    direction: Any | None = None,
) -> list[Trial]:
    """Return the top finite completed trials for one output."""
    limit = int(top_k)
    if limit < 0:
        raise ValueError("top_k must be non-negative.")
    resolved_direction = _resolve_direction(self, output_index, direction)
    scored: list[tuple[float, Trial]] = []
    for trial in self.completed_trials():
        value = _trial_value(trial, output_index)
        if value is not None:
            scored.append((value, trial))
    scored.sort(key=lambda item: item[0], reverse=resolved_direction == "maximize")
    return [trial for _, trial in scored[:limit]]


def get_best_trial(
    self: Any,
    *,
    output_index: int = 0,
    direction: Any | None = None,
) -> Trial:
    """Return the best finite completed trial for one output."""
    trials = best_trials(
        self,
        top_k=1,
        output_index=output_index,
        direction=direction,
    )
    if not trials:
        raise ValueError("No finite completed trial is available.")
    return trials[0]


def get_best_value(
    self: Any,
    *,
    output_index: int = 0,
    direction: Any | None = None,
) -> float:
    """Return the scalar objective value of the best trial."""
    trial = get_best_trial(
        self,
        output_index=output_index,
        direction=direction,
    )
    value = _trial_value(trial, output_index)
    assert value is not None
    return value


def get_best_x(
    self: Any,
    *,
    output_index: int = 0,
    direction: Any | None = None,
) -> Any:
    """Return the candidate row of the best trial."""
    return get_best_trial(
        self,
        output_index=output_index,
        direction=direction,
    ).x


def _parameter_names(study: Any, width: int, param_names: Sequence[str] | None) -> list[str]:
    if param_names is not None:
        names = [str(name) for name in param_names]
    else:
        metadata = getattr(study, "metadata", {}) or {}
        configured = metadata.get("param_names") or metadata.get("feature_names")
        names = [str(name) for name in configured] if configured is not None else []
    if names and len(names) != width:
        raise ValueError(
            f"param_names must contain {width} names. Got {len(names)}."
        )
    return names or [f"x{index}" for index in range(width)]


def get_best_params(
    self: Any,
    *,
    param_names: Sequence[str] | None = None,
    output_index: int = 0,
    direction: Any | None = None,
) -> dict[str, Any]:
    """Return the best candidate as an Optuna-like parameter mapping."""
    x = get_best_x(
        self,
        output_index=output_index,
        direction=direction,
    )
    if isinstance(x, Mapping):
        return dict(x)
    values = _row_values(x)
    names = _parameter_names(self, len(values), param_names)
    return dict(zip(names, values, strict=True))


def best_result(
    self: Any,
    *,
    param_names: Sequence[str] | None = None,
    output_index: int = 0,
    direction: Any | None = None,
) -> dict[str, Any]:
    """Return a serializable summary of the best trial."""
    trial = get_best_trial(
        self,
        output_index=output_index,
        direction=direction,
    )
    resolved_direction = _resolve_direction(self, output_index, direction)
    return {
        "trial_id": int(trial.trial_id),
        "value": get_best_value(
            self,
            output_index=output_index,
            direction=resolved_direction,
        ),
        "values": _row_values(trial.y),
        "x": trial.x,
        "params": get_best_params(
            self,
            param_names=param_names,
            output_index=output_index,
            direction=resolved_direction,
        ),
        "output_index": int(output_index),
        "direction": resolved_direction,
        "metadata": dict(trial.metadata),
    }


def pareto_trials(
    self: Any,
    *,
    output_indices: Sequence[int] | None = None,
    directions: Sequence[Any] | None = None,
) -> list[Trial]:
    """Return finite non-dominated completed trials."""
    completed = [trial for trial in self.completed_trials() if trial.y is not None]
    if not completed:
        return []

    output_count = max(len(_row_values(trial.y)) for trial in completed)
    indices = list(range(output_count)) if output_indices is None else [int(i) for i in output_indices]
    if not indices:
        raise ValueError("output_indices must contain at least one output.")

    if directions is None:
        resolved_directions = [_resolve_direction(self, index, None) for index in indices]
    else:
        if len(directions) != len(indices):
            raise ValueError(
                "directions must have the same length as output_indices. "
                f"Got {len(directions)} and {len(indices)}."
            )
        resolved_directions = [_normalize_direction(value) for value in directions]

    scored: list[tuple[Trial, tuple[float, ...]]] = []
    for trial in completed:
        values: list[float] = []
        valid = True
        for index, direction in zip(indices, resolved_directions, strict=True):
            value = _trial_value(trial, index)
            if value is None:
                valid = False
                break
            values.append(value if direction == "maximize" else -value)
        if valid:
            scored.append((trial, tuple(values)))

    pareto: list[Trial] = []
    for candidate_index, (candidate_trial, candidate_values) in enumerate(scored):
        dominated = False
        for other_index, (_, other_values) in enumerate(scored):
            if candidate_index == other_index:
                continue
            no_worse = all(
                other >= candidate
                for other, candidate in zip(other_values, candidate_values, strict=True)
            )
            strictly_better = any(
                other > candidate
                for other, candidate in zip(other_values, candidate_values, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate_trial)
    return pareto


def _best_trial_property(self: Any) -> Trial:
    _require_single_objective(self)
    return get_best_trial(self)


def _best_value_property(self: Any) -> float:
    _require_single_objective(self)
    return get_best_value(self)


def _best_x_property(self: Any) -> Any:
    _require_single_objective(self)
    return get_best_x(self)


def _best_params_property(self: Any) -> dict[str, Any]:
    _require_single_objective(self)
    return get_best_params(self)


def install_study_result_api(cls: type) -> type:
    """Install result helpers on a Study class once."""
    if getattr(cls, "_bochan_result_api_installed", False):
        return cls
    cls.best_trials = best_trials
    cls.get_best_trial = get_best_trial
    cls.get_best_value = get_best_value
    cls.get_best_x = get_best_x
    cls.get_best_params = get_best_params
    cls.best_result = best_result
    cls.pareto_trials = pareto_trials
    cls.best_trial = property(_best_trial_property)
    cls.best_value = property(_best_value_property)
    cls.best_x = property(_best_x_property)
    cls.best_params = property(_best_params_property)
    cls._bochan_result_api_installed = True
    return cls


__all__ = [
    "best_result",
    "best_trials",
    "get_best_params",
    "get_best_trial",
    "get_best_value",
    "get_best_x",
    "install_study_result_api",
    "pareto_trials",
]
