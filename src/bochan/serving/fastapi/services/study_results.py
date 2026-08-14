"""Study result projection helpers for FastAPI responses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from bochan.api import BochanStudy, Trial

from ..converters import to_serializable


def _normalize_direction(value: Any) -> str:
    if isinstance(value, bool):
        return "maximize" if value else "minimize"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "maximize" if float(value) >= 0.0 else "minimize"
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"maximize", "max", "greater", "higher", "high"}:
        return "maximize"
    if normalized in {"minimize", "min", "less", "lower", "low"}:
        return "minimize"
    raise ValueError(f"Unknown optimization direction: {value!r}.")


def _direction(study: BochanStudy, output_index: int, direction: Any | None) -> str:
    if direction is not None:
        return _normalize_direction(direction)
    objective = getattr(getattr(study, "acq_config", None), "objective_config", None)
    directions = getattr(objective, "directions", None) if objective is not None else None
    if directions is not None:
        values = list(directions)
        index = output_index if output_index >= 0 else len(values) + output_index
        if 0 <= index < len(values):
            return _normalize_direction(values[index])
    configured = getattr(objective, "direction", None) if objective is not None else None
    if configured is not None:
        return _normalize_direction(configured)
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
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
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
