"""Request-scoped fitted-model reuse for the React Web workbench."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_WEB_REUSE_MODEL_KEY = "web_reuse_model_run_id"
_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "bochan_web_model_reuse_state",
    default=None,
)


def _plain(value: Any) -> Any:
    """Convert Pydantic and container values into stable JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=False))
    if hasattr(value, "dict"):
        return _plain(value.dict())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_reuse_signature(request: Any) -> str:
    """Return a fingerprint containing only settings that affect model fitting."""

    payload = {
        "dataset_id": getattr(request, "dataset_id", None),
        "feature_columns": list(getattr(request, "feature_columns", []) or []),
        "target_column": getattr(request, "target_column", None),
        "target_columns": list(getattr(request, "target_columns", []) or []),
        "direction": getattr(request, "direction", None),
        "directions": dict(getattr(request, "directions", {}) or {}),
        "model_type": getattr(request, "model_type", None),
        "model_kwargs": dict(getattr(request, "model_kwargs", {}) or {}),
        "fit_maxiter": getattr(request, "fit_maxiter", None),
        "normalize": getattr(request, "normalize", None),
        "outcome_transform": getattr(request, "outcome_transform", None),
        "input_perturbation": getattr(request, "input_perturbation", None),
        "n_w": getattr(request, "n_w", None),
        "perturbation_std": getattr(request, "perturbation_std", None),
        "search_space": list(getattr(request, "search_space", []) or []),
        "drop_missing": getattr(request, "drop_missing", None),
    }
    encoded = json.dumps(
        _plain(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_model_reuse_request(request: Any) -> tuple[Any, str | None]:
    """Remove the Web-only reuse identifier before normal model configuration."""

    model_kwargs = dict(getattr(request, "model_kwargs", {}) or {})
    raw_source = model_kwargs.pop(_WEB_REUSE_MODEL_KEY, None)
    source_run_id = str(raw_source).strip() if raw_source is not None else ""
    source_run_id = source_run_id or None

    if hasattr(request, "model_copy"):
        cleaned = request.model_copy(update={"model_kwargs": model_kwargs})
    else:
        values = dict(vars(request))
        values["model_kwargs"] = model_kwargs
        cleaned = type("WebRequest", (), values)()
    return cleaned, source_run_id


@contextmanager
def model_reuse_run(request: Any, source_run_id: str | None) -> Iterator[dict[str, Any]]:
    """Activate reuse metadata for one Web request."""

    state: dict[str, Any] = {
        "requested": source_run_id is not None,
        "source_run_id": source_run_id,
        "model_signature": model_reuse_signature(request),
        "model_reused": False,
        "fit_skipped": False,
    }
    token = _STATE.set(state)
    try:
        yield state
    finally:
        _STATE.reset(token)


def current_model_reuse_state() -> dict[str, Any] | None:
    """Return the mutable state active in the current request context."""

    return _STATE.get()


def mark_model_reused(source_run_id: str) -> None:
    """Record that candidate generation reused an existing fitted model."""

    state = _STATE.get()
    if state is None:
        return
    state["source_run_id"] = source_run_id
    state["model_reused"] = True
    state["fit_skipped"] = True


__all__ = [
    "current_model_reuse_state",
    "mark_model_reused",
    "model_reuse_run",
    "model_reuse_signature",
    "prepare_model_reuse_request",
]
