"""Dataset adapter for fitted-model reuse in the Web workflow."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .visualization_sessions import get_visualization_session


class ModelReuseDatasetStore:
    """Expose one retained model session as a workflow dataset store."""

    def __init__(self, record: Any) -> None:
        self._record = record

    def get(self, dataset_id: str) -> Any:
        """Return the retained record without reading the transient store."""

        del dataset_id
        return self._record


def store_for_model_reuse(
    store: Any,
    request: Any,
    source_run_id: str | None,
) -> Any:
    """Use source-session data when candidate generation reuses a fitted model."""

    if not source_run_id:
        return store

    session = get_visualization_session(source_run_id)
    data = session.data.copy()
    source_result = dict(getattr(session, "result", {}) or {})
    record = SimpleNamespace(
        dataset_id=str(
            getattr(request, "dataset_id", None)
            or source_result.get("dataset_id")
            or ""
        ),
        name=str(source_result.get("dataset_name") or "reused_model"),
        data=data,
        source_type="model_reuse",
        profile={
            "n_rows": int(len(data)),
            "n_columns": int(len(data.columns)),
        },
        metadata={"source_run_id": source_run_id},
    )
    return ModelReuseDatasetStore(record)


__all__ = ["ModelReuseDatasetStore", "store_for_model_reuse"]
