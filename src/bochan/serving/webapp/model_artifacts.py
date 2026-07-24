"""Portable Web-workbench model artifacts.

The Core FastAPI already persists :class:`BayesianOptimizer` objects with
``torch.save``.  The React workbench uses :class:`TabularBayesianOptimizer`
and keeps additional data required by the Results visualizations, so it needs
its own versioned bundle around the same serialization mechanism.
"""

from __future__ import annotations

import copy
import io
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from bochan.tabular import TabularBayesianOptimizer

from .model_reuse import get_registered_model_signature, register_model_signature
from .visualization_sessions import (
    VisualizationSession,
    get_visualization_session,
    register_visualization_session,
    visualization_options,
)

_ARTIFACT_VERSION = 1
_OBJECT_TYPE = "bochan.web.TabularModelArtifact"
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _bochan_version() -> str:
    try:
        return version("bochan")
    except PackageNotFoundError:
        return "development"


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return stem or "bochan_model"


def _serializable_tabular_optimizer(session: VisualizationSession) -> TabularBayesianOptimizer:
    """Return a fitted optimizer copy without the request-local candidate wrapper."""

    optimizer = copy.copy(session.tabular_optimizer)
    optimizer.__dict__.pop("candidate", None)
    return optimizer


def serialize_web_model_artifact(run_id: str) -> tuple[bytes, str]:
    """Serialize one fitted Web run and its reproducible Results state."""

    import torch

    session = get_visualization_session(run_id)
    if not session.result:
        raise RuntimeError("The selected Web run has no finalized result to export.")

    result = copy.deepcopy(session.result)
    payload = {
        "artifact_version": _ARTIFACT_VERSION,
        "object_type": _OBJECT_TYPE,
        "bochan_version": _bochan_version(),
        "original_run_id": run_id,
        "model_signature": get_registered_model_signature(run_id),
        "tabular_optimizer": _serializable_tabular_optimizer(session),
        "data": session.data.copy(),
        "encoded_targets": session.encoded_targets.copy(),
        "feature_columns": list(session.feature_columns),
        "target_columns": list(session.target_columns),
        "target_metadata": copy.deepcopy(session.target_metadata),
        "hybrid_model": bool(session.hybrid_model),
        "feature_constraints": copy.deepcopy(session.feature_constraints),
        "candidate_result": session.candidate_result,
        "rows": copy.deepcopy(session.rows),
        "request_details": copy.deepcopy(session.request_details),
        "result": result,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    dataset_name = str(result.get("dataset_name") or "bochan_model")
    filename = f"{_safe_stem(dataset_name.rsplit('.', 1)[0])}.bochan.pt"
    return buffer.getvalue(), filename


def deserialize_web_model_artifact(
    content: bytes,
    *,
    trust_pickle: bool,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load and validate a trusted Web model artifact.

    ``torch.load`` uses pickle and can execute code while loading.  The caller
    must therefore provide an explicit trust confirmation.
    """

    if not trust_pickle:
        raise ValueError(
            "Model artifacts use torch.load / pickle. Confirm trust only for "
            "files created by this bochan Web application."
        )
    if not content:
        raise ValueError("The uploaded model artifact is empty.")
    if len(content) > _MAX_ARTIFACT_BYTES:
        raise ValueError("The uploaded model artifact exceeds the 512 MiB limit.")

    import torch

    buffer = io.BytesIO(content)
    try:
        payload = torch.load(buffer, map_location=map_location, weights_only=False)
    except TypeError:
        buffer.seek(0)
        payload = torch.load(buffer, map_location=map_location)

    if not isinstance(payload, dict):
        raise TypeError("The uploaded file is not a bochan Web model artifact.")
    if payload.get("object_type") != _OBJECT_TYPE:
        raise TypeError("The uploaded file has an unsupported model artifact type.")
    if int(payload.get("artifact_version", -1)) != _ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {payload.get('artifact_version')!r}."
        )

    optimizer = payload.get("tabular_optimizer")
    if not isinstance(optimizer, TabularBayesianOptimizer):
        raise TypeError("The artifact does not contain a TabularBayesianOptimizer.")
    if optimizer.dataset is None or optimizer.bo.bundle is None:
        raise ValueError("The artifact does not contain a fitted tabular model and dataset.")

    required = (
        "data",
        "encoded_targets",
        "feature_columns",
        "target_columns",
        "target_metadata",
        "result",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"The model artifact is incomplete: missing {missing!r}.")
    if not isinstance(payload["result"], dict):
        raise TypeError("The model artifact result payload is invalid.")
    return payload


def restore_web_model_artifact(
    payload: dict[str, Any],
    *,
    dataset_id: str,
    dataset_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Register an imported artifact as a new in-memory Web visualization run."""

    optimizer = payload["tabular_optimizer"]
    optimizer.__dict__.pop("candidate", None)
    run_id = uuid4().hex
    session = VisualizationSession(
        optimizer=optimizer.bo,
        tabular_optimizer=optimizer,
        data=payload["data"].copy(),
        encoded_targets=payload["encoded_targets"].copy(),
        feature_columns=list(payload["feature_columns"]),
        target_columns=list(payload["target_columns"]),
        target_metadata=copy.deepcopy(payload["target_metadata"]),
        hybrid_model=bool(payload.get("hybrid_model")),
        feature_constraints=copy.deepcopy(payload.get("feature_constraints") or []),
        candidate_result=payload.get("candidate_result"),
        rows=copy.deepcopy(payload.get("rows") or []),
        request_details=copy.deepcopy(payload.get("request_details") or {}),
    )
    result = copy.deepcopy(payload["result"])
    result["dataset_id"] = dataset_id
    result["dataset_name"] = dataset_name
    result["visualization_run_id"] = run_id
    result["visualization_options"] = visualization_options(session)
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "model_artifact_loaded": True,
            "model_artifact_version": _ARTIFACT_VERSION,
            "model_artifact_bochan_version": payload.get("bochan_version"),
            "model_artifact_original_run_id": payload.get("original_run_id"),
            "visualization_session": "imported_artifact",
        }
    )
    result["metadata"] = metadata
    session.result = copy.deepcopy(result)
    register_visualization_session(run_id, session)

    signature = payload.get("model_signature")
    if isinstance(signature, str) and signature:
        register_model_signature(run_id, signature)

    request_payload = copy.deepcopy(
        session.request_details.get("request_payload") or {}
    )
    if isinstance(request_payload, dict):
        request_payload["dataset_id"] = dataset_id
    else:
        request_payload = {}
    return run_id, result, request_payload


__all__ = [
    "deserialize_web_model_artifact",
    "restore_web_model_artifact",
    "serialize_web_model_artifact",
]
