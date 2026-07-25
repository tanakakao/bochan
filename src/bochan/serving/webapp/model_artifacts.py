"""Web-workbench state stored inside the common ``.bochan.pt`` model artifact."""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from bochan.model_artifact import (
    MODEL_ARTIFACT_VERSION,
    deserialize_model_artifact,
    serialize_model_artifact,
)
from bochan.tabular import TabularBayesianOptimizer

from .model_reuse import (
    get_registered_model_signature,
    model_reuse_signature,
    register_model_signature,
)
from .visualization_sessions import (
    VisualizationSession,
    get_visualization_session,
    register_visualization_session,
    visualization_options,
)

_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _safe_stem(value: str) -> str:
    """Return a filename stem that preserves Unicode while removing unsafe characters."""

    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", value).strip(" ._")
    return stem or "bochan_model"


def _artifact_filename(dataset_name: str, requested_filename: str | None) -> str:
    """Resolve a browser- and filesystem-safe ``.bochan.pt`` download name."""

    if requested_filename and requested_filename.strip():
        stem = requested_filename.strip()
        if stem.lower().endswith(".bochan.pt"):
            stem = stem[: -len(".bochan.pt")]
        elif stem.lower().endswith(".pt"):
            stem = stem[:-3]
    else:
        stem = dataset_name.rsplit(".", 1)[0]
    return f"{_safe_stem(stem)}.bochan.pt"


def _serializable_tabular_optimizer(session: VisualizationSession) -> TabularBayesianOptimizer:
    """Return a fitted optimizer copy without the request-local candidate wrapper."""

    optimizer = copy.copy(session.tabular_optimizer)
    optimizer.__dict__.pop("candidate", None)
    return optimizer


def serialize_web_model_artifact(
    run_id: str,
    *,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Serialize one fitted Web run using the common tensor/tabular envelope."""

    session = get_visualization_session(run_id)
    if not session.result:
        raise RuntimeError("The selected Web run has no finalized result to export.")

    result = copy.deepcopy(session.result)
    state = {
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
    content = serialize_model_artifact(
        _serializable_tabular_optimizer(session),
        backend="tabular",
        metadata={
            "surface": "web",
            "original_run_id": run_id,
            "model_signature": get_registered_model_signature(run_id),
        },
        state=state,
    )
    dataset_name = str(result.get("dataset_name") or "bochan_model")
    return content, _artifact_filename(dataset_name, filename)


def _task_type_for_web(value: Any) -> str:
    task = str(value or "regression").lower()
    if task in {"binary", "multiclass", "classification"}:
        return "classification"
    if task == "ordinal":
        return "ordinal"
    return "regression"


def _fallback_web_state(
    optimizer: TabularBayesianOptimizer,
    artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build minimal Web state for a common tabular artifact saved outside the UI."""

    import pandas as pd

    if optimizer.dataset is None or optimizer.bo.bundle is None:
        raise ValueError("The artifact does not contain a fitted tabular model and dataset.")
    try:
        feature_data, target_data = optimizer.visualization_training_dataframe()
        data = pd.concat(
            [feature_data.reset_index(drop=True), target_data.reset_index(drop=True)],
            axis=1,
        )
    except Exception as exc:
        raise ValueError(
            "This tabular artifact has no Web state and its training table could not be reconstructed."
        ) from exc

    feature_columns = [str(value) for value in optimizer.dataset.feature_names]
    target_columns = [str(value) for value in optimizer.dataset.target_names]
    task_type = _task_type_for_web(optimizer.bo.bundle.task_type)
    target_metadata = {
        target: {
            "target": target,
            "task_type": task_type,
            "optimize": True,
            "direction": "maximize",
            "goal": "none",
            "internal_task": str(optimizer.bo.bundle.task_type),
        }
        for target in target_columns
    }
    numeric_best: dict[str, float] = {}
    for target in target_columns:
        series = pd.to_numeric(data[target], errors="coerce")
        numeric_best[target] = float(series.max()) if series.notna().any() else 0.0

    result = {
        "dataset_id": "",
        "dataset_name": "imported_tabular_model",
        "task_type": task_type,
        "model_type": str(optimizer.bo.bundle.model_type),
        "n_train": int(len(data)),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_column": target_columns[0] if target_columns else "",
        "target_settings": list(target_metadata.values()),
        "target_metadata": target_metadata,
        "directions": {target: "maximize" for target in target_columns},
        "direction": "maximize",
        "best_observed": numeric_best,
        "candidates": [],
        "visualizations": [],
        "visualization_warnings": [
            "The model was saved outside the Web workbench, so candidate and plot state was not included."
        ],
        "metadata": {
            "common_model_artifact": True,
            "web_state_reconstructed": True,
            "artifact_metadata": copy.deepcopy(artifact_metadata),
        },
    }
    return {
        "data": data,
        "encoded_targets": data[target_columns].copy(),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_metadata": target_metadata,
        "hybrid_model": False,
        "feature_constraints": [],
        "candidate_result": None,
        "rows": [],
        "request_details": {},
        "result": result,
    }


def deserialize_web_model_artifact(
    content: bytes,
    *,
    trust_pickle: bool,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a trusted common tabular artifact and normalize its optional Web state."""

    if not content:
        raise ValueError("The uploaded model artifact is empty.")
    if len(content) > _MAX_ARTIFACT_BYTES:
        raise ValueError("The uploaded model artifact exceeds the 512 MiB limit.")

    artifact = deserialize_model_artifact(
        content,
        trust_pickle=trust_pickle,
        map_location=map_location,
        expected_backend="tabular",
    )
    optimizer = artifact["optimizer"]
    if not isinstance(optimizer, TabularBayesianOptimizer):
        raise TypeError("The artifact does not contain a TabularBayesianOptimizer.")

    metadata = dict(artifact.get("metadata") or {})
    state = dict(artifact.get("state") or {})
    required = {
        "data",
        "encoded_targets",
        "feature_columns",
        "target_columns",
        "target_metadata",
        "result",
    }
    if not required.issubset(state):
        state = _fallback_web_state(optimizer, metadata)
    if not isinstance(state.get("result"), dict):
        raise TypeError("The model artifact result payload is invalid.")

    return {
        "artifact_version": artifact.get("artifact_version", MODEL_ARTIFACT_VERSION),
        "bochan_version": artifact.get("bochan_version"),
        "original_run_id": metadata.get("original_run_id"),
        "model_signature": metadata.get("model_signature"),
        "artifact_metadata": metadata,
        "tabular_optimizer": optimizer,
        **state,
    }


def restore_web_model_artifact(
    payload: dict[str, Any],
    *,
    dataset_id: str,
    dataset_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Register an imported common tabular artifact as a Web visualization run."""

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
            "model_artifact_version": payload.get("artifact_version"),
            "model_artifact_bochan_version": payload.get("bochan_version"),
            "model_artifact_original_run_id": payload.get("original_run_id"),
            "visualization_session": "imported_artifact",
        }
    )
    result["metadata"] = metadata
    session.result = copy.deepcopy(result)
    register_visualization_session(run_id, session)

    request_payload = copy.deepcopy(session.request_details.get("request_payload") or {})
    if isinstance(request_payload, dict):
        request_payload["dataset_id"] = dataset_id
    else:
        request_payload = {}

    if request_payload:
        try:
            signature = model_reuse_signature(SimpleNamespace(**request_payload))
        except (AttributeError, TypeError, ValueError):
            signature = payload.get("model_signature")
    else:
        signature = payload.get("model_signature")
    if isinstance(signature, str) and signature:
        register_model_signature(run_id, signature)

    return run_id, result, request_payload


__all__ = [
    "deserialize_web_model_artifact",
    "restore_web_model_artifact",
    "serialize_web_model_artifact",
]
