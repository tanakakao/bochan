"""Common ``.bochan.pt`` model artifact format for tensor and tabular APIs.

The artifact is a pickle-backed ``torch.save`` payload. Loading therefore always
requires explicit trust. The versioned envelope is shared by the tensor API,
tabular API, Web workbench, and project archives; surface-specific state is kept
under the optional ``state`` mapping instead of defining another model format.
"""

from __future__ import annotations

import io
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO, Literal

from bochan.api import BayesianOptimizer
from bochan.tabular import TabularBayesianOptimizer

MODEL_ARTIFACT_FORMAT = "bochan-model-artifact"
MODEL_ARTIFACT_VERSION = 1
MODEL_ARTIFACT_SUFFIX = ".bochan.pt"
ModelBackend = Literal["tensor", "tabular"]

_LEGACY_TENSOR_OBJECT_TYPE = "BayesianOptimizer"
_LEGACY_WEB_OBJECT_TYPE = "bochan.web.TabularModelArtifact"


def _bochan_version() -> str:
    try:
        return version("bochan")
    except PackageNotFoundError:
        return "development"


def infer_model_backend(optimizer: Any) -> ModelBackend:
    """Infer the public API backend from a supported optimizer object."""

    if isinstance(optimizer, TabularBayesianOptimizer):
        return "tabular"
    if isinstance(optimizer, BayesianOptimizer):
        return "tensor"
    raise TypeError(
        "Model artifacts support BayesianOptimizer or TabularBayesianOptimizer, "
        f"not {type(optimizer).__name__}."
    )


def _validate_optimizer(optimizer: Any, backend: ModelBackend) -> None:
    actual = infer_model_backend(optimizer)
    if actual != backend:
        raise TypeError(
            f"Artifact backend is {backend!r}, but the contained optimizer is {actual!r}."
        )
    if backend == "tensor" and optimizer.bundle is None:
        raise ValueError("The tensor optimizer is not fitted.")
    if backend == "tabular" and (
        optimizer.dataset is None or optimizer.bo.bundle is None
    ):
        raise ValueError("The tabular optimizer is not fitted.")


def build_model_artifact(
    optimizer: BayesianOptimizer | TabularBayesianOptimizer,
    *,
    backend: ModelBackend | None = None,
    metadata: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical versioned artifact envelope."""

    resolved_backend = backend or infer_model_backend(optimizer)
    _validate_optimizer(optimizer, resolved_backend)
    return {
        "format": MODEL_ARTIFACT_FORMAT,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "backend": resolved_backend,
        "bochan_version": _bochan_version(),
        "optimizer": optimizer,
        "metadata": dict(metadata or {}),
        "state": dict(state or {}),
    }


def serialize_model_artifact(
    optimizer: BayesianOptimizer | TabularBayesianOptimizer,
    *,
    backend: ModelBackend | None = None,
    metadata: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> bytes:
    """Serialize a supported optimizer using the common ``.bochan.pt`` envelope."""

    import torch

    payload = build_model_artifact(
        optimizer,
        backend=backend,
        metadata=metadata,
        state=state,
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def save_model_artifact(
    optimizer: BayesianOptimizer | TabularBayesianOptimizer,
    destination: str | Path | BinaryIO,
    *,
    backend: ModelBackend | None = None,
    metadata: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Write a common model artifact to a path or binary stream."""

    import torch

    torch.save(
        build_model_artifact(
            optimizer,
            backend=backend,
            metadata=metadata,
            state=state,
        ),
        destination,
    )


def _torch_load(source: bytes | bytearray | str | Path | BinaryIO, map_location: str | None) -> Any:
    import torch

    value: Any = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else source
    try:
        return torch.load(value, map_location=map_location, weights_only=False)
    except TypeError:
        if hasattr(value, "seek"):
            value.seek(0)
        return torch.load(value, map_location=map_location)


def _normalize_loaded_payload(payload: Any) -> dict[str, Any]:
    """Normalize canonical and legacy tensor/Web payloads to one envelope."""

    if isinstance(payload, (BayesianOptimizer, TabularBayesianOptimizer)):
        return build_model_artifact(
            payload,
            metadata={"legacy_artifact": True, "legacy_object_type": type(payload).__name__},
        )
    if not isinstance(payload, dict):
        raise TypeError("The uploaded file is not a supported bochan model artifact.")

    if payload.get("format") == MODEL_ARTIFACT_FORMAT:
        version_value = int(payload.get("artifact_version", -1))
        if version_value != MODEL_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported model artifact version: {version_value}.")
        backend = payload.get("backend")
        if backend not in {"tensor", "tabular"}:
            raise ValueError(f"Unsupported model artifact backend: {backend!r}.")
        optimizer = payload.get("optimizer")
        _validate_optimizer(optimizer, backend)
        metadata = payload.get("metadata")
        state = payload.get("state")
        if not isinstance(metadata, dict) or not isinstance(state, dict):
            raise TypeError("Model artifact metadata and state must be JSON-like mappings.")
        return {
            **payload,
            "metadata": dict(metadata),
            "state": dict(state),
        }

    if payload.get("object_type") == _LEGACY_TENSOR_OBJECT_TYPE:
        optimizer = payload.get("optimizer")
        _validate_optimizer(optimizer, "tensor")
        return {
            "format": MODEL_ARTIFACT_FORMAT,
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "backend": "tensor",
            "bochan_version": payload.get("bochan_version"),
            "optimizer": optimizer,
            "metadata": {
                "legacy_artifact": True,
                "legacy_object_type": _LEGACY_TENSOR_OBJECT_TYPE,
                "legacy_version": payload.get("version"),
            },
            "state": {},
        }

    if payload.get("object_type") == _LEGACY_WEB_OBJECT_TYPE:
        optimizer = payload.get("tabular_optimizer")
        _validate_optimizer(optimizer, "tabular")
        state = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "artifact_version",
                "object_type",
                "bochan_version",
                "original_run_id",
                "model_signature",
                "tabular_optimizer",
            }
        }
        return {
            "format": MODEL_ARTIFACT_FORMAT,
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "backend": "tabular",
            "bochan_version": payload.get("bochan_version"),
            "optimizer": optimizer,
            "metadata": {
                "legacy_artifact": True,
                "legacy_object_type": _LEGACY_WEB_OBJECT_TYPE,
                "original_run_id": payload.get("original_run_id"),
                "model_signature": payload.get("model_signature"),
            },
            "state": state,
        }

    raise TypeError("The uploaded file has an unsupported bochan model artifact type.")


def deserialize_model_artifact(
    source: bytes | bytearray | str | Path | BinaryIO,
    *,
    trust_pickle: bool,
    map_location: str | None = "cpu",
    expected_backend: ModelBackend | None = None,
) -> dict[str, Any]:
    """Load and normalize a trusted common or legacy model artifact."""

    if not trust_pickle:
        raise ValueError(
            "Model artifacts use torch.load / pickle. Set trust_pickle=true only "
            "for .bochan.pt files created by a trusted bochan process."
        )
    payload = _normalize_loaded_payload(_torch_load(source, map_location))
    backend = payload["backend"]
    if expected_backend is not None and backend != expected_backend:
        raise TypeError(
            f"This endpoint expects a {expected_backend} model artifact, but the file "
            f"contains a {backend} optimizer."
        )
    return payload


__all__ = [
    "MODEL_ARTIFACT_FORMAT",
    "MODEL_ARTIFACT_SUFFIX",
    "MODEL_ARTIFACT_VERSION",
    "ModelBackend",
    "build_model_artifact",
    "deserialize_model_artifact",
    "infer_model_backend",
    "save_model_artifact",
    "serialize_model_artifact",
]
