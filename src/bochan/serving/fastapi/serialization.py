"""Serialization helpers for FastAPI and web-serving boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any


def to_serializable(value: Any) -> Any:
    """Convert torch, numpy, dataclass, and container values to JSON-safe data."""
    if value is None:
        return None

    try:
        import torch

        if torch.is_tensor(value):
            detached = value.detach().cpu()
            if detached.ndim == 0:
                return to_serializable(detached.item())
            return to_serializable(detached.tolist())
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return to_serializable(value.tolist())
        if isinstance(value, np.generic):
            return to_serializable(value.item())
    except Exception:
        pass

    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def model_metadata(optimizer: Any) -> dict[str, Any]:
    """Return JSON-safe public metadata from a fitted optimizer bundle."""
    bundle = getattr(optimizer, "bundle", None)
    if bundle is None:
        return {}
    metadata = dict(getattr(bundle, "metadata", {}) or {})
    metadata.pop("sub_bundles", None)
    return to_serializable(metadata)


__all__ = ["model_metadata", "to_serializable"]
