"""Candidate-level acquisition provenance helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_SNAPSHOT_ATTR = "_acquisition_diagnostics_snapshot"


def candidate_acquisition_diagnostics(result: Any) -> dict[str, Any] | None:
    """Return the acquisition diagnostics captured for one candidate result.

    The returned mapping is a deep copy so callers can safely enrich or serialize
    it without mutating optimizer history.
    """

    context = getattr(result, "data_context", None)
    if context is None:
        return None
    diagnostics = getattr(context, _SNAPSHOT_ATTR, None)
    if diagnostics is None:
        return None
    return deepcopy(dict(diagnostics))


__all__ = ["candidate_acquisition_diagnostics"]
