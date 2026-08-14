"""Public Web visualization sessions with explicit composition dispatch."""

from __future__ import annotations

from typing import Any

from . import _visualization_sessions_core as _core

VisualizationSession = _core.VisualizationSession
attach_fitted_tabular_optimizer = _core.attach_fitted_tabular_optimizer
begin_visualization_run = _core.begin_visualization_run
discard_visualization_run = _core.discard_visualization_run
finalize_visualization_run = _core.finalize_visualization_run
get_visualization_session = _core.get_visualization_session
model_details = _core.model_details
register_visualization_session = _core.register_visualization_session

# Keep shared in-memory state reachable for existing diagnostics/tests while the
# canonical storage remains owned by the core session module.
_LOCK = _core._LOCK
_SESSIONS = _core._SESSIONS
_PENDING = _core._PENDING
_MAX_SESSIONS = _core._MAX_SESSIONS


def _numeric_features(session: VisualizationSession) -> list[str]:
    """Expose the source-aware feature typing contract from the public owner."""

    return _core._numeric_features(session)


def visualization_options(session: VisualizationSession) -> dict[str, Any]:
    """Return normal options extended explicitly for composition sessions."""

    from .composition_visualization_dispatch import extend_visualization_options

    return extend_visualization_options(_core.visualization_options(session), session)


def build_visualization(run_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Build a result plot through explicit composition or generic dispatch."""

    from .composition_visualization_dispatch import build_composition_visualization

    session = get_visualization_session(run_id)
    composition_result = build_composition_visualization(session, request)
    if composition_result is not None:
        return composition_result
    return _core.build_visualization(run_id, request)


def __getattr__(name: str) -> Any:
    """Delegate private compatibility reads to the unchanged core implementation."""

    return getattr(_core, name)


__all__ = [
    "VisualizationSession",
    "attach_fitted_tabular_optimizer",
    "begin_visualization_run",
    "build_visualization",
    "discard_visualization_run",
    "finalize_visualization_run",
    "get_visualization_session",
    "model_details",
    "register_visualization_session",
    "visualization_options",
]
