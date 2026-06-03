"""FastAPI app with session persistence support.

This module extends ``bochan.api.fastapi`` with file-based save / load endpoints
for fitted ``BayesianOptimizer`` sessions.

Run:
    uvicorn bochan.api.fastapi_persistent:app --reload

Saved files are stored under ``BOCHAN_API_MODEL_DIR`` if the environment variable
is set. Otherwise ``bochan_sessions/`` is used.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .engine import BayesianOptimizer
from .fastapi import (
    APIBaseModel,
    SessionInfo,
    SessionStore,
    _get_session,
    _make_session_info,
    _to_python,
    create_router,
)


class SaveSessionRequest(APIBaseModel):
    """Request body for saving a fitted session."""

    filename: str | None = None
    overwrite: bool = False


class LoadSessionRequest(APIBaseModel):
    """Request body for loading a fitted session.

    ``torch.load`` uses pickle internally, so loading is intentionally gated by
    ``trust_pickle=True``. Only load files you created or otherwise trust.
    """

    filename: str
    map_location: str | None = "cpu"
    trust_pickle: bool = False


class SaveSessionResponse(APIBaseModel):
    session_id: str
    filename: str
    path: str
    metadata: dict[str, Any]


class LoadSessionResponse(SessionInfo):
    filename: str
    path: str


class SavedModelsResponse(APIBaseModel):
    root_dir: str
    filenames: list[str]


PERSISTENT_SESSION_STORE = SessionStore()


def _model_root_dir() -> Path:
    root = Path(os.environ.get("BOCHAN_API_MODEL_DIR", "bochan_sessions")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_model_path(filename: str | None, *, default_stem: str | None = None) -> Path:
    if filename is None or str(filename).strip() == "":
        filename = f"{default_stem or uuid.uuid4().hex}.pt"

    rel_path = Path(str(filename))
    if rel_path.is_absolute():
        raise ValueError("filename must be relative to BOCHAN_API_MODEL_DIR.")
    if any(part == ".." for part in rel_path.parts):
        raise ValueError("filename must not contain '..'.")
    if rel_path.suffix == "":
        rel_path = rel_path.with_suffix(".pt")

    root = _model_root_dir()
    full_path = (root / rel_path).resolve()
    try:
        full_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("filename escapes BOCHAN_API_MODEL_DIR.") from exc
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def _relative_model_filename(path: Path) -> str:
    root = _model_root_dir()
    return str(path.resolve().relative_to(root))


def _list_saved_models() -> list[str]:
    root = _model_root_dir()
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.pt") if path.is_file())


def _save_optimizer(optimizer: BayesianOptimizer, path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {_relative_model_filename(path)}")

    import torch

    payload = {
        "version": 1,
        "object_type": "BayesianOptimizer",
        "optimizer": optimizer,
    }
    torch.save(payload, path)


def _load_optimizer(path: Path, *, map_location: str | None = "cpu", trust_pickle: bool = False) -> BayesianOptimizer:
    if not trust_pickle:
        raise ValueError(
            "Loading uses torch.load / pickle. Set trust_pickle=true only for files you trust."
        )
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {_relative_model_filename(path)}")

    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch versions do not support weights_only
        payload = torch.load(path, map_location=map_location)

    optimizer = payload.get("optimizer") if isinstance(payload, dict) else payload
    if not isinstance(optimizer, BayesianOptimizer):
        raise TypeError("Loaded object is not a BayesianOptimizer session.")
    return optimizer


router = create_router(PERSISTENT_SESSION_STORE)


@router.get("/models", response_model=SavedModelsResponse)
def list_saved_models() -> SavedModelsResponse:
    """List saved BayesianOptimizer session files."""
    root = _model_root_dir()
    return SavedModelsResponse(root_dir=str(root), filenames=_list_saved_models())


@router.post("/sessions/{session_id}/save", response_model=SaveSessionResponse)
def save_session(session_id: str, request: SaveSessionRequest) -> SaveSessionResponse:
    """Save an in-memory BayesianOptimizer session to disk."""
    optimizer = _get_session(PERSISTENT_SESSION_STORE, session_id)
    try:
        path = _safe_model_path(request.filename, default_stem=session_id)
        _save_optimizer(optimizer, path, overwrite=request.overwrite)
        info = _make_session_info(session_id, optimizer)
        return SaveSessionResponse(
            session_id=session_id,
            filename=_relative_model_filename(path),
            path=str(path),
            metadata=_to_python(info.metadata),
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sessions/load", response_model=LoadSessionResponse)
def load_session(request: LoadSessionRequest) -> LoadSessionResponse:
    """Load a saved BayesianOptimizer session and register it as a new session."""
    try:
        path = _safe_model_path(request.filename)
        optimizer = _load_optimizer(
            path,
            map_location=request.map_location,
            trust_pickle=request.trust_pickle,
        )
        session_id = PERSISTENT_SESSION_STORE.create(optimizer)
        info = _make_session_info(session_id, optimizer)
        return LoadSessionResponse(
            session_id=info.session_id,
            task_type=info.task_type,
            model_type=info.model_type,
            input_type=info.input_type,
            metadata=info.metadata,
            filename=_relative_model_filename(path),
            path=str(path),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app() -> FastAPI:
    """Create the persistent FastAPI application."""
    app = FastAPI(title="bochan API with persistence", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()


__all__ = [
    "LoadSessionRequest",
    "LoadSessionResponse",
    "PERSISTENT_SESSION_STORE",
    "SaveSessionRequest",
    "SaveSessionResponse",
    "SavedModelsResponse",
    "app",
    "create_app",
    "router",
]
