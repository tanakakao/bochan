"""Deployment-time TabPFN model asset management for bochan Web.

The public Web runtime intentionally does not authenticate with Prior Labs or
attempt to download TabPFN checkpoints while handling user requests. Required
foundation-model weights are downloaded ahead of time by a deployment/preload
step and are then treated as immutable runtime assets.

This module intentionally lives outside ``bochan.serving.webapp`` so the preload
CLI can run without importing the Web application and its BoTorch model stack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

_TABPFN_WEB_VERSION = "v3"
_TABPFN_NO_BROWSER_ENV = "TABPFN_NO_BROWSER"
_TABPFN_MODEL_CACHE_ENV = "TABPFN_MODEL_CACHE_DIR"

TabPFNModelKind = Literal["classifier", "regressor"]


def _tabpfn_model_loading() -> tuple[Any, Any, Any, Any]:
    """Import the optional TabPFN download API lazily."""

    try:
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import download_model, get_cache_dir, resolve_model_path
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TabPFN Web assets require the optional dependency. "
            "Install bochan with `pip install 'bochan[web]'` or `bochan[tabpfn]`."
        ) from exc
    return ModelVersion, download_model, get_cache_dir, resolve_model_path


def tabpfn_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the checkpoint directory used by both preload and Web runtime."""

    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    _, _, get_cache_dir, _ = _tabpfn_model_loading()
    return Path(get_cache_dir()).expanduser().resolve()


def _default_model_filename(which: TabPFNModelKind) -> str:
    """Resolve TabPFN's official default v3 checkpoint filename."""

    _, _, _, resolve_model_path = _tabpfn_model_loading()
    _, _, names, _ = resolve_model_path(
        None,
        which=which,
        version=_TABPFN_WEB_VERSION,
    )
    if len(names) != 1:
        raise RuntimeError(
            "TabPFN default model resolution returned an unexpected number of checkpoints: "
            f"{names!r}."
        )
    return str(names[0])


def required_tabpfn_assets(
    cache_dir: str | Path | None = None,
) -> dict[TabPFNModelKind, Path]:
    """Return the exact default classifier/regressor files required by Web."""

    root = tabpfn_cache_dir(cache_dir)
    return {
        "classifier": root / _default_model_filename("classifier"),
        "regressor": root / _default_model_filename("regressor"),
    }


def tabpfn_asset_status(cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Inspect preloaded TabPFN assets without authenticating or downloading."""

    root = tabpfn_cache_dir(cache_dir)
    assets = required_tabpfn_assets(root)
    models: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for which, path in assets.items():
        exists = path.is_file() and path.stat().st_size > 0
        if not exists:
            missing.append(path.name)
        models[which] = {
            "filename": path.name,
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
        }
    return {
        "available": not missing,
        "cache_dir": str(root),
        "version": _TABPFN_WEB_VERSION,
        "models": models,
        "missing_models": missing,
    }


def require_preloaded_tabpfn_assets(cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Fail before estimator construction when deployment assets are missing."""

    status = tabpfn_asset_status(cache_dir)
    if status["available"]:
        return status

    missing = ", ".join(status["missing_models"])
    raise RuntimeError(
        "TabPFN is not provisioned for the bochan Web runtime. "
        f"Missing preloaded model weight(s): {missing}. "
        "Download the weights during deployment with "
        "`python -m bochan.tabpfn_preload` and make the same "
        f"checkpoint directory available at runtime via {_TABPFN_MODEL_CACHE_ENV}. "
        "Runtime authentication and model downloads are intentionally disabled."
    )


def _download_default_model(root: Path, which: TabPFNModelKind) -> Path:
    """Download one official default v3 checkpoint to ``root``."""

    ModelVersion, download_model, _, _ = _tabpfn_model_loading()
    filename = _default_model_filename(which)
    path = root / filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    if path.exists():
        path.unlink()

    result = download_model(
        to=path,
        version=ModelVersion.V3,
        which=which,
        model_name=filename,
    )
    if result != "ok":
        details = "; ".join(str(error) for error in list(result or []))
        raise RuntimeError(
            f"Failed to preload TabPFN {which} checkpoint {filename}. "
            f"{details or 'TabPFN did not provide an error message.'}"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(
            f"TabPFN reported a successful download but checkpoint is missing or empty: {path}"
        )
    return path


def preload_tabpfn_assets(
    cache_dir: str | Path | None = None,
    *,
    allow_browser_auth: bool = False,
) -> dict[str, Any]:
    """Download the Web runtime's TabPFN assets during deployment.

    By default browser authentication is disabled so this command is safe for CI,
    containers, and deployment jobs. Supply ``TABPFN_TOKEN`` through the platform's
    secret manager. Interactive local provisioning can explicitly opt into the
    library-managed browser flow with ``allow_browser_auth=True``.
    """

    root = tabpfn_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if allow_browser_auth:
        os.environ.pop(_TABPFN_NO_BROWSER_ENV, None)
    else:
        os.environ[_TABPFN_NO_BROWSER_ENV] = "1"

    _download_default_model(root, "classifier")
    _download_default_model(root, "regressor")
    status = tabpfn_asset_status(root)
    if not status["available"]:  # defensive consistency check
        raise RuntimeError(
            "TabPFN preload completed without all required Web runtime checkpoints."
        )
    return status


__all__ = [
    "preload_tabpfn_assets",
    "require_preloaded_tabpfn_assets",
    "required_tabpfn_assets",
    "tabpfn_asset_status",
    "tabpfn_cache_dir",
]
