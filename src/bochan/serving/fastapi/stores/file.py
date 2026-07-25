"""File-backed common model artifact utilities for FastAPI serving."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from bochan.api import BayesianOptimizer
from bochan.model_artifact import (
    MODEL_ARTIFACT_SUFFIX,
    ModelBackend,
    deserialize_model_artifact,
    infer_model_backend,
    save_model_artifact,
)
from bochan.tabular import TabularBayesianOptimizer

SupportedOptimizer = BayesianOptimizer | TabularBayesianOptimizer


class FileOptimizerStore:
    """Save and load tensor or tabular optimizers under one controlled directory.

    The storage directory is controlled by ``BOCHAN_API_MODEL_DIR`` and defaults
    to ``bochan_models``. New artifacts use the shared ``.bochan.pt`` envelope.
    Existing legacy tensor and Web artifacts remain readable. Loading uses pickle
    through ``torch.load`` and therefore requires explicit trust.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else None

    @property
    def root_dir(self) -> Path:
        root = self._root_dir or Path(os.environ.get("BOCHAN_API_MODEL_DIR", "bochan_models"))
        resolved = root.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def safe_path(self, filename: str | None, *, default_stem: str | None = None) -> Path:
        """Resolve a safe artifact path without allowing storage-root escape."""

        if filename is None or str(filename).strip() == "":
            filename = f"{default_stem or uuid4().hex}{MODEL_ARTIFACT_SUFFIX}"
        rel_path = Path(str(filename))
        if rel_path.is_absolute():
            raise ValueError("filename must be relative to BOCHAN_API_MODEL_DIR.")
        if any(part == ".." for part in rel_path.parts):
            raise ValueError("filename must not contain '..'.")
        if rel_path.suffix == "":
            rel_path = Path(f"{rel_path}{MODEL_ARTIFACT_SUFFIX}")

        root = self.root_dir
        full_path = (root / rel_path).resolve()
        try:
            full_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("filename escapes BOCHAN_API_MODEL_DIR.") from exc
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def relative_name(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root_dir))

    def list(self) -> list[str]:
        """List common and legacy ``.pt`` artifact names."""

        root = self.root_dir
        return sorted(str(path.relative_to(root)) for path in root.rglob("*.pt") if path.is_file())

    def save(
        self,
        optimizer: SupportedOptimizer,
        filename: str | None,
        *,
        default_stem: str | None = None,
        overwrite: bool = False,
        backend: ModelBackend | None = None,
        metadata: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> Path:
        """Save a tensor or tabular optimizer using the common artifact envelope."""

        path = self.safe_path(filename, default_stem=default_stem)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Model file already exists: {self.relative_name(path)}")
        save_model_artifact(
            optimizer,
            path,
            backend=backend,
            metadata=metadata,
            state=state,
        )
        return path

    def load_any(
        self,
        filename: str,
        *,
        map_location: str | None = "cpu",
        trust_pickle: bool = False,
    ) -> tuple[dict[str, Any], Path]:
        """Load a trusted artifact and return its normalized common envelope."""

        path = self.safe_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {self.relative_name(path)}")
        payload = deserialize_model_artifact(
            path,
            map_location=map_location,
            trust_pickle=trust_pickle,
        )
        return payload, path

    def load(
        self,
        filename: str,
        *,
        map_location: str | None = "cpu",
        trust_pickle: bool = False,
        expected_backend: ModelBackend = "tensor",
    ) -> tuple[SupportedOptimizer, Path]:
        """Load a trusted artifact for one API backend.

        ``expected_backend`` defaults to ``tensor`` to preserve the behavior of
        the existing ``/models/load`` endpoint. Tabular endpoints pass
        ``expected_backend='tabular'`` while using the same file format and store.
        """

        payload, path = self.load_any(
            filename,
            map_location=map_location,
            trust_pickle=trust_pickle,
        )
        optimizer = payload["optimizer"]
        actual_backend = infer_model_backend(optimizer)
        if actual_backend != expected_backend:
            raise TypeError(
                f"This endpoint expects a {expected_backend} model artifact, but "
                f"{self.relative_name(path)} contains a {actual_backend} optimizer."
            )
        return optimizer, path


__all__ = ["FileOptimizerStore", "SupportedOptimizer"]
