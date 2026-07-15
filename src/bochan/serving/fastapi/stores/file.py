"""File-backed optimizer artifact utilities for FastAPI serving."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from bochan.api import BayesianOptimizer


class FileOptimizerStore:
    """Save and load optimizer artifacts under a controlled directory.

    The storage directory is controlled by ``BOCHAN_API_MODEL_DIR`` and defaults
    to ``bochan_models``. Filenames must be relative paths that do not contain
    ``..`` and cannot escape the storage root. Loading uses pickle via
    ``torch.load`` and therefore requires explicit trust.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        """Create a file artifact store.

        Args:
            root_dir: Optional root directory. When omitted,
                ``BOCHAN_API_MODEL_DIR`` or ``bochan_models`` is used.
        """
        self._root_dir = Path(root_dir) if root_dir is not None else None

    @property
    def root_dir(self) -> Path:
        """Return the resolved artifact root, creating it if necessary."""
        root = self._root_dir or Path(os.environ.get("BOCHAN_API_MODEL_DIR", "bochan_models"))
        resolved = root.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def safe_path(self, filename: str | None, *, default_stem: str | None = None) -> Path:
        """Resolve a safe model artifact path under the root directory.

        Args:
            filename: Relative filename requested by the client.
            default_stem: Stem used when ``filename`` is empty.

        Returns:
            Resolved path under :attr:`root_dir`.

        Raises:
            ValueError: If the filename is absolute, contains ``..``, or escapes
                the artifact root.
        """
        if filename is None or str(filename).strip() == "":
            filename = f"{default_stem or uuid4().hex}.pt"
        rel_path = Path(str(filename))
        if rel_path.is_absolute():
            raise ValueError("filename must be relative to BOCHAN_API_MODEL_DIR.")
        if any(part == ".." for part in rel_path.parts):
            raise ValueError("filename must not contain '..'.")
        if rel_path.suffix == "":
            rel_path = rel_path.with_suffix(".pt")

        root = self.root_dir
        full_path = (root / rel_path).resolve()
        try:
            full_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("filename escapes BOCHAN_API_MODEL_DIR.") from exc
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def relative_name(self, path: Path) -> str:
        """Return a root-relative artifact name for API responses.

        Args:
            path: Resolved artifact path.

        Returns:
            Relative path string.
        """
        return str(path.resolve().relative_to(self.root_dir))

    def list(self) -> list[str]:
        """List saved ``.pt`` optimizer artifact names."""
        root = self.root_dir
        return sorted(str(path.relative_to(root)) for path in root.rglob("*.pt") if path.is_file())

    def save(self, optimizer: BayesianOptimizer, filename: str | None, *, default_stem: str | None = None, overwrite: bool = False) -> Path:
        """Save an optimizer artifact.

        Args:
            optimizer: Optimizer to serialize.
            filename: Relative target filename.
            default_stem: Fallback stem when filename is omitted.
            overwrite: Whether existing files may be replaced.

        Returns:
            Saved artifact path.

        Raises:
            FileExistsError: If the target exists and overwrite is false.
        """
        path = self.safe_path(filename, default_stem=default_stem)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Model file already exists: {self.relative_name(path)}")
        import torch

        torch.save({"version": 1, "object_type": "BayesianOptimizer", "optimizer": optimizer}, path)
        return path

    def load(self, filename: str, *, map_location: str | None = "cpu", trust_pickle: bool = False) -> tuple[BayesianOptimizer, Path]:
        """Load an optimizer artifact after explicit pickle trust confirmation.

        Args:
            filename: Relative artifact filename.
            map_location: Torch map location.
            trust_pickle: Must be true because ``torch.load`` uses pickle.

        Returns:
            Loaded optimizer and artifact path.

        Raises:
            ValueError: If ``trust_pickle`` is false or filename is unsafe.
            FileNotFoundError: If the artifact is missing.
            TypeError: If the artifact does not contain a ``BayesianOptimizer``.
        """
        if not trust_pickle:
            raise ValueError(
                "Loading uses torch.load / pickle. Set trust_pickle=true only for model files you trust; never load untrusted artifacts."
            )
        path = self.safe_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {self.relative_name(path)}")
        import torch

        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        optimizer = payload.get("optimizer") if isinstance(payload, dict) else payload
        if not isinstance(optimizer, BayesianOptimizer):
            raise TypeError("Loaded object is not a BayesianOptimizer model.")
        return optimizer, path
