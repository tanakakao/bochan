"""Pure-PyTorch ALIGNN graph construction and pretrained-bundle helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from io import BytesIO
from json import loads
from os import PathLike
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import torch

from bochan.composition import ALIGNNEncoder

from .adapter import StructureAdapter

_ALIGNN_INSTALL_HINT = (
    "ALIGNN graph construction requires alignn with build_pure_torch_graph. "
    "Install alignn==2026.8.11 or a compatible newer release. DGL is not required."
)
_SUPPORTED_DTYPES = {"float16", "float32", "float64", "bfloat"}
_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat": torch.bfloat16,
}
_PURE_MODEL_NAME = "alignn_atomwise_pure"


def _pure_graph_builder():
    """Return upstream ``build_pure_torch_graph`` after lazy dependency checks."""

    try:
        module = import_module("alignn.torch_graph_builder")
    except ImportError as error:
        raise ImportError(_ALIGNN_INSTALL_HINT) from error
    builder = getattr(module, "build_pure_torch_graph", None)
    if not callable(builder):
        raise RuntimeError("The installed ALIGNN package does not expose build_pure_torch_graph.")
    return builder


class ALIGNNGraphBuilder:
    """Build ALIGNN ``TorchGraph`` atom/line graphs without DGL.

    ``neighbor_strategy`` is intentionally fixed to ``"pure_torch"``. The
    upstream pure builder performs periodic neighbor search with torch tensors,
    caps pair neighbors per source atom, and constructs the line graph with a
    separate three-body cutoff.

    For pretrained models, prefer :meth:`from_training_config` or
    :meth:`ALIGNNPretrainedBundle.build_graph_builder` so graph construction
    follows the checkpoint training metadata.
    """

    def __init__(
        self,
        *,
        neighbor_strategy: str = "pure_torch",
        cutoff: float = 8.0,
        max_neighbors: int | None = 12,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        dtype: str = "float32",
        three_body_cutoff: float | None = 3.5,
        adapter: StructureAdapter | None = None,
    ) -> None:
        if neighbor_strategy != "pure_torch":
            raise ValueError("Pure-PyTorch ALIGNN requires neighbor_strategy='pure_torch'.")
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or cutoff <= 0:
            raise ValueError("cutoff must be a positive number.")
        if max_neighbors is not None and (
            isinstance(max_neighbors, bool)
            or not isinstance(max_neighbors, int)
            or max_neighbors <= 0
        ):
            raise ValueError("max_neighbors must be a positive integer or None.")
        if not isinstance(atom_features, str) or not atom_features:
            raise ValueError("atom_features must be a non-empty string.")
        if not isinstance(compute_line_graph, bool):
            raise TypeError("compute_line_graph must be a bool.")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_SUPPORTED_DTYPES)}, got {dtype!r}.")
        if three_body_cutoff is not None:
            if (
                isinstance(three_body_cutoff, bool)
                or not isinstance(three_body_cutoff, (int, float))
                or three_body_cutoff <= 0
            ):
                raise ValueError("three_body_cutoff must be a positive number or None.")
            if three_body_cutoff > cutoff:
                raise ValueError("three_body_cutoff must not exceed cutoff.")
        if adapter is not None and not isinstance(adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")

        self.neighbor_strategy = neighbor_strategy
        self.cutoff = float(cutoff)
        self.max_neighbors = max_neighbors
        self.atom_features = atom_features
        self.compute_line_graph = compute_line_graph
        self.dtype = dtype
        self.three_body_cutoff = None if three_body_cutoff is None else float(three_body_cutoff)
        self.adapter = adapter or StructureAdapter()

    @classmethod
    def from_training_config(
        cls,
        config: Mapping[str, Any],
        *,
        adapter: StructureAdapter | None = None,
    ) -> ALIGNNGraphBuilder:
        """Build graph settings from an upstream pure ALIGNN training config."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        neighbor_strategy = str(config.get("neighbor_strategy", "pure_torch"))
        if neighbor_strategy != "pure_torch":
            raise ValueError(
                "Pretrained ALIGNN config is not pure PyTorch: "
                f"neighbor_strategy={neighbor_strategy!r}."
            )
        max_neighbors = config.get("max_neighbors", 12)
        if max_neighbors is not None:
            max_neighbors = int(max_neighbors)
        three_body_cutoff = config.get("three_body_cutoff", 3.5)
        if three_body_cutoff is None:
            model_config = config.get("model")
            if isinstance(model_config, Mapping):
                three_body_cutoff = model_config.get("three_body_cutoff", 3.5)
        if three_body_cutoff is not None:
            three_body_cutoff = float(three_body_cutoff)
        return cls(
            neighbor_strategy=neighbor_strategy,
            cutoff=float(config.get("cutoff", 8.0)),
            max_neighbors=max_neighbors,
            atom_features=str(config.get("atom_features", "cgcnn")),
            compute_line_graph=bool(config.get("compute_line_graph", True)),
            dtype=str(config.get("dtype", "float32")),
            three_body_cutoff=three_body_cutoff,
            adapter=adapter,
        )

    @property
    def config(self) -> dict[str, Any]:
        """Return graph settings suitable for serialization and debugging."""

        return {
            "neighbor_strategy": self.neighbor_strategy,
            "cutoff": self.cutoff,
            "max_neighbors": self.max_neighbors,
            "atom_features": self.atom_features,
            "compute_line_graph": self.compute_line_graph,
            "dtype": self.dtype,
            "three_body_cutoff": self.three_body_cutoff,
        }

    def build(self, structure: Any) -> Any:
        """Build one pure ALIGNN ``(TorchGraph, TorchGraph)`` pair."""

        atoms = self.adapter.adapt(structure)
        dtype = _TORCH_DTYPES[self.dtype]
        positions = torch.as_tensor(atoms.cart_coords, dtype=dtype)
        lattice = torch.as_tensor(atoms.lattice_mat, dtype=dtype)
        builder = _pure_graph_builder()
        result = builder(
            atoms=atoms,
            two_body_cutoff=self.cutoff,
            three_body_cutoff=self.three_body_cutoff,
            max_neighbors=self.max_neighbors,
            atom_features=self.atom_features,
            positions=positions,
            lattice=lattice,
            use_matscipy_topology=False,
            compute_line_graph=self.compute_line_graph,
        )
        if self.compute_line_graph and (not isinstance(result, tuple) or len(result) != 2):
            raise RuntimeError(
                "Pure ALIGNN graph builder must return (graph, line_graph) "
                "when compute_line_graph=True."
            )
        return result

    def build_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Build a pure-Torch structure graph bank for ALIGNN-GP / ALIGNN-DKL."""

        atoms = self.adapter.adapt_many(structures)
        return tuple(self.build(item) for item in atoms)


class ALIGNNPretrainedBundle:
    """Parsed local pure-PyTorch scalar-property ALIGNN bundle.

    Model weights and graph-construction metadata stay together so pretrained
    embeddings reuse the same pair/three-body settings as the upstream run.
    Legacy DGL ``model.name='alignn'`` bundles are deliberately rejected rather
    than silently crossing model/graph contracts.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        source: str,
        checkpoint_name: str,
    ) -> None:
        self._config = dict(config)
        self._checkpoint = dict(checkpoint)
        self.source = source
        self.checkpoint_name = checkpoint_name

        model_config = self._config.get("model")
        if not isinstance(model_config, Mapping):
            raise ValueError("ALIGNN pretrained config must contain a model mapping.")
        if model_config.get("name") != _PURE_MODEL_NAME:
            raise NotImplementedError(
                "Bochan pretrained loading requires the pure-PyTorch "
                f"model.name={_PURE_MODEL_NAME!r}. Legacy DGL ALIGNN bundles are not accepted."
            )
        if str(self._config.get("neighbor_strategy", "pure_torch")) != "pure_torch":
            raise NotImplementedError(
                "Bochan pretrained loading requires neighbor_strategy='pure_torch'."
            )

    @property
    def config(self) -> dict[str, Any]:
        """Return a shallow copy of the upstream training config."""

        return dict(self._config)

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the upstream ``ALIGNNAtomWisePureConfig`` mapping."""

        model_config = self._config["model"]
        return dict(model_config)

    def build_encoder(self, *, strict: bool = True) -> ALIGNNEncoder:
        """Instantiate the pure ``ALIGNNEncoder`` from bundled weights."""

        return ALIGNNEncoder(
            config=self.model_config,
            checkpoint=self._checkpoint,
            strict_checkpoint=strict,
        )

    def build_graph_builder(
        self,
        *,
        adapter: StructureAdapter | None = None,
    ) -> ALIGNNGraphBuilder:
        """Build pure graph settings matching the pretrained training config."""

        return ALIGNNGraphBuilder.from_training_config(self._config, adapter=adapter)


def _checkpoint_rank(name: str) -> tuple[int, int, str]:
    """Rank numbered checkpoints numerically, with deterministic fallback."""

    stem = Path(name).stem
    suffix = stem.removeprefix("checkpoint_")
    if suffix.isdigit():
        return 1, int(suffix), name
    return 0, -1, name


def _select_checkpoint(names: Sequence[str]) -> str:
    best = sorted(name for name in names if Path(name).name == "best_model.pt")
    if best:
        return best[0]
    checkpoints = [
        name
        for name in names
        if Path(name).name.startswith("checkpoint_") and Path(name).suffix == ".pt"
    ]
    if checkpoints:
        return max(checkpoints, key=_checkpoint_rank)
    raise FileNotFoundError("ALIGNN bundle contains neither best_model.pt nor checkpoint_*.pt.")


def _load_checkpoint_bytes(data: bytes) -> Mapping[str, Any]:
    loaded = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("ALIGNN checkpoint must deserialize to a mapping.")
    return loaded


def load_alignn_pretrained_bundle(path: str | PathLike[str]) -> ALIGNNPretrainedBundle:
    """Load an extracted pure ALIGNN training directory or ZIP bundle.

    Only local paths are supported. The function does not download model
    archives or execute arbitrary pickle payloads; checkpoints are loaded with
    ``weights_only=True``.
    """

    resolved = Path(path)
    if resolved.is_dir():
        config_paths = sorted(resolved.rglob("config.json"))
        if not config_paths:
            raise FileNotFoundError(f"ALIGNN config.json does not exist in {resolved}.")
        config_path = config_paths[0]
        checkpoint_paths = sorted(resolved.rglob("*.pt"))
        relative_names = [str(item.relative_to(resolved)) for item in checkpoint_paths]
        checkpoint_name = _select_checkpoint(relative_names)
        checkpoint_path = resolved / checkpoint_name
        config = loads(config_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        source = str(resolved)
    elif resolved.is_file():
        try:
            with ZipFile(resolved) as archive:
                names = archive.namelist()
                configs = sorted(name for name in names if Path(name).name == "config.json")
                if not configs:
                    raise FileNotFoundError("ALIGNN ZIP bundle does not contain config.json.")
                config_name = configs[0]
                checkpoint_name = _select_checkpoint(names)
                config = loads(archive.read(config_name).decode("utf-8"))
                checkpoint = _load_checkpoint_bytes(archive.read(checkpoint_name))
        except BadZipFile as error:
            raise ValueError(f"ALIGNN pretrained bundle is not a valid ZIP archive: {resolved}") from error
        source = str(resolved)
    else:
        raise FileNotFoundError(f"ALIGNN pretrained bundle does not exist: {resolved}")

    if not isinstance(config, Mapping):
        raise TypeError("ALIGNN config.json must contain a JSON object.")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("ALIGNN checkpoint must deserialize to a mapping.")

    return ALIGNNPretrainedBundle(
        config=config,
        checkpoint=checkpoint,
        source=source,
        checkpoint_name=checkpoint_name,
    )


__all__ = [
    "ALIGNNGraphBuilder",
    "ALIGNNPretrainedBundle",
    "load_alignn_pretrained_bundle",
]
