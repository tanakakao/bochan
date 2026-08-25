"""ALIGNN graph construction and local pretrained-bundle helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    "ALIGNN graph construction requires alignn and a compatible DGL installation. "
    "Install alignn==2026.8.11, then install a DGL build matching your "
    "PyTorch/CUDA environment as documented by ALIGNN."
)

_SUPPORTED_NEIGHBOR_STRATEGIES = {
    "k-nearest",
    "voronoi",
    "radius_graph",
    "radius_graph_jarvis",
    "fast_graph",
    "torch_graph",
}


def _alignn_graph_class() -> type[Any]:
    """Return upstream ``alignn.graphs.Graph`` after lazy dependency checks."""

    try:
        from alignn import graphs as graph_module
    except ImportError as error:
        raise ImportError(_ALIGNN_INSTALL_HINT) from error

    if getattr(graph_module, "dgl", None) is None:
        raise ImportError(_ALIGNN_INSTALL_HINT)
    graph_class = getattr(graph_module, "Graph", None)
    if not isinstance(graph_class, type):
        raise RuntimeError("The installed ALIGNN package does not expose alignn.graphs.Graph.")
    builder = getattr(graph_class, "atom_dgl_multigraph", None)
    if not callable(builder):
        raise RuntimeError("alignn.graphs.Graph.atom_dgl_multigraph is unavailable.")
    return graph_class


class ALIGNNGraphBuilder:
    """Build DGL atom graphs and line graphs using ALIGNN's canonical builder."""

    def __init__(
        self,
        *,
        neighbor_strategy: str = "k-nearest",
        cutoff: float = 8.0,
        max_neighbors: int | None = 12,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        adapter: StructureAdapter | None = None,
    ) -> None:
        if neighbor_strategy not in _SUPPORTED_NEIGHBOR_STRATEGIES:
            raise ValueError(
                "neighbor_strategy must be one of "
                f"{sorted(_SUPPORTED_NEIGHBOR_STRATEGIES)}, got {neighbor_strategy!r}."
            )
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
        if not isinstance(use_canonize, bool):
            raise TypeError("use_canonize must be a bool.")
        if adapter is not None and not isinstance(adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")

        self.neighbor_strategy = neighbor_strategy
        self.cutoff = float(cutoff)
        self.max_neighbors = max_neighbors
        self.atom_features = atom_features
        self.compute_line_graph = compute_line_graph
        self.use_canonize = use_canonize
        self.adapter = adapter or StructureAdapter()

    @classmethod
    def from_training_config(
        cls,
        config: Mapping[str, Any],
        *,
        adapter: StructureAdapter | None = None,
    ) -> ALIGNNGraphBuilder:
        """Build graph settings from an upstream ALIGNN training config."""

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        max_neighbors = config.get("max_neighbors", 12)
        if max_neighbors is not None:
            max_neighbors = int(max_neighbors)
        return cls(
            neighbor_strategy=str(config.get("neighbor_strategy", "k-nearest")),
            cutoff=float(config.get("cutoff", 8.0)),
            max_neighbors=max_neighbors,
            atom_features=str(config.get("atom_features", "cgcnn")),
            compute_line_graph=bool(config.get("compute_line_graph", True)),
            use_canonize=bool(config.get("use_canonize", True)),
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
            "use_canonize": self.use_canonize,
        }

    def build(self, structure: Any) -> Any:
        """Build one ALIGNN graph or ``(graph, line_graph)`` pair."""

        atoms = self.adapter.adapt(structure)
        graph_class = _alignn_graph_class()
        result = graph_class.atom_dgl_multigraph(
            atoms=atoms,
            neighbor_strategy=self.neighbor_strategy,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            atom_features=self.atom_features,
            compute_line_graph=self.compute_line_graph,
            use_canonize=self.use_canonize,
        )
        if self.compute_line_graph and (not isinstance(result, tuple) or len(result) != 2):
            raise RuntimeError(
                "ALIGNN graph builder must return (graph, line_graph) when compute_line_graph=True."
            )
        return result

    def build_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Build a structure graph bank for ``ALIGNNGPModel`` / ``ALIGNNDKLModel``."""

        atoms = self.adapter.adapt_many(structures)
        return tuple(self.build(item) for item in atoms)


class ALIGNNPretrainedBundle:
    """Parsed local scalar-property ALIGNN bundle.

    Model weights and graph-construction metadata stay together so pretrained
    embeddings can reuse the same graph settings as the upstream training run.
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
        if model_config.get("name") != "alignn":
            raise NotImplementedError(
                "Phase 2 pretrained loading supports scalar-property model.name='alignn' only."
            )

    @property
    def config(self) -> dict[str, Any]:
        """Return a shallow copy of the upstream training config."""

        return dict(self._config)

    @property
    def model_config(self) -> dict[str, Any]:
        """Return the upstream ``ALIGNNConfig`` mapping."""

        model_config = self._config["model"]
        return dict(model_config)

    def build_encoder(self, *, strict: bool = True) -> ALIGNNEncoder:
        """Instantiate ``ALIGNNEncoder`` from the bundled config and weights."""

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
        """Build graph settings matching the pretrained training config."""

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
    """Load an extracted ALIGNN training directory or upstream pretrained ZIP.

    Only local paths are supported. The function deliberately does not download
    model archives or execute arbitrary pickle payloads; checkpoints are loaded
    with ``weights_only=True``.
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
