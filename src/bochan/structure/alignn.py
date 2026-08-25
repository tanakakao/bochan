"""ALIGNN graph construction from canonical crystal structures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TypeAlias

import numpy as np
import torch
from torch import Tensor

from .adapter import StructureAdapter

ALIGNNGraphBundle: TypeAlias = tuple[Any, Any, Tensor]
GraphFactory: TypeAlias = Callable[..., Any]

_ALIGNN_INSTALL_HINT = (
    "ALIGNN graph construction requires the optional ALIGNN dependencies. "
    "Install ALIGNN, JARVIS-Tools, and a DGL build compatible with your PyTorch environment."
)
_SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "bfloat": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class ALIGNNGraphConfig:
    """Configuration matching the upstream ALIGNN crystal-graph defaults."""

    neighbor_strategy: str = "k-nearest"
    cutoff: float = 5.0
    cutoff_extra: float = 3.0
    max_neighbors: int = 12
    atom_features: str = "cgcnn"
    use_canonize: bool = True
    dtype: str = "float32"
    three_body_cutoff: float | None = 3.5

    def __post_init__(self) -> None:
        if not isinstance(self.neighbor_strategy, str) or not self.neighbor_strategy:
            raise ValueError("neighbor_strategy must be a non-empty string.")
        if isinstance(self.cutoff, bool) or not isinstance(self.cutoff, (int, float)) or self.cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if isinstance(self.cutoff_extra, bool) or not isinstance(self.cutoff_extra, (int, float)) or self.cutoff_extra < 0:
            raise ValueError("cutoff_extra must be non-negative.")
        if isinstance(self.max_neighbors, bool) or not isinstance(self.max_neighbors, int) or self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be a positive integer.")
        if not isinstance(self.atom_features, str) or not self.atom_features:
            raise ValueError("atom_features must be a non-empty string.")
        if self.dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_DTYPES))
            raise ValueError(f"dtype must be one of: {supported}.")
        if self.three_body_cutoff is not None and (
            isinstance(self.three_body_cutoff, bool)
            or not isinstance(self.three_body_cutoff, (int, float))
            or self.three_body_cutoff <= 0
        ):
            raise ValueError("three_body_cutoff must be positive or None.")
        if self.three_body_cutoff is not None and self.three_body_cutoff > self.cutoff:
            raise ValueError("three_body_cutoff must not exceed cutoff.")


class ALIGNNGraphBuilder:
    """Convert crystal structures into graph bundles consumed by ALIGNN-GP/DKL.

    Upstream ``Graph.atom_dgl_multigraph`` currently returns ``(g, lg)`` while
    the current scalar ALIGNN ``forward`` contract consumes ``(g, lg, lattice)``.
    Bochan therefore normalizes every built structure to the three-element
    ``(graph, line_graph, lattice_tensor)`` bundle. Phase-1 ``ALIGNNEncoder``
    accepts this bundle and uses the graph and line graph for its pooled
    representation backbone.
    """

    def __init__(
        self,
        config: ALIGNNGraphConfig | None = None,
        *,
        structure_adapter: StructureAdapter | None = None,
        graph_factory: GraphFactory | None = None,
    ) -> None:
        if config is not None and not isinstance(config, ALIGNNGraphConfig):
            raise TypeError("config must be an ALIGNNGraphConfig or None.")
        if structure_adapter is not None and not isinstance(structure_adapter, StructureAdapter):
            raise TypeError("structure_adapter must be a StructureAdapter or None.")
        if graph_factory is not None and not callable(graph_factory):
            raise TypeError("graph_factory must be callable or None.")
        self.config = config or ALIGNNGraphConfig()
        self.structure_adapter = structure_adapter or StructureAdapter()
        self._graph_factory = graph_factory

    def _resolve_graph_factory(self) -> GraphFactory:
        if self._graph_factory is not None:
            return self._graph_factory
        try:
            module = import_module("alignn.graphs")
        except ImportError as error:
            raise ImportError(_ALIGNN_INSTALL_HINT) from error
        graph_class = getattr(module, "Graph", None)
        graph_factory = getattr(graph_class, "atom_dgl_multigraph", None)
        if not callable(graph_factory):
            raise RuntimeError("The installed ALIGNN package does not expose Graph.atom_dgl_multigraph().")
        self._graph_factory = graph_factory
        return graph_factory

    def _lattice_tensor(self, atoms: Any) -> Tensor:
        lattice = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
        if lattice.shape != (3, 3):
            raise ValueError(f"JARVIS atoms lattice_mat must have shape (3, 3), got {lattice.shape}.")
        if not np.isfinite(lattice).all():
            raise ValueError("JARVIS atoms lattice_mat must contain only finite values.")
        return torch.as_tensor(lattice, dtype=_SUPPORTED_DTYPES[self.config.dtype])

    @staticmethod
    def _normalize_graph_result(result: Any, *, lattice: Tensor) -> ALIGNNGraphBundle:
        if not isinstance(result, (tuple, list)):
            raise TypeError("ALIGNN graph factory must return (graph, line_graph) or (graph, line_graph, lattice).")
        if len(result) == 2:
            graph, line_graph = result
            resolved_lattice = lattice
        elif len(result) == 3:
            graph, line_graph, returned_lattice = result
            if returned_lattice is None:
                resolved_lattice = lattice
            elif torch.is_tensor(returned_lattice):
                resolved_lattice = returned_lattice.to(dtype=lattice.dtype)
            else:
                resolved_lattice = torch.as_tensor(returned_lattice, dtype=lattice.dtype)
        else:
            raise ValueError(
                "ALIGNN graph factory must return exactly two or three values: graph, line_graph, and optional lattice."
            )
        if graph is None or line_graph is None:
            raise ValueError("ALIGNN graph construction must produce both a graph and a line graph.")
        if resolved_lattice.shape != (3, 3):
            raise ValueError(f"ALIGNN lattice tensor must have shape (3, 3), got {tuple(resolved_lattice.shape)}.")
        if not torch.isfinite(resolved_lattice).all():
            raise ValueError("ALIGNN lattice tensor must contain only finite values.")
        return graph, line_graph, resolved_lattice

    def build(self, structure: Any) -> ALIGNNGraphBundle:
        """Build one canonical ALIGNN graph bundle from ``structure``."""

        atoms = self.structure_adapter.to_jarvis(structure)
        graph_factory = self._resolve_graph_factory()
        config = self.config
        result = graph_factory(
            atoms=atoms,
            neighbor_strategy=config.neighbor_strategy,
            cutoff=float(config.cutoff),
            cutoff_extra=float(config.cutoff_extra),
            max_neighbors=config.max_neighbors,
            atom_features=config.atom_features,
            compute_line_graph=True,
            use_canonize=config.use_canonize,
            dtype=config.dtype,
            three_body_cutoff=config.three_body_cutoff,
        )
        return self._normalize_graph_result(result, lattice=self._lattice_tensor(atoms))

    def build_many(self, structures: Sequence[Any]) -> tuple[ALIGNNGraphBundle, ...]:
        """Build an ordered graph bank suitable for ``structure_graphs=...``."""

        if not isinstance(structures, Sequence) or isinstance(structures, (str, bytes)):
            raise TypeError("structures must be a sequence of crystal structures.")
        if not structures:
            raise ValueError("structures must contain at least one crystal structure.")
        return tuple(self.build(structure) for structure in structures)


__all__ = ["ALIGNNGraphBuilder", "ALIGNNGraphBundle", "ALIGNNGraphConfig"]
