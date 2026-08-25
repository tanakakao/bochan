"""Canonical crystal-structure adapters for atomistic models."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

_STRUCTURE_INSTALL_HINT = (
    "Crystal-structure conversion requires JARVIS-Tools. "
    "Install the ALIGNN/JARVIS optional dependencies before using StructureAdapter."
)


def _as_3x3_lattice(value: Any) -> np.ndarray:
    lattice = np.asarray(value, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"lattice_mat must have shape (3, 3), got {lattice.shape}.")
    if not np.isfinite(lattice).all():
        raise ValueError("lattice_mat must contain only finite values.")
    if abs(float(np.linalg.det(lattice))) <= 1e-12:
        raise ValueError("lattice_mat must be non-singular.")
    return lattice


def _as_coordinates(value: Any, *, n_atoms: int) -> np.ndarray:
    coords = np.asarray(value, dtype=float)
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"coords must have shape ({n_atoms}, 3), got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError("coords must contain only finite values.")
    return coords


def _element_symbol(species: Any) -> str:
    symbol = getattr(species, "symbol", None)
    if isinstance(symbol, str) and symbol:
        return symbol
    element = getattr(species, "element", None)
    symbol = getattr(element, "symbol", None)
    if isinstance(symbol, str) and symbol:
        return symbol
    value = str(species)
    if not value:
        raise ValueError("Could not determine an element symbol from pymatgen species.")
    return value


class StructureAdapter:
    """Convert common crystal-structure inputs to ``jarvis.core.atoms.Atoms``.

    Supported inputs are JARVIS ``Atoms``, mappings with ``lattice_mat`` /
    ``coords`` / ``elements``, ASE ``Atoms``, pymatgen ``Structure``, and
    trusted local CIF/POSCAR paths. Optional packages are resolved lazily so
    importing :mod:`bochan.structure` does not require ALIGNN or JARVIS.
    """

    def __init__(self, *, atoms_class: type[Any] | None = None) -> None:
        self._atoms_class = atoms_class

    def _resolve_atoms_class(self) -> type[Any]:
        if self._atoms_class is not None:
            return self._atoms_class
        try:
            module = import_module("jarvis.core.atoms")
        except ImportError as error:
            raise ImportError(_STRUCTURE_INSTALL_HINT) from error
        atoms_class = getattr(module, "Atoms", None)
        if not isinstance(atoms_class, type):
            raise RuntimeError("JARVIS-Tools does not expose jarvis.core.atoms.Atoms.")
        self._atoms_class = atoms_class
        return atoms_class

    @staticmethod
    def _normalize_elements(elements: Any) -> list[str]:
        if isinstance(elements, (str, bytes)):
            raise TypeError("elements must be a sequence of element symbols.")
        try:
            normalized = [str(element) for element in elements]
        except TypeError as error:
            raise TypeError("elements must be a sequence of element symbols.") from error
        if not normalized or any(not element for element in normalized):
            raise ValueError("elements must contain at least one non-empty element symbol.")
        return normalized

    def _from_mapping(self, structure: Mapping[str, Any], atoms_class: type[Any]) -> Any:
        missing = [key for key in ("lattice_mat", "coords", "elements") if key not in structure]
        if missing:
            raise ValueError(f"Structure mapping is missing required keys: {', '.join(missing)}.")

        elements = self._normalize_elements(structure["elements"])
        lattice = _as_3x3_lattice(structure["lattice_mat"])
        coords = _as_coordinates(structure["coords"], n_atoms=len(elements))
        cartesian = structure.get("cartesian", False)
        if not isinstance(cartesian, bool):
            raise TypeError("Structure mapping 'cartesian' must be a bool when provided.")

        kwargs: dict[str, Any] = {
            "lattice_mat": lattice,
            "coords": coords,
            "elements": elements,
            "cartesian": cartesian,
        }
        if "props" in structure:
            kwargs["props"] = structure["props"]
        return atoms_class(**kwargs)

    def _from_ase(self, structure: Any, atoms_class: type[Any]) -> Any:
        pbc = np.asarray(structure.get_pbc(), dtype=bool)
        if pbc.shape != (3,) or not bool(np.all(pbc)):
            raise ValueError("ASE structures must be periodic in all three directions for ALIGNN crystal graphs.")
        elements = self._normalize_elements(structure.get_chemical_symbols())
        cell_object = structure.get_cell()
        cell = getattr(cell_object, "array", cell_object)
        lattice = _as_3x3_lattice(cell)
        coords = _as_coordinates(structure.get_positions(), n_atoms=len(elements))
        return atoms_class(
            lattice_mat=lattice,
            coords=coords,
            elements=elements,
            cartesian=True,
        )

    def _from_pymatgen(self, structure: Any, atoms_class: type[Any]) -> Any:
        is_ordered = getattr(structure, "is_ordered", True)
        if not bool(is_ordered):
            raise ValueError("Disordered pymatgen structures are not supported; resolve occupancies first.")
        species = structure.species
        elements = [_element_symbol(item) for item in species]
        lattice = _as_3x3_lattice(structure.lattice.matrix)
        coords = _as_coordinates(structure.frac_coords, n_atoms=len(elements))
        return atoms_class(
            lattice_mat=lattice,
            coords=coords,
            elements=elements,
            cartesian=False,
        )

    @staticmethod
    def _infer_path_format(path: Path) -> str:
        suffix = path.suffix.lower()
        name = path.name.upper()
        if suffix == ".cif":
            return "cif"
        if suffix in {".vasp", ".poscar"} or name in {"POSCAR", "CONTCAR"}:
            return "poscar"
        raise ValueError(
            "Could not infer the structure file format. Supported local files are .cif, .vasp, .poscar, POSCAR, and CONTCAR."
        )

    def _from_path(self, value: str | PathLike[str], atoms_class: type[Any]) -> Any:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Structure file does not exist: {path}")
        file_format = self._infer_path_format(path)
        if file_format == "cif":
            loader = getattr(atoms_class, "from_cif", None)
            if not callable(loader):
                raise RuntimeError("The installed JARVIS Atoms class does not expose from_cif().")
            return loader(
                filename=str(path),
                get_primitive_atoms=False,
                use_cif2cell=False,
            )
        loader = getattr(atoms_class, "from_poscar", None)
        if not callable(loader):
            raise RuntimeError("The installed JARVIS Atoms class does not expose from_poscar().")
        return loader(filename=str(path))

    def to_jarvis(self, structure: Any) -> Any:
        """Return a canonical JARVIS ``Atoms`` object for ``structure``."""

        atoms_class = self._resolve_atoms_class()
        if isinstance(structure, atoms_class):
            return structure
        if isinstance(structure, Mapping):
            return self._from_mapping(structure, atoms_class)
        if isinstance(structure, (str, PathLike)):
            return self._from_path(structure, atoms_class)

        if all(
            callable(getattr(structure, name, None))
            for name in ("get_chemical_symbols", "get_cell", "get_positions", "get_pbc")
        ):
            return self._from_ase(structure, atoms_class)

        lattice = getattr(structure, "lattice", None)
        if lattice is not None and hasattr(lattice, "matrix") and hasattr(structure, "species") and hasattr(
            structure, "frac_coords"
        ):
            return self._from_pymatgen(structure, atoms_class)

        raise TypeError(
            "Unsupported crystal structure type. Expected JARVIS Atoms, a structure mapping, ASE Atoms, "
            "pymatgen Structure, or a trusted local CIF/POSCAR path."
        )


__all__ = ["StructureAdapter"]
