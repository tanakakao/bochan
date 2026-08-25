"""Canonical crystal-structure adapters for ALIGNN integrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import numpy as np

_STRUCTURE_INSTALL_HINT = (
    "Crystal-structure support requires jarvis-tools. "
    "Install alignn==2026.8.11 for the tested ALIGNN integration, "
    "or install jarvis-tools directly for structure conversion."
)


def _jarvis_atoms_api() -> tuple[type[Any], Any, Any]:
    """Return JARVIS Atoms and conversion helpers after a lazy import."""

    try:
        module = import_module("jarvis.core.atoms")
    except ImportError as error:
        raise ImportError(_STRUCTURE_INSTALL_HINT) from error

    atoms_class = getattr(module, "Atoms", None)
    ase_to_atoms = getattr(module, "ase_to_atoms", None)
    pmg_to_atoms = getattr(module, "pmg_to_atoms", None)
    if not isinstance(atoms_class, type):
        raise RuntimeError("jarvis.core.atoms.Atoms is unavailable.")
    if not callable(ase_to_atoms) or not callable(pmg_to_atoms):
        raise RuntimeError("jarvis.core.atoms conversion helpers are unavailable.")
    return atoms_class, ase_to_atoms, pmg_to_atoms


def _as_float_array(name: str, value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric.") from error
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


class StructureAdapter:
    """Normalize common crystal-structure objects to ``jarvis.core.atoms.Atoms``.

    ``adapt`` intentionally does not accept filesystem paths. Local file access
    is explicit through :meth:`from_file`, which keeps later API/Web layers from
    accidentally treating user-provided strings as server-side paths.
    """

    def adapt(self, structure: Any) -> Any:
        """Return a JARVIS ``Atoms`` object for one in-memory structure."""

        atoms_class, ase_to_atoms, pmg_to_atoms = _jarvis_atoms_api()

        if isinstance(structure, atoms_class):
            return structure
        if isinstance(structure, Mapping):
            return self._from_mapping(structure, atoms_class=atoms_class)
        if isinstance(structure, (str, bytes, PathLike)):
            raise TypeError(
                "Filesystem paths are not accepted by StructureAdapter.adapt(); "
                "use StructureAdapter.from_file() explicitly."
            )

        try:
            ase_module = import_module("ase")
        except ImportError:
            ase_module = None
        if ase_module is not None:
            ase_atoms_class = getattr(ase_module, "Atoms", None)
            if isinstance(ase_atoms_class, type) and isinstance(structure, ase_atoms_class):
                return ase_to_atoms(ase_atoms=structure)

        if type(structure).__module__.startswith("pymatgen."):
            return pmg_to_atoms(pmg=structure)

        raise TypeError(
            "Unsupported structure type. Expected JARVIS Atoms, ASE Atoms, "
            "pymatgen Structure, or a mapping with lattice_mat/coords/elements."
        )

    def adapt_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Normalize a non-empty sequence of in-memory structures."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")
        return tuple(self.adapt(structure) for structure in structures)

    def from_file(
        self,
        path: str | PathLike[str],
        *,
        file_format: Literal["cif", "poscar"] | None = None,
        use_cif2cell: bool = False,
    ) -> Any:
        """Read a trusted local CIF or POSCAR/CONTCAR file into JARVIS Atoms.

        CIF loading preserves the supplied cell by disabling primitive-cell
        reduction. ``cif2cell`` is opt-in because it is not an ALIGNN runtime
        dependency.
        """

        atoms_class, _, _ = _jarvis_atoms_api()
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Structure file does not exist: {resolved}")

        resolved_format = file_format or self._infer_file_format(resolved)
        if resolved_format == "cif":
            return atoms_class.from_cif(
                filename=str(resolved),
                get_primitive_atoms=False,
                use_cif2cell=use_cif2cell,
            )
        if resolved_format == "poscar":
            return atoms_class.from_poscar(filename=str(resolved))
        raise ValueError(f"Unsupported structure file format: {resolved_format}")

    @staticmethod
    def _infer_file_format(path: Path) -> Literal["cif", "poscar"]:
        suffix = path.suffix.lower()
        name = path.name.upper()
        if suffix == ".cif":
            return "cif"
        if suffix in {".vasp", ".poscar"} or name.startswith("POSCAR") or name.startswith("CONTCAR"):
            return "poscar"
        raise ValueError(
            "Could not infer structure file format. Use file_format='cif' or file_format='poscar'."
        )

    @staticmethod
    def _from_mapping(structure: Mapping[str, Any], *, atoms_class: type[Any]) -> Any:
        required = ("lattice_mat", "coords", "elements")
        missing = [name for name in required if name not in structure]
        if missing:
            raise ValueError(f"Structure mapping is missing required keys: {missing}.")

        lattice = _as_float_array("lattice_mat", structure["lattice_mat"], shape=(3, 3))
        coords = _as_float_array("coords", structure["coords"])
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape [n_atoms, 3], got {coords.shape}.")

        raw_elements = structure["elements"]
        if isinstance(raw_elements, (str, bytes)) or not isinstance(raw_elements, Sequence):
            raise TypeError("elements must be a sequence of element symbols.")
        elements = [str(element) for element in raw_elements]
        if not elements or any(not element for element in elements):
            raise ValueError("elements must contain non-empty element symbols.")
        if len(elements) != coords.shape[0]:
            raise ValueError(
                "elements and coords must contain the same number of atoms: "
                f"{len(elements)} != {coords.shape[0]}."
            )

        cartesian = structure.get("cartesian", False)
        if not isinstance(cartesian, bool):
            raise TypeError("cartesian must be a bool when provided.")

        return atoms_class(
            lattice_mat=lattice.tolist(),
            coords=coords.tolist(),
            elements=elements,
            cartesian=cartesian,
        )


__all__ = ["StructureAdapter"]
