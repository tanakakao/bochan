"""Canonical crystal-structure adapters for material graph integrations."""

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
_PYMATGEN_INSTALL_HINT = (
    "Pymatgen structure conversion requires pymatgen. "
    "Install bochan[materials], chgnet>=0.4.2,<0.5, matgl>=4.0.3,<5, "
    "or pymatgen directly."
)
_ASE_INSTALL_HINT = (
    "ASE structure conversion requires ase. "
    "Install bochan[materials], mace-torch>=0.3.16,<0.4, or ase directly."
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


def _pymatgen_structure_api() -> tuple[type[Any], type[Any]]:
    """Return pymatgen Structure and ASE adaptor classes lazily."""

    try:
        core_module = import_module("pymatgen.core")
        ase_module = import_module("pymatgen.io.ase")
    except ImportError as error:
        raise ImportError(_PYMATGEN_INSTALL_HINT) from error

    structure_class = getattr(core_module, "Structure", None)
    ase_adaptor_class = getattr(ase_module, "AseAtomsAdaptor", None)
    if not isinstance(structure_class, type):
        raise RuntimeError("pymatgen.core.Structure is unavailable.")
    if not isinstance(ase_adaptor_class, type):
        raise RuntimeError("pymatgen.io.ase.AseAtomsAdaptor is unavailable.")
    return structure_class, ase_adaptor_class


def _ase_atoms_api() -> type[Any]:
    """Return the ASE Atoms class lazily."""

    try:
        module = import_module("ase")
    except ImportError as error:
        raise ImportError(_ASE_INSTALL_HINT) from error
    atoms_class = getattr(module, "Atoms", None)
    if not isinstance(atoms_class, type):
        raise RuntimeError("ase.Atoms is unavailable.")
    return atoms_class


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


def _validate_periodic_ase(structure: Any) -> None:
    pbc = np.asarray(structure.get_pbc(), dtype=bool)
    if pbc.shape != (3,) or not bool(np.all(pbc)):
        raise ValueError("ASE structures must be periodic in all three directions for crystal graphs.")


def _validate_ordered_pymatgen(structure: Any) -> None:
    if not bool(getattr(structure, "is_ordered", True)):
        raise ValueError(
            "Disordered pymatgen structures are not supported; resolve occupancies first."
        )


class StructureAdapter:
    """Normalize common crystal structures for atomistic model backends.

    ``adapt`` preserves the existing JARVIS canonical representation used by
    ALIGNN. ``to_pymatgen`` provides the direct path used by CHGNet and M3GNet.
    ``to_ase`` provides the direct path used by MACE and future ASE-native
    atomistic backends without forcing a JARVIS or pymatgen round trip.

    Filesystem paths are never accepted by in-memory conversion methods. Local
    file access remains explicit through :meth:`from_file`.
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
                _validate_periodic_ase(structure)
                return ase_to_atoms(ase_atoms=structure)

        if type(structure).__module__.startswith("pymatgen."):
            _validate_ordered_pymatgen(structure)
            return pmg_to_atoms(pmg=structure)

        raise TypeError(
            "Unsupported structure type. Expected JARVIS Atoms, ASE Atoms, "
            "pymatgen Structure, or a mapping with lattice_mat/coords/elements."
        )

    def adapt_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Normalize a non-empty sequence to JARVIS ``Atoms`` objects."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")
        return tuple(self.adapt(structure) for structure in structures)

    def to_pymatgen(self, structure: Any) -> Any:
        """Return a pymatgen ``Structure`` without canonicalizing through JARVIS."""

        if isinstance(structure, (str, bytes, PathLike)):
            raise TypeError(
                "Filesystem paths are not accepted by StructureAdapter.to_pymatgen(); "
                "load the trusted local file explicitly first."
            )

        structure_class, ase_adaptor_class = _pymatgen_structure_api()
        if isinstance(structure, structure_class):
            _validate_ordered_pymatgen(structure)
            return structure

        if isinstance(structure, Mapping):
            lattice, coords, elements, cartesian = self._mapping_components(structure)
            return structure_class(
                lattice=lattice,
                species=elements,
                coords=coords,
                coords_are_cartesian=cartesian,
            )

        try:
            ase_module = import_module("ase")
        except ImportError:
            ase_module = None
        if ase_module is not None:
            ase_atoms_class = getattr(ase_module, "Atoms", None)
            if isinstance(ase_atoms_class, type) and isinstance(structure, ase_atoms_class):
                _validate_periodic_ase(structure)
                converted = ase_adaptor_class.get_structure(structure)
                _validate_ordered_pymatgen(converted)
                return converted

        if type(structure).__module__.startswith("jarvis."):
            converter = getattr(structure, "pymatgen_converter", None)
            if not callable(converter):
                raise TypeError("JARVIS structure does not expose pymatgen_converter().")
            converted = converter()
            if not isinstance(converted, structure_class):
                raise TypeError("JARVIS pymatgen_converter() did not return pymatgen Structure.")
            _validate_ordered_pymatgen(converted)
            return converted

        raise TypeError(
            "Unsupported structure type. Expected pymatgen Structure, JARVIS Atoms, "
            "ASE Atoms, or a mapping with lattice_mat/coords/elements."
        )

    def to_pymatgen_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Normalize a non-empty sequence directly to pymatgen structures."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")
        return tuple(self.to_pymatgen(structure) for structure in structures)

    def to_ase(self, structure: Any) -> Any:
        """Return a periodic ASE ``Atoms`` object without unnecessary round trips."""

        if isinstance(structure, (str, bytes, PathLike)):
            raise TypeError(
                "Filesystem paths are not accepted by StructureAdapter.to_ase(); "
                "load the trusted local file explicitly first."
            )

        atoms_class = _ase_atoms_api()
        if isinstance(structure, atoms_class):
            _validate_periodic_ase(structure)
            return structure

        if isinstance(structure, Mapping):
            lattice, coords, elements, cartesian = self._mapping_components(structure)
            kwargs: dict[str, Any] = {
                "symbols": elements,
                "cell": lattice,
                "pbc": True,
            }
            if cartesian:
                kwargs["positions"] = coords
            else:
                kwargs["scaled_positions"] = coords
            return atoms_class(**kwargs)

        if type(structure).__module__.startswith("pymatgen."):
            structure_class, ase_adaptor_class = _pymatgen_structure_api()
            if not isinstance(structure, structure_class):
                raise TypeError("Unsupported pymatgen structure object.")
            _validate_ordered_pymatgen(structure)
            converted = ase_adaptor_class.get_atoms(structure)
            _validate_periodic_ase(converted)
            return converted

        if type(structure).__module__.startswith("jarvis."):
            converter = getattr(structure, "ase_converter", None)
            if not callable(converter):
                raise TypeError("JARVIS structure does not expose ase_converter().")
            converted = converter()
            if not isinstance(converted, atoms_class):
                raise TypeError("JARVIS ase_converter() did not return ASE Atoms.")
            _validate_periodic_ase(converted)
            return converted

        raise TypeError(
            "Unsupported structure type. Expected ASE Atoms, pymatgen Structure, "
            "JARVIS Atoms, or a mapping with lattice_mat/coords/elements."
        )

    def to_ase_many(self, structures: Sequence[Any]) -> tuple[Any, ...]:
        """Normalize a non-empty sequence directly to periodic ASE structures."""

        if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
            raise TypeError("structures must be a non-empty sequence.")
        if not structures:
            raise ValueError("structures must contain at least one structure.")
        return tuple(self.to_ase(structure) for structure in structures)

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
    def _mapping_components(
        structure: Mapping[str, Any],
    ) -> tuple[list[list[float]], list[list[float]], list[str], bool]:
        required = ("lattice_mat", "coords", "elements")
        missing = [name for name in required if name not in structure]
        if missing:
            raise ValueError(f"Structure mapping is missing required keys: {missing}.")

        lattice = _as_float_array("lattice_mat", structure["lattice_mat"], shape=(3, 3))
        if abs(float(np.linalg.det(lattice))) <= 1e-12:
            raise ValueError("lattice_mat must be non-singular.")
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

        return lattice.tolist(), coords.tolist(), elements, cartesian

    @classmethod
    def _from_mapping(cls, structure: Mapping[str, Any], *, atoms_class: type[Any]) -> Any:
        lattice, coords, elements, cartesian = cls._mapping_components(structure)
        return atoms_class(
            lattice_mat=lattice,
            coords=coords,
            elements=elements,
            cartesian=cartesian,
        )


__all__ = ["StructureAdapter"]
