"""Crystal-structure relaxation with MatGL M3GNet and ASE."""

from __future__ import annotations

from importlib import import_module
from math import sqrt
from typing import Any, Literal

from bochan.structure.adapter import StructureAdapter

from ..common.relaxation import StructureRelaxationResult
from .m3gnet_tensor_residual import _DEFAULT_MODEL_NAME, _load_calculator, _load_potential

OptimizerName = Literal["FIRE", "BFGS", "LBFGS"]


def _resolve_optimizer(name: OptimizerName) -> type[Any]:
    module = import_module("ase.optimize")
    optimizer = getattr(module, name, None)
    if not isinstance(optimizer, type):
        raise RuntimeError(f"ASE optimizer {name!r} is unavailable.")
    return optimizer


def _as_structure_mapping(atoms: Any) -> dict[str, Any]:
    return {
        "lattice_mat": atoms.cell.array.tolist(),
        "coords": atoms.get_scaled_positions(wrap=False).tolist(),
        "elements": atoms.get_chemical_symbols(),
        "cartesian": False,
    }


def _force_rows(forces: Any) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in forces.tolist())
    if any(len(row) != 3 for row in rows):
        raise ValueError("M3GNet forces must have shape [n_atoms, 3].")
    return rows


def _stress_rows(stress: Any) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    rows = tuple(tuple(float(value) for value in row) for row in stress.tolist())
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("M3GNet stress must have shape [3, 3].")
    return rows  # type: ignore[return-value]


def _max_force_norm(forces: tuple[tuple[float, float, float], ...]) -> float:
    if not forces:
        return 0.0
    return max(sqrt(sum(component * component for component in force)) for force in forces)


class M3GNetStructureRelaxer:
    """Relax periodic structures through MatGL's official ``PESCalculator``.

    bochan owns the ASE optimizer loop so M3GNet follows the same external
    relaxation contract as MACE and CHGNet. Energy, force and stress values are
    reported exactly in the ASE calculator convention without hidden conversion.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        potential: Any | None = None,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        self.model_name = model_name
        self.adapter = StructureAdapter() if adapter is None else adapter
        if not isinstance(self.adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")
        self.potential = _load_potential(model_name) if potential is None and calculator is None else potential
        self.calculator = _load_calculator(self.potential) if calculator is None else calculator

    def relax(
        self,
        structure: Any,
        *,
        optimizer: OptimizerName = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> StructureRelaxationResult:
        if optimizer not in {"FIRE", "BFGS", "LBFGS"}:
            raise ValueError("optimizer must be 'FIRE', 'BFGS', or 'LBFGS'.")
        if isinstance(fmax, bool) or not isinstance(fmax, (int, float)) or float(fmax) <= 0:
            raise ValueError("fmax must be a positive number.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if not isinstance(relax_cell, bool):
            raise TypeError("relax_cell must be a bool.")

        atoms = self.adapter.to_ase(structure).copy()
        atoms.calc = self.calculator
        initial_energy = float(atoms.get_potential_energy())

        target: Any = atoms
        if relax_cell:
            filters = import_module("ase.filters")
            filter_class = getattr(filters, "FrechetCellFilter", None)
            if not isinstance(filter_class, type):
                raise RuntimeError("ASE FrechetCellFilter is unavailable.")
            target = filter_class(atoms)

        optimizer_class = _resolve_optimizer(optimizer)
        dynamics = optimizer_class(target, logfile=None)
        converged = bool(dynamics.run(fmax=float(fmax), steps=max_steps))
        n_steps = int(getattr(dynamics, "nsteps", 0))

        energy = float(atoms.get_potential_energy())
        forces = _force_rows(atoms.get_forces())
        stress = _stress_rows(atoms.get_stress(voigt=False))

        return StructureRelaxationResult(
            structure=_as_structure_mapping(atoms),
            energy=energy,
            initial_energy=initial_energy,
            forces=forces,
            stress=stress,
            max_force=_max_force_norm(forces),
            n_steps=n_steps,
            converged=converged,
            optimizer=optimizer,
            fmax=float(fmax),
            relax_cell=relax_cell,
            backend="m3gnet",
            model_name=self.model_name,
        )


def relax_structure_m3gnet(
    structure: Any,
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    optimizer: OptimizerName = "FIRE",
    fmax: float = 0.05,
    max_steps: int = 200,
    relax_cell: bool = False,
) -> StructureRelaxationResult:
    """Convenience wrapper for one-off M3GNet structure relaxation."""
    return M3GNetStructureRelaxer(model_name=model_name).relax(
        structure,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        relax_cell=relax_cell,
    )


__all__ = ["M3GNetStructureRelaxer", "OptimizerName", "relax_structure_m3gnet"]
