"""Fast crystal-structure relaxation with pretrained MACE and ASE."""

from __future__ import annotations

from importlib import import_module
from math import sqrt
from typing import Any, Literal

from bochan.structure.adapter import StructureAdapter

from ..common.relaxation import StructureRelaxationResult

_DEFAULT_MODEL_NAME = "medium-mpa-0"
OptimizerName = Literal["FIRE", "BFGS", "LBFGS"]


def _load_mace_calculator(*, model_name: str, device: str) -> Any:
    """Load a pretrained MACE ASE calculator lazily."""

    try:
        module = import_module("mace.calculators.foundations_models")
    except ImportError as error:
        raise ImportError(
            "MACE structure relaxation requires mace-torch>=0.3.16,<0.4. "
            "Install bochan[mace] or bochan[materials]."
        ) from error
    loader = getattr(module, "mace_mp", None)
    if not callable(loader):
        raise RuntimeError("The installed mace-torch package does not expose mace_mp().")
    return loader(model=model_name, device=device)


def _resolve_optimizer(name: OptimizerName) -> type[Any]:
    module = import_module("ase.optimize")
    optimizer = getattr(module, name, None)
    if not isinstance(optimizer, type):
        raise RuntimeError(f"ASE optimizer {name!r} is unavailable.")
    return optimizer


def _as_structure_mapping(atoms: Any) -> dict[str, Any]:
    """Convert ASE Atoms to bochan's canonical periodic structure mapping."""

    return {
        "lattice_mat": atoms.cell.array.tolist(),
        "coords": atoms.get_scaled_positions(wrap=False).tolist(),
        "elements": atoms.get_chemical_symbols(),
        "cartesian": False,
    }


def _force_rows(forces: Any) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in forces.tolist())
    if any(len(row) != 3 for row in rows):
        raise ValueError("MACE forces must have shape [n_atoms, 3].")
    return rows


def _stress_rows(stress: Any) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    rows = tuple(tuple(float(value) for value in row) for row in stress.tolist())
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("MACE stress must have shape [3, 3].")
    return rows  # type: ignore[return-value]


def _max_force_norm(forces: tuple[tuple[float, float, float], ...]) -> float:
    if not forces:
        return 0.0
    return max(sqrt(sum(component * component for component in force)) for force in forces)


class MACEStructureRelaxer:
    """Relax periodic structures with a pretrained MACE ASE calculator.

    Atomic positions are relaxed by default. Set ``relax_cell=True`` to also
    optimize lattice degrees of freedom through ASE's ``FrechetCellFilter``.
    The returned energy, forces, and stress retain the MACE calculator's native
    conventions; bochan performs no implicit unit, stress-sign, or aggregation
    conversion.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: str = "cpu",
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string.")
        self.model_name = model_name
        self.device = device
        self.adapter = StructureAdapter() if adapter is None else adapter
        if not isinstance(self.adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")
        self.calculator = (
            _load_mace_calculator(model_name=model_name, device=device)
            if calculator is None
            else calculator
        )

    def relax(
        self,
        structure: Any,
        *,
        optimizer: OptimizerName = "FIRE",
        fmax: float = 0.05,
        max_steps: int = 200,
        relax_cell: bool = False,
    ) -> StructureRelaxationResult:
        """Relax one periodic structure and return diagnostics and final observables.

        Args:
            structure: Any in-memory structure accepted by ``StructureAdapter.to_ase``.
            optimizer: ASE optimizer. ``FIRE`` is the default robust choice.
            fmax: Convergence threshold for the maximum force component used by ASE.
            max_steps: Maximum number of optimization steps.
            relax_cell: Also optimize cell degrees of freedom when ``True``.

        Returns:
            Serializable relaxation result with relaxed structure and final
            energy/force/stress diagnostics.
        """

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
            backend="mace",
            model_name=self.model_name,
        )


def relax_structure_mace(
    structure: Any,
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    device: str = "cpu",
    optimizer: OptimizerName = "FIRE",
    fmax: float = 0.05,
    max_steps: int = 200,
    relax_cell: bool = False,
) -> StructureRelaxationResult:
    """Convenience wrapper for one-off MACE structure relaxation."""

    return MACEStructureRelaxer(model_name=model_name, device=device).relax(
        structure,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        relax_cell=relax_cell,
    )


__all__ = ["MACEStructureRelaxer", "OptimizerName", "relax_structure_mace"]
