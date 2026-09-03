"""Fast crystal-structure relaxation with pretrained CHGNet and ASE."""

from __future__ import annotations

from importlib import import_module
from math import sqrt
from typing import Any, Literal

from torch import nn

from bochan.composition import CHGNetEncoder
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.structure.adapter import StructureAdapter

from ..common.relaxation import StructureRelaxationResult
from .chgnet_residual import _resolve_encoder

_DEFAULT_MODEL_NAME = "0.3.0"
OptimizerName = Literal["FIRE", "BFGS", "LBFGS"]


def _load_chgnet_calculator(*, model: Any, device: str) -> Any:
    try:
        module = import_module("chgnet.model.dynamics")
    except ImportError as error:
        raise ImportError(
            "CHGNet structure relaxation requires chgnet>=0.4.2,<0.5. "
            "Install bochan[materials]."
        ) from error
    calculator_class = getattr(module, "CHGNetCalculator", None)
    if not isinstance(calculator_class, type):
        raise RuntimeError("The installed chgnet package does not expose CHGNetCalculator.")
    return calculator_class(model=model, use_device=device)


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
        raise ValueError("CHGNet forces must have shape [n_atoms, 3].")
    return rows


def _stress_rows(stress: Any) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    rows = tuple(tuple(float(value) for value in row) for row in stress.tolist())
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("CHGNet stress must have shape [3, 3].")
    return rows  # type: ignore[return-value]


def _max_force_norm(forces: tuple[tuple[float, float, float], ...]) -> float:
    if not forces:
        return 0.0
    return max(sqrt(sum(component * component for component in force)) for force in forces)


class CHGNetStructureRelaxer:
    """Relax periodic structures through CHGNet's official ASE calculator.

    CHGNet is evaluated through ``CHGNetCalculator`` while bochan owns the ASE
    optimizer loop so the result contract is identical to the MACE backend.
    Energies, forces and stress retain the ASE calculator's native conventions;
    bochan performs no implicit unit or sign conversion.
    """

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: str = "cpu",
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        strict_checkpoint: bool = True,
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
        self.material_encoder = _resolve_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        self.calculator = (
            _load_chgnet_calculator(model=self.material_encoder.encoder, device=device)
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
            backend="chgnet",
            model_name=self.model_name,
        )


def relax_structure_chgnet(
    structure: Any,
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    device: str = "cpu",
    optimizer: OptimizerName = "FIRE",
    fmax: float = 0.05,
    max_steps: int = 200,
    relax_cell: bool = False,
) -> StructureRelaxationResult:
    """Convenience wrapper for one-off CHGNet structure relaxation."""

    return CHGNetStructureRelaxer(model_name=model_name, device=device).relax(
        structure,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        relax_cell=relax_cell,
    )


__all__ = ["CHGNetStructureRelaxer", "OptimizerName", "relax_structure_chgnet"]
