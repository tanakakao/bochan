"""Common contracts for MLIP-backed crystal-structure relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StructureRelaxationResult:
    """Serializable result returned by a material structure relaxer.

    Attributes:
        structure: Relaxed periodic structure in bochan's canonical mapping form.
        energy: Final potential energy in the backend calculator's native unit.
        initial_energy: Potential energy before relaxation.
        forces: Final per-atom Cartesian forces with shape ``[n_atoms, 3]``.
        stress: Final full Cartesian stress tensor with shape ``[3, 3]``.
        max_force: Maximum Euclidean force norm over atoms.
        n_steps: Number of optimizer steps executed.
        converged: Whether the optimizer satisfied the requested force threshold.
        optimizer: ASE optimizer name.
        fmax: Requested maximum-force convergence threshold.
        relax_cell: Whether lattice degrees of freedom were relaxed.
        backend: Material potential backend identifier.
        model_name: Pretrained model identifier used by the backend.
    """

    structure: dict[str, Any]
    energy: float
    initial_energy: float
    forces: tuple[tuple[float, float, float], ...]
    stress: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    max_force: float
    n_steps: int
    converged: bool
    optimizer: str
    fmax: float
    relax_cell: bool
    backend: str
    model_name: str

    @property
    def energy_change(self) -> float:
        """Return final minus initial potential energy."""

        return self.energy - self.initial_energy

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "structure": self.structure,
            "energy": self.energy,
            "initial_energy": self.initial_energy,
            "energy_change": self.energy_change,
            "forces": [list(row) for row in self.forces],
            "stress": [list(row) for row in self.stress],
            "max_force": self.max_force,
            "n_steps": self.n_steps,
            "converged": self.converged,
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "relax_cell": self.relax_cell,
            "backend": self.backend,
            "model_name": self.model_name,
        }


__all__ = ["StructureRelaxationResult"]
