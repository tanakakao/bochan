from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from bochan.models.regression.gaussian.materials.structure import MACEStructureRelaxer


class HarmonicCalculator(Calculator):
    """Simple differentiable bowl used to test ASE relaxation without MACE downloads."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        self.results = {
            "energy": 0.5 * float(np.square(positions).sum()),
            "forces": -positions,
            "stress": np.zeros(6, dtype=float),
        }


_STRUCTURE = {
    "lattice_mat": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    "coords": [[0.4, 0.0, 0.0]],
    "elements": ["Si"],
    "cartesian": True,
}


def test_mace_relaxer_moves_structure_and_reports_convergence() -> None:
    relaxer = MACEStructureRelaxer(
        calculator=HarmonicCalculator(),
        model_name="test-harmonic",
    )

    result = relaxer.relax(
        _STRUCTURE,
        optimizer="FIRE",
        fmax=0.01,
        max_steps=200,
    )

    assert result.converged is True
    assert result.n_steps > 0
    assert result.energy < result.initial_energy
    assert result.max_force <= pytest.approx(0.01, abs=1e-6)
    assert result.backend == "mace"
    assert result.model_name == "test-harmonic"
    assert result.relax_cell is False
    assert result.structure["cartesian"] is False
    assert len(result.forces) == 1
    assert len(result.stress) == 3
    assert result.as_dict()["energy_change"] < 0.0


def test_mace_relaxer_validates_controls() -> None:
    relaxer = MACEStructureRelaxer(calculator=HarmonicCalculator(), model_name="test")

    with pytest.raises(ValueError, match="fmax"):
        relaxer.relax(_STRUCTURE, fmax=0.0)
    with pytest.raises(ValueError, match="max_steps"):
        relaxer.relax(_STRUCTURE, max_steps=0)
    with pytest.raises(ValueError, match="optimizer"):
        relaxer.relax(_STRUCTURE, optimizer="BAD")  # type: ignore[arg-type]
