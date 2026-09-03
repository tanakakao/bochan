from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase.calculators.calculator import Calculator, all_changes

from bochan.models.regression.gaussian.materials.structure import (
    ALIGNNFFDirectEnergyPredictor,
    ALIGNNFFDirectForcePredictor,
    ALIGNNFFDirectStressPredictor,
    ALIGNNFFStructureRelaxer,
)


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        self.results["energy"] = float((positions**2).sum())
        self.results["forces"] = -2.0 * positions
        self.results["stress"] = np.zeros(6, dtype=float)


def _structure(scale: float = 1.0) -> dict[str, object]:
    return {
        "lattice_mat": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        "coords": [[0.1 * scale, 0.0, 0.0], [0.3 * scale, 0.0, 0.0]],
        "elements": ["Si", "Si"],
        "cartesian": True,
    }


def test_alignn_ff_direct_energy_force_stress_follow_common_layout() -> None:
    structures = [_structure(1.0), _structure(2.0)]
    calculator = HarmonicCalculator()
    energy = ALIGNNFFDirectEnergyPredictor(structures, calculator=calculator)
    force = ALIGNNFFDirectForcePredictor(structures, num_atoms=2, calculator=calculator)
    stress = ALIGNNFFDirectStressPredictor(structures, calculator=calculator)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    assert energy(X).shape == (2, 1)
    assert force(X).shape == (2, 6)
    assert stress(X).shape == (2, 9)
    assert force.num_atoms == 2
    torch.testing.assert_close(stress(X), torch.zeros(2, 9, dtype=torch.double))


def test_alignn_ff_structure_relaxer_uses_common_result_contract() -> None:
    relaxer = ALIGNNFFStructureRelaxer(calculator=HarmonicCalculator())
    result = relaxer.relax(_structure(), optimizer="FIRE", fmax=0.05, max_steps=50)

    assert result.backend == "alignn-ff"
    assert result.model_name == "alignnff_wt10"
    assert result.energy <= result.initial_energy
    assert len(result.forces) == 2
    assert len(result.stress) == 3
    assert result.n_steps <= 50
