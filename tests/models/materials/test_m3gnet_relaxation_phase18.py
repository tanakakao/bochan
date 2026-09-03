from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from bochan.models.regression.gaussian.materials.structure import (
    M3GNetRelaxationAcquisitionSelector,
    M3GNetRelaxationRanker,
    M3GNetStructureRelaxer,
    MaterialRelaxationAcquisitionSelector,
    MaterialRelaxationRanker,
)


class ZeroForceCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(atoms.cell.lengths()[0]),
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros((3, 3)),
        }


def _structure(scale: float = 5.4) -> dict[str, object]:
    return {
        "lattice_mat": [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def test_m3gnet_structure_relaxer_returns_common_contract() -> None:
    relaxer = M3GNetStructureRelaxer(calculator=ZeroForceCalculator())
    result = relaxer.relax(_structure(), fmax=0.05, max_steps=2)

    assert result.backend == "m3gnet"
    assert result.model_name == "M3GNet-PES-MatPES-PBE-2025.2"
    assert result.converged
    assert result.max_force == 0.0
    assert result.forces == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert result.stress == ((0.0, 0.0, 0.0),) * 3
    assert result.structure["elements"] == ["Si", "Si"]


def test_m3gnet_wrappers_reuse_phase16_generic_layer() -> None:
    relaxer = M3GNetStructureRelaxer(calculator=ZeroForceCalculator())
    ranker = M3GNetRelaxationRanker(relaxer=relaxer)
    selector = M3GNetRelaxationAcquisitionSelector(relaxer=relaxer)

    assert isinstance(ranker, MaterialRelaxationRanker)
    assert isinstance(selector, MaterialRelaxationAcquisitionSelector)
    assert ranker.relaxer is relaxer
    assert selector.relaxer is relaxer
