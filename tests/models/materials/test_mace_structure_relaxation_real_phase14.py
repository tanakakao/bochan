from __future__ import annotations

import pytest

from bochan.models.regression.gaussian.materials.structure import MACEStructureRelaxer

_STRUCTURE = {
    "lattice_mat": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
    "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    "elements": ["Si", "Si"],
}


@pytest.mark.slow
def test_real_mace_structure_relaxation_surface() -> None:
    pytest.importorskip("mace")
    relaxer = MACEStructureRelaxer(model_name="medium-mpa-0", device="cpu")

    result = relaxer.relax(
        _STRUCTURE,
        optimizer="FIRE",
        fmax=100.0,
        max_steps=1,
    )

    assert result.converged is True
    assert result.backend == "mace"
    assert result.model_name == "medium-mpa-0"
    assert len(result.forces) == 2
    assert len(result.stress) == 3
    assert result.max_force >= 0.0
