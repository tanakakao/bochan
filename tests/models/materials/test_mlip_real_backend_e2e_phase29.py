from __future__ import annotations

import os

import pytest

from bochan.models.regression.gaussian.materials.structure.factory import create_structure_relaxer


pytestmark = pytest.mark.slow


_STRUCTURE = {
    "lattice_mat": [
        [5.43, 0.0, 0.0],
        [0.0, 5.43, 0.0],
        [0.0, 0.0, 5.43],
    ],
    "coords": [
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ],
    "elements": ["Si", "Si"],
    "cartesian": False,
}


def test_real_backend_relaxation_e2e() -> None:
    backend = os.environ.get("BOCHAN_REAL_MLIP_BACKEND")
    if not backend:
        pytest.skip("Set BOCHAN_REAL_MLIP_BACKEND to run the real-backend E2E smoke.")

    relaxer = create_structure_relaxer(backend)
    result = relaxer.relax(
        _STRUCTURE,
        optimizer="FIRE",
        fmax=0.5,
        max_steps=1,
        relax_cell=False,
    )

    payload = result.as_dict()
    assert payload["backend"] in {"mace", "chgnet", "m3gnet", "alignn-ff"}
    assert isinstance(payload["energy"], float)
    assert len(payload["forces"]) == 2
    assert len(payload["stress"]) == 3
    assert all(len(row) == 3 for row in payload["stress"])
