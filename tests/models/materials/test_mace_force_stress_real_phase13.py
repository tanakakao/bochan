from __future__ import annotations

import os

import pytest
import torch

_STRUCTURE = {
    "lattice_mat": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
    "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    "elements": ["Si", "Si"],
}


@pytest.mark.slow
def test_real_mace_force_and_stress_direct_baselines() -> None:
    if os.environ.get("BOCHAN_MACE_TENSOR_REAL") != "1":
        pytest.skip("Set BOCHAN_MACE_TENSOR_REAL=1 to run the real MACE tensor smoke.")
    pytest.importorskip("mace")

    from bochan.composition import MACEEncoder
    from bochan.models.regression.gaussian.materials.structure import (
        MACEDirectForcePredictor,
        MACEDirectStressPredictor,
    )

    encoder = MACEEncoder(model_name="medium-mpa-0")
    force = MACEDirectForcePredictor(encoder, [_STRUCTURE])
    stress = MACEDirectStressPredictor(encoder, [_STRUCTURE])
    X = torch.tensor([[0.0, 300.0]], dtype=torch.double)

    force_value = force(X)
    stress_value = stress(X)

    assert force_value.shape == (1, 6)
    assert stress_value.shape == (1, 9)
    assert torch.isfinite(force_value).all()
    assert torch.isfinite(stress_value).all()
