from __future__ import annotations

import os

import pytest
import torch

from bochan.models.regression.gaussian.materials.common import validate_residual_production_model

_STRUCTURE = {
    "lattice_mat": [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
    "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    "elements": ["Si", "Si"],
}


def _backend_model_class(backend: str):
    if backend == "chgnet":
        pytest.importorskip("chgnet")
        from bochan.models.regression.gaussian.materials.structure import CHGNetResidualGPModel

        return CHGNetResidualGPModel
    if backend == "m3gnet":
        pytest.importorskip("matgl")
        from bochan.models.regression.gaussian.materials.structure import M3GNetResidualGPModel

        return M3GNetResidualGPModel
    if backend == "mace":
        pytest.importorskip("mace")
        from bochan.models.regression.gaussian.materials.structure import MACEResidualGPModel

        return MACEResidualGPModel
    raise ValueError(f"Unknown backend {backend!r}.")


@pytest.mark.slow
def test_real_material_residual_backend_fit_posterior_contract() -> None:
    backend = os.environ.get("BOCHAN_MATERIAL_BACKEND")
    if backend not in {"chgnet", "m3gnet", "mace"}:
        pytest.skip("Set BOCHAN_MATERIAL_BACKEND to chgnet, m3gnet, or mace.")

    model_cls = _backend_model_class(backend)
    train_X = torch.tensor(
        [[0.0, 300.0], [0.0, 500.0], [0.0, 700.0]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[-1.0], [-0.8], [-0.6]], dtype=torch.double)
    model = model_cls(
        train_X=train_X,
        train_Y=train_Y,
        structures=[_STRUCTURE],
    )

    query = torch.tensor([[0.0, 400.0], [0.0, 600.0]], dtype=torch.double)
    report = validate_residual_production_model(
        model,
        query,
        expected_num_outputs=1,
    )

    assert report.baseline_output_indices == (0,)
    assert report.posterior_shape == (2, 1)
