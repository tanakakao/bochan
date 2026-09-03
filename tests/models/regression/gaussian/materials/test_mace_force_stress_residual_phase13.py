from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.models import SingleTaskGP
from torch import Tensor, nn

from bochan.composition import MACEEncoder
from bochan.models.regression.gaussian.materials.common.baseline import MaterialPropertyContract
from bochan.models.regression.gaussian.materials.common.tensor_target import TensorTargetLayout
from bochan.models.regression.gaussian.materials.structure import mace_tensor_residual as module


class _RawMACE(nn.Module):
    def forward(
        self,
        batch: dict[str, Any],
        *,
        compute_force: bool,
        compute_virials: bool,
        compute_stress: bool,
    ) -> dict[str, Tensor]:
        index = float(batch["id"])
        output: dict[str, Tensor] = {}
        if compute_force:
            output["forces"] = torch.tensor(
                [[index + 0.1, 0.2, 0.3], [0.4, index + 0.5, 0.6]],
                dtype=torch.double,
            )
        if compute_stress:
            output["stress"] = torch.eye(3, dtype=torch.double).unsqueeze(0) * (index + 1.0)
        if compute_virials:
            output["virials"] = torch.zeros(1, 3, 3, dtype=torch.double)
        return output


class _FakeMACEEncoder(MACEEncoder):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.encoder = _RawMACE()
        self._head = "Default"

    @property
    def head(self) -> str:
        return self._head

    @property
    def output_dim(self) -> int:
        return 2

    def _build_batch(self, structure: Any) -> dict[str, Any]:
        return dict(structure)

    def forward(self, structures: Any) -> Tensor:
        count = len(structures) if isinstance(structures, (list, tuple)) else 1
        return torch.ones(count, 2, dtype=torch.double)


class _FakeWideGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(train_X, train_Y, train_Yvar=train_Yvar)


_STRUCTURES = (
    {"id": 0, "elements": ["Si", "Si"]},
    {"id": 1, "elements": ["Si", "Si"]},
)


def test_tensor_target_layout_roundtrip() -> None:
    force = TensorTargetLayout.force(2)
    values = torch.arange(12, dtype=torch.double).reshape(2, 2, 3)
    flat = force.flatten(values, n=2)
    assert flat.shape == (2, 6)
    torch.testing.assert_close(force.unflatten(flat), values)

    stress = TensorTargetLayout.stress()
    tensor = torch.arange(18, dtype=torch.double).reshape(2, 3, 3)
    stress_flat = stress.flatten(tensor, n=2)
    assert stress_flat.shape == (2, 9)
    torch.testing.assert_close(stress.unflatten(stress_flat), tensor)


def test_force_predictor_is_fixed_topology_and_process_independent() -> None:
    predictor = module.MACEDirectForcePredictor(_FakeMACEEncoder(), _STRUCTURES)
    X = torch.tensor([[0.0, 300.0], [0.0, 900.0], [1.0, 500.0]], dtype=torch.double)
    baseline = predictor(X)
    assert baseline.shape == (3, 6)
    torch.testing.assert_close(baseline[0], baseline[1])
    assert predictor.num_atoms == 2
    assert predictor.layout.tensor_shape == (2, 3)

    ragged = (
        {"id": 0, "elements": ["Si"]},
        {"id": 1, "elements": ["Si", "Si"]},
    )
    with pytest.raises(ValueError, match="fixed topology"):
        module.MACEDirectForcePredictor(_FakeMACEEncoder(), ragged)


def test_force_residual_model_flattens_physical_targets(monkeypatch) -> None:
    monkeypatch.setattr(module, "MACEMultiTaskGPModel", _FakeWideGP)
    X = torch.tensor([[0.0, 300.0], [1.0, 600.0]], dtype=torch.double)
    predictor = module.MACEDirectForcePredictor(_FakeMACEEncoder(), _STRUCTURES)
    baseline = predictor(X).reshape(2, 2, 3)
    Y = baseline + 0.05

    model = module.MACEForceResidualGPModel(
        X,
        Y,
        structures=_STRUCTURES,
        encoder=_FakeMACEEncoder(),
        target_contract=MaterialPropertyContract("force", "eV/A", "unspecified"),
    )
    assert model.num_outputs == 6
    assert model.layout.output_dim == 6
    assert model.baseline_metadata is not None
    assert model.baseline_metadata["property"]["quantity"] == "force"
    posterior = model.posterior(X)
    assert posterior.mean.shape == (2, 6)
    assert model.unflatten(posterior.mean).shape == (2, 2, 3)


def test_stress_residual_model_preserves_full_tensor(monkeypatch) -> None:
    monkeypatch.setattr(module, "MACEMultiTaskGPModel", _FakeWideGP)
    X = torch.tensor([[0.0, 300.0], [1.0, 600.0]], dtype=torch.double)
    predictor = module.MACEDirectStressPredictor(_FakeMACEEncoder(), _STRUCTURES)
    Y = predictor(X).reshape(2, 3, 3) + 0.01

    model = module.MACEStressResidualGPModel(
        X,
        Y,
        structures=_STRUCTURES,
        encoder=_FakeMACEEncoder(),
        target_contract=MaterialPropertyContract("stress", "eV/A^3", "intensive"),
    )
    assert model.num_outputs == 9
    assert model.layout.tensor_shape == (3, 3)
    posterior = model.posterior(X)
    assert posterior.mean.shape == (2, 9)
    assert model.unflatten(posterior.mean).shape == (2, 3, 3)


def test_force_contract_rejects_wrong_quantity(monkeypatch) -> None:
    monkeypatch.setattr(module, "MACEMultiTaskGPModel", _FakeWideGP)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    Y = torch.zeros(2, 2, 3, dtype=torch.double)
    with pytest.raises(ValueError, match="quantity='force'"):
        module.MACEForceResidualGPModel(
            X,
            Y,
            structures=_STRUCTURES,
            encoder=_FakeMACEEncoder(),
            target_contract=MaterialPropertyContract("energy", "eV", "total"),
        )
