from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder
from bochan.models.regression.gaussian.materials.structure import (
    M3GNetDirectForcePredictor,
    M3GNetDirectStressPredictor,
    M3GNetForceResidualGPModel,
    M3GNetStressResidualGPModel,
)


class FakeGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((1, 3), dtype=torch.float32)
        self.pbc_offshift = torch.zeros((1, 3), dtype=torch.float32)
        self.pos = self.frac_coords.clone()

    def to(self, device: Any):
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        self.pos = self.pos.to(device)
        return self


class FakeGraphConverter:
    def get_graph(self, structure: Any) -> tuple[FakeGraph, Tensor, None]:
        lattice = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
        return FakeGraph(structure), lattice, None


class FakeM3GNet(nn.Module):
    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.is_intensive = True
        self.include_state = False
        self.embedding = nn.Linear(3, output_dim)
        self.graph_layers = nn.ModuleList([nn.Linear(output_dim, output_dim)])
        self.final_layer = nn.Linear(output_dim, 1)
        self.feature_dict: dict[str, Tensor] = {}

    def forward(self, g: FakeGraph, state_attr: Tensor | None = None) -> Tensor:
        features = torch.tanh(self.embedding(g.frac_coords.mean(dim=0)))
        features = features + torch.tanh(self.graph_layers[0](features))
        self.feature_dict = {"readout": features.unsqueeze(0)}
        return self.final_layer(features).squeeze(-1)


class FakePESCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        scale = float(atoms.cell.lengths()[0])
        n_atoms = len(atoms)
        self.results = {
            "energy": scale,
            "forces": np.full((n_atoms, 3), scale / 5.0),
            "stress": np.eye(3) * (scale / 10.0),
        }


def _structure(scale: float) -> dict[str, object]:
    return {
        "lattice_mat": [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def _structures() -> tuple[dict[str, object], ...]:
    return (_structure(5.0), _structure(5.5), _structure(6.0))


def _encoder() -> M3GNetEncoder:
    return M3GNetEncoder(encoder=FakeM3GNet(), graph_converter=FakeGraphConverter())


def test_m3gnet_direct_force_and_stress_follow_tensor_layout() -> None:
    structures = _structures()
    calculator = FakePESCalculator()
    force = M3GNetDirectForcePredictor(structures, num_atoms=2, calculator=calculator)
    stress = M3GNetDirectStressPredictor(structures, calculator=calculator)
    X = torch.tensor([[0.0], [2.0]], dtype=torch.double)

    force_y = force(X)
    stress_y = stress(X)

    assert force_y.shape == (2, 6)
    assert stress_y.shape == (2, 9)
    torch.testing.assert_close(force_y[0], torch.ones(6, dtype=torch.double))
    torch.testing.assert_close(force_y[1], torch.full((6,), 1.2, dtype=torch.double))
    expected_stress = torch.eye(3, dtype=torch.double).reshape(-1) * 0.6
    torch.testing.assert_close(stress_y[1], expected_stress)


def test_m3gnet_force_residual_gp_restores_force_axes() -> None:
    structures = _structures()
    train_X = torch.arange(3, dtype=torch.double).unsqueeze(-1)
    baseline = torch.stack(
        [torch.full((2, 3), scale / 5.0, dtype=torch.double) for scale in (5.0, 5.5, 6.0)]
    )
    model = M3GNetForceResidualGPModel(
        train_X,
        baseline + 0.1,
        structures=structures,
        num_atoms=2,
        encoder=_encoder(),
        calculator=FakePESCalculator(),
        latent_dim=4,
        outcome_transform=None,
    )

    posterior = model.posterior(train_X[:1])
    assert posterior.mean.shape == (1, 6)
    assert model.unflatten(posterior.mean).shape == (1, 2, 3)
    assert model.num_atoms == 2


def test_m3gnet_stress_residual_gp_restores_tensor_axes() -> None:
    structures = _structures()
    train_X = torch.arange(3, dtype=torch.double).unsqueeze(-1)
    baseline = torch.stack(
        [torch.eye(3, dtype=torch.double) * (scale / 10.0) for scale in (5.0, 5.5, 6.0)]
    )
    model = M3GNetStressResidualGPModel(
        train_X,
        baseline + 0.05,
        structures=structures,
        encoder=_encoder(),
        calculator=FakePESCalculator(),
        latent_dim=4,
        outcome_transform=None,
    )

    posterior = model.posterior(train_X[:1])
    assert posterior.mean.shape == (1, 9)
    assert model.unflatten(posterior.mean).shape == (1, 3, 3)
