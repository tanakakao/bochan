from __future__ import annotations

import torch
from torch import nn

from bochan.composition import CHGNetEncoder
from bochan.models.regression.gaussian.materials.structure import (
    CHGNetDirectForcePredictor,
    CHGNetDirectStressPredictor,
    CHGNetForceResidualGPModel,
    CHGNetStressResidualGPModel,
)


class CrystalGraph:
    __module__ = "chgnet.graph.crystalgraph"

    def __init__(self, index: int) -> None:
        self.index = index
        self.atom_frac_coord = torch.zeros(2, 3)
        self.neighbor_image = torch.zeros(1, 3)
        self.lattice = torch.eye(3)

    def to(self, device: str):
        self.atom_frac_coord = self.atom_frac_coord.to(device)
        self.neighbor_image = self.neighbor_image.to(device)
        self.lattice = self.lattice.to(device)
        return self

    def __len__(self) -> int:
        return 2


class FakeCHGNet(nn.Module):
    atom_fea_dim = 4
    mlp_first = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, graphs, *, task="e", return_crystal_feas=False):
        crystal = torch.stack(
            [torch.arange(4, dtype=self.weight.dtype, device=self.weight.device) + graph.index for graph in graphs]
        )
        result = {"e": torch.tensor([float(graph.index) for graph in graphs], device=self.weight.device)}
        if "f" in task:
            result["f"] = [
                torch.full((2, 3), float(graph.index + 1), device=self.weight.device)
                for graph in graphs
            ]
        if "s" in task:
            result["s"] = [
                torch.eye(3, device=self.weight.device) * float(graph.index + 2)
                for graph in graphs
            ]
        if return_crystal_feas:
            result["crystal_fea"] = crystal
        return result


def _encoder() -> CHGNetEncoder:
    return CHGNetEncoder(encoder=FakeCHGNet())


def test_chgnet_direct_force_and_stress_predictors_follow_tensor_layout() -> None:
    structures = [CrystalGraph(0), CrystalGraph(1)]
    encoder = _encoder()
    force = CHGNetDirectForcePredictor(encoder, structures, num_atoms=2)
    stress = CHGNetDirectStressPredictor(encoder, structures)
    X = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    force_y = force(X)
    stress_y = stress(X)

    assert force_y.shape == (2, 6)
    assert stress_y.shape == (2, 9)
    torch.testing.assert_close(force_y[0], torch.ones(6, dtype=torch.double))
    torch.testing.assert_close(force_y[1], torch.full((6,), 2.0, dtype=torch.double))
    expected_stress = torch.eye(3, dtype=torch.double).reshape(-1) * 3.0
    torch.testing.assert_close(stress_y[1], expected_stress)


def test_chgnet_force_residual_gp_restores_force_axes() -> None:
    structures = [CrystalGraph(0), CrystalGraph(1), CrystalGraph(2)]
    train_X = torch.arange(3, dtype=torch.double).unsqueeze(-1)
    baseline = torch.stack(
        [torch.full((2, 3), float(index + 1), dtype=torch.double) for index in range(3)]
    )
    train_Y = baseline + 0.1

    model = CHGNetForceResidualGPModel(
        train_X,
        train_Y,
        structures=structures,
        num_atoms=2,
        encoder=_encoder(),
        latent_dim=4,
        outcome_transform=None,
    )
    posterior = model.posterior(train_X[:1])

    assert posterior.mean.shape == (1, 6)
    assert model.unflatten(posterior.mean).shape == (1, 2, 3)
    assert model.num_atoms == 2


def test_chgnet_stress_residual_gp_restores_tensor_axes() -> None:
    structures = [CrystalGraph(0), CrystalGraph(1), CrystalGraph(2)]
    train_X = torch.arange(3, dtype=torch.double).unsqueeze(-1)
    baseline = torch.stack(
        [torch.eye(3, dtype=torch.double) * float(index + 2) for index in range(3)]
    )
    train_Y = baseline + 0.05

    model = CHGNetStressResidualGPModel(
        train_X,
        train_Y,
        structures=structures,
        encoder=_encoder(),
        latent_dim=4,
        outcome_transform=None,
    )
    posterior = model.posterior(train_X[:1])

    assert posterior.mean.shape == (1, 9)
    assert model.unflatten(posterior.mean).shape == (1, 3, 3)
