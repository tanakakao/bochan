from __future__ import annotations

import math

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from botorch.models import SingleTaskGP
from torch import nn

from bochan.api.configs import AcquisitionConfig, ModelBundle, ModelConfig
from bochan.composition import CHGNetEncoder
from bochan.models.regression.gaussian.materials.common import StructureRelaxationResult
from bochan.models.regression.gaussian.materials.structure import (
    CHGNetRelaxationAcquisitionSelector,
    CHGNetRelaxationRanker,
    CHGNetStructureRelaxer,
)


class FakeCHGNet(nn.Module):
    atom_fea_dim = 4
    mlp_first = True

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, graphs, *, task="e", return_crystal_feas=False):
        result = {"e": torch.zeros(len(graphs), device=self.weight.device)}
        if return_crystal_feas:
            result["crystal_fea"] = torch.zeros(len(graphs), 4, device=self.weight.device)
        return result


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        energy = float((positions**2).sum())
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -2.0 * positions,
            "stress": np.zeros(6),
        }


def _structure():
    return {
        "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "coords": [[0.10, 0.10, 0.10], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def test_chgnet_structure_relaxer_returns_common_result_contract() -> None:
    relaxer = CHGNetStructureRelaxer(
        encoder=CHGNetEncoder(encoder=FakeCHGNet()),
        calculator=HarmonicCalculator(),
    )
    result = relaxer.relax(_structure(), fmax=0.2, max_steps=20)

    assert result.backend == "chgnet"
    assert result.model_name == "0.3.0"
    assert len(result.forces) == 2
    assert len(result.stress) == 3
    assert all(len(row) == 3 for row in result.stress)
    assert result.energy <= result.initial_energy
    assert result.n_steps <= 20
    assert result.structure["elements"] == ["Si", "Si"]


class FakeRelaxer:
    def relax(self, structure, **kwargs):
        index = int(structure["index"])
        return StructureRelaxationResult(
            structure={"index": index, "relaxed": True},
            energy=float(index),
            initial_energy=float(index) + 1.0,
            forces=((0.0, 0.0, 0.0),),
            stress=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            max_force=0.0,
            n_steps=1,
            converged=True,
            optimizer=str(kwargs.get("optimizer", "FIRE")),
            fmax=float(kwargs.get("fmax", 0.05)),
            relax_cell=bool(kwargs.get("relax_cell", False)),
            backend="chgnet",
            model_name="fake",
        )


def _bundle_factory(structures):
    train_X = torch.arange(len(structures), dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([[0.2], [1.0], [0.6]], dtype=torch.double)[: len(structures)]
    model = SingleTaskGP(train_X, train_Y)
    return ModelBundle(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        model_config=ModelConfig(task_type="regression", model_type="base", outcome_transform=False),
        task_type="regression",
        model_type="base",
    )


def test_chgnet_ranker_reuses_generic_relaxation_ranking() -> None:
    model = SingleTaskGP(
        torch.arange(3, dtype=torch.double).unsqueeze(-1),
        torch.tensor([[3.0], [1.0], [2.0]], dtype=torch.double),
    )
    ranker = CHGNetRelaxationRanker(relaxer=FakeRelaxer())
    result = ranker.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        model_factory=lambda structures: model,
        direction="minimize",
    )

    assert len(result.candidates) == 3
    assert result.best.source_index in {0, 1, 2}
    assert result.best.relaxation.backend == "chgnet"


def test_chgnet_selector_reuses_bochan_active_learning() -> None:
    selector = CHGNetRelaxationAcquisitionSelector(relaxer=FakeRelaxer())
    result = selector.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        bundle_factory=_bundle_factory,
        acquisition_config=AcquisitionConfig(name="variance"),
        q=1,
    )

    assert result.acquisition_name == "variance"
    assert len(result.candidates) == 1
    assert result.best.relaxation.backend == "chgnet"
    assert math.isfinite(result.best.individual_acquisition_value)
