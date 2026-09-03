from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.models.regression.gaussian.materials.common import (
    MaterialStructureRelaxer,
    StructureRelaxationResult,
    validate_structure_relaxer,
)
from bochan.models.regression.gaussian.materials.structure import (
    MACERelaxationRanker,
    MaterialRelaxationRanker,
)


class FakeCHGNetRelaxer:
    def relax(
        self,
        structure,
        *,
        optimizer="FIRE",
        fmax=0.05,
        max_steps=200,
        relax_cell=False,
    ):
        index = int(structure["index"])
        return StructureRelaxationResult(
            structure={"index": index, "relaxed": True},
            energy=float(index),
            initial_energy=float(index) + 0.5,
            forces=((0.0, 0.0, 0.0),),
            stress=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            max_force=0.0,
            n_steps=1,
            converged=True,
            optimizer=optimizer,
            fmax=float(fmax),
            relax_cell=relax_cell,
            backend="chgnet",
            model_name="fake-chgnet",
        )


class FakePosteriorModel:
    def posterior(self, X: torch.Tensor):
        mean = torch.tensor([[2.0], [0.5]], dtype=X.dtype, device=X.device)
        variance = torch.tensor([[0.04], [0.01]], dtype=X.dtype, device=X.device)
        return SimpleNamespace(mean=mean, variance=variance)


def test_material_relaxer_contract_is_backend_neutral() -> None:
    relaxer = FakeCHGNetRelaxer()

    assert isinstance(relaxer, MaterialStructureRelaxer)
    assert validate_structure_relaxer(relaxer) is relaxer


def test_generic_ranker_accepts_non_mace_backend() -> None:
    ranker = MaterialRelaxationRanker(relaxer=FakeCHGNetRelaxer())
    result = ranker.run(
        [{"index": 0}, {"index": 1}],
        model_factory=lambda structures: FakePosteriorModel(),
        direction="minimize",
    )

    assert result.best.source_index == 1
    assert result.best.relaxation.backend == "chgnet"
    assert all(candidate.relaxation.structure["relaxed"] for candidate in result.candidates)


def test_mace_ranker_remains_a_generic_ranker() -> None:
    ranker = MACERelaxationRanker(relaxer=FakeCHGNetRelaxer())

    assert isinstance(ranker, MaterialRelaxationRanker)
