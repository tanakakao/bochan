from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.models import SingleTaskGP

from bochan.models.regression.gaussian.materials.common import StructureRelaxationResult
from bochan.models.regression.gaussian.materials.structure import MACERelaxationRanker


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
            backend="mace",
            model_name="fake",
        )


class FakePosteriorModel:
    def __init__(self, means: torch.Tensor, variances: torch.Tensor) -> None:
        self.means = means
        self.variances = variances
        self.last_X = None

    def posterior(self, X: torch.Tensor):
        self.last_X = X.clone()
        return SimpleNamespace(mean=self.means.to(X), variance=self.variances.to(X))


def test_relax_and_rank_rebuilds_relaxed_structure_bank() -> None:
    seen = {}
    model = FakePosteriorModel(
        means=torch.tensor([[3.0], [1.0], [2.0]], dtype=torch.double),
        variances=torch.tensor([[0.01], [0.04], [0.09]], dtype=torch.double),
    )

    def factory(structures):
        seen["structures"] = structures
        return model

    ranker = MACERelaxationRanker(relaxer=FakeRelaxer())
    result = ranker.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        model_factory=factory,
        process_X=torch.tensor([[300.0], [500.0], [700.0]], dtype=torch.double),
        direction="minimize",
    )

    assert tuple(item["index"] for item in seen["structures"]) == (0, 1, 2)
    assert all(item["relaxed"] is True for item in seen["structures"])
    assert model.last_X is not None
    torch.testing.assert_close(
        model.last_X,
        torch.tensor([[0.0, 300.0], [1.0, 500.0], [2.0, 700.0]], dtype=torch.double),
    )
    assert [candidate.source_index for candidate in result.candidates] == [1, 2, 0]
    assert result.best.posterior_mean == pytest.approx(1.0)
    assert result.best.relaxation.structure["relaxed"] is True


def test_ucb_ranking_can_prioritize_uncertain_candidate() -> None:
    model = FakePosteriorModel(
        means=torch.tensor([[1.0], [1.1]], dtype=torch.double),
        variances=torch.tensor([[0.01], [1.0]], dtype=torch.double),
    )
    ranker = MACERelaxationRanker(relaxer=FakeRelaxer())
    result = ranker.run(
        [{"index": 0}, {"index": 1}],
        model_factory=lambda structures: model,
        direction="minimize",
        criterion="ucb",
        beta=2.0,
    )

    assert result.best.source_index == 1
    assert result.best.posterior_std == pytest.approx(1.0)


def test_relax_and_rank_accepts_real_botorch_scalar_model() -> None:
    def factory(structures):
        train_X = torch.arange(len(structures), dtype=torch.double).unsqueeze(-1)
        train_Y = torch.tensor([[2.0], [0.5], [1.5]], dtype=torch.double)
        return SingleTaskGP(train_X, train_Y)

    ranker = MACERelaxationRanker(relaxer=FakeRelaxer())
    result = ranker.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        model_factory=factory,
        direction="minimize",
    )

    assert len(result.candidates) == 3
    assert {candidate.source_index for candidate in result.candidates} == {0, 1, 2}
    assert all(candidate.posterior_std >= 0.0 for candidate in result.candidates)


def test_relax_and_rank_rejects_stale_or_invalid_shapes() -> None:
    ranker = MACERelaxationRanker(relaxer=FakeRelaxer())
    model = FakePosteriorModel(
        means=torch.zeros(2, 2, dtype=torch.double),
        variances=torch.ones(2, 2, dtype=torch.double),
    )

    with pytest.raises(ValueError, match="scalar posterior"):
        ranker.run(
            [{"index": 0}, {"index": 1}],
            model_factory=lambda structures: model,
        )

    with pytest.raises(ValueError, match="process_X"):
        ranker.run(
            [{"index": 0}, {"index": 1}],
            model_factory=lambda structures: model,
            process_X=torch.zeros(3, 1, dtype=torch.double),
        )
