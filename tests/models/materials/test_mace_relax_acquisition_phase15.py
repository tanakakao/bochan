from __future__ import annotations

import math

import torch
from botorch.models import SingleTaskGP

from bochan.api.configs import AcquisitionConfig, ModelBundle, ModelConfig, ObjectiveConfig
from bochan.models.regression.gaussian.materials.common import StructureRelaxationResult
from bochan.models.regression.gaussian.materials.structure import MACERelaxationAcquisitionSelector


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


def _bundle_factory(structures):
    train_X = torch.arange(len(structures), dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([[0.2], [1.0], [0.6]], dtype=torch.double)[: len(structures)]
    model = SingleTaskGP(train_X, train_Y)
    return ModelBundle(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        task_type="regression",
        model_type="base",
    )


def test_relaxed_candidates_can_be_selected_with_bochan_qucb() -> None:
    selector = MACERelaxationAcquisitionSelector(relaxer=FakeRelaxer())
    result = selector.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        bundle_factory=_bundle_factory,
        acquisition_config=AcquisitionConfig(name="qucb"),
        q=2,
    )

    assert result.acquisition_name == "qucb"
    assert result.q == 2
    assert len(result.candidates) == 2
    assert len({candidate.source_index for candidate in result.candidates}) == 2
    assert all(candidate.relaxation.structure["relaxed"] is True for candidate in result.candidates)
    assert all(math.isfinite(candidate.individual_acquisition_value) for candidate in result.candidates)
    assert result.acquisition_value
    assert all(math.isfinite(value) for value in result.acquisition_value)


def test_relaxed_candidates_can_be_selected_with_minimizing_qei() -> None:
    selector = MACERelaxationAcquisitionSelector(relaxer=FakeRelaxer())
    result = selector.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        bundle_factory=_bundle_factory,
        acquisition_config=AcquisitionConfig(
            name="qei",
            objective_config=ObjectiveConfig(direction="minimize"),
        ),
        q=1,
    )

    assert len(result.candidates) == 1
    assert result.best.source_index in {0, 1, 2}
    assert math.isfinite(result.best.individual_acquisition_value)


def test_relaxed_acquisition_selection_preserves_process_columns() -> None:
    seen = {}

    def bundle_factory(structures):
        train_X = torch.tensor(
            [[0.0, 300.0], [1.0, 500.0], [2.0, 700.0]],
            dtype=torch.double,
        )
        train_Y = torch.tensor([[0.2], [1.0], [0.6]], dtype=torch.double)
        model = SingleTaskGP(train_X, train_Y)
        seen["structures"] = structures
        return ModelBundle(
            model=model,
            train_X=train_X,
            train_Y=train_Y,
            model_config=ModelConfig(
                task_type="regression",
                model_type="base",
                outcome_transform=False,
            ),
            task_type="regression",
            model_type="base",
        )

    selector = MACERelaxationAcquisitionSelector(relaxer=FakeRelaxer())
    result = selector.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        bundle_factory=bundle_factory,
        acquisition_config=AcquisitionConfig(name="qucb"),
        process_X=torch.tensor([[300.0], [500.0], [700.0]], dtype=torch.double),
    )

    assert tuple(item["index"] for item in seen["structures"]) == (0, 1, 2)
    assert len(result.candidates) == 1
