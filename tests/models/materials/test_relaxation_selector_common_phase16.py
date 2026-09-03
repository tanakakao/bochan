from __future__ import annotations

import torch
from botorch.models import SingleTaskGP

from bochan.api.configs import AcquisitionConfig, ModelBundle, ModelConfig
from bochan.models.regression.gaussian.materials.common import StructureRelaxationResult
from bochan.models.regression.gaussian.materials.structure import (
    MACERelaxationAcquisitionSelector,
    MaterialRelaxationAcquisitionSelector,
)


class FakeM3GNetRelaxer:
    def relax(self, structure, **kwargs):
        index = int(structure["index"])
        return StructureRelaxationResult(
            structure={"index": index, "relaxed": True},
            energy=float(index),
            initial_energy=float(index) + 0.25,
            forces=((0.0, 0.0, 0.0),),
            stress=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            max_force=0.0,
            n_steps=1,
            converged=True,
            optimizer=str(kwargs.get("optimizer", "FIRE")),
            fmax=float(kwargs.get("fmax", 0.05)),
            relax_cell=bool(kwargs.get("relax_cell", False)),
            backend="m3gnet",
            model_name="fake-m3gnet",
        )


def _bundle_factory(structures):
    train_X = torch.arange(len(structures), dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor([[0.2], [1.0], [0.6]], dtype=torch.double)[: len(structures)]
    return ModelBundle(
        model=SingleTaskGP(train_X, train_Y),
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


def test_generic_selector_accepts_non_mace_relaxer() -> None:
    selector = MaterialRelaxationAcquisitionSelector(relaxer=FakeM3GNetRelaxer())
    result = selector.run(
        [{"index": 0}, {"index": 1}, {"index": 2}],
        bundle_factory=_bundle_factory,
        acquisition_config=AcquisitionConfig(name="variance"),
        q=1,
    )

    assert result.best.relaxation.backend == "m3gnet"
    assert result.best.relaxation.structure["relaxed"] is True


def test_mace_selector_remains_a_generic_selector() -> None:
    selector = MACERelaxationAcquisitionSelector(relaxer=FakeM3GNetRelaxer())

    assert isinstance(selector, MaterialRelaxationAcquisitionSelector)
