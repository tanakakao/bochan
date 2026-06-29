from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.acquisition.objective import (
    make_outcome_constraint,
    make_outcome_constraints,
)
from bochan.api import AcquisitionConfig, ModelBundle, ModelConfig, build_acquisition


class _AcquisitionWithConstraints:
    def __init__(self, model, constraints=None, objective=None) -> None:
        self.model = model
        self.constraints = constraints
        self.objective = objective


def _make_bundle() -> ModelBundle:
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=None,
        train_Y=None,
        model_config=ModelConfig(
            task_type="regression",
            model_type="base",
            outcome_transform=False,
        ),
        task_type="regression",
        model_type="base",
    )


def test_acquisition_config_forwards_single_constraint() -> None:
    constraints = [make_outcome_constraint(1, "ge", 0.5)]
    objective = object()
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        objective=objective,
        constraints=constraints,
    )

    acqf = build_acquisition(bundle=_make_bundle(), config=config)

    assert acqf.constraints is constraints
    assert acqf.objective is objective


def test_acquisition_config_forwards_parallel_list_constraints() -> None:
    constraints = make_outcome_constraints(
        output_indices=[1, 2],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    )
    config = AcquisitionConfig(
        name="qei",
        acqf_cls=_AcquisitionWithConstraints,
        constraints=constraints,
    )

    acqf = build_acquisition(bundle=_make_bundle(), config=config)

    assert acqf.constraints is constraints
    assert len(acqf.constraints) == 2


def test_constraints_are_parallel_to_objective_in_config() -> None:
    constraint = make_outcome_constraint(0, "le", 1.0)
    config = AcquisitionConfig(
        name="qei",
        constraints=[constraint],
    )

    assert config.constraints == [constraint]
    assert config.acqf_kwargs["constraints"] == [constraint]


def test_legacy_constraints_inside_acqf_kwargs_are_rejected() -> None:
    with pytest.raises(ValueError, match="AcquisitionConfig.constraints"):
        AcquisitionConfig(
            name="qei",
            acqf_kwargs={"constraints": [lambda samples: samples[..., 0]]},
        )
