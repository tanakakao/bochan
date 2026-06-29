from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_acquisition_config_forwards_first_class_constraints() -> None:
    constraints = [lambda samples: 0.5 - samples[..., 1]]
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


def test_constraints_are_parallel_to_objective_in_config() -> None:
    constraint = lambda samples: samples[..., 0] - 1.0
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
