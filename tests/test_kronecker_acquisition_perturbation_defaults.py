from __future__ import annotations

from dataclasses import replace

import torch

from bochan.api import (
    AcquisitionConfig,
    InputTransformConfig,
    ModelConfig,
    ObjectiveConfig,
)
from bochan.api.factory import build_model
import bochan.api.engine as engine


def _make_bundle(*, n_w: int = 4):
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = torch.rand(6, 2, dtype=torch.double)
    config = ModelConfig(
        task_type="regression",
        model_type="kronecker",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=True,
            n_w=n_w,
            std=0.05,
        ),
        outcome_transform=False,
    )
    return build_model(train_X, train_Y, config)


def test_multi_output_acquisitions_infer_kronecker_n_w_objective() -> None:
    bundle = _make_bundle(n_w=4)

    for name in ("ehvi", "nehvi", "nparego"):
        resolved = engine._resolve_objective_config_n_w_from_input_transform(
            acq_config=AcquisitionConfig(name=name),
            bundle=bundle,
        )

        assert resolved.objective_config is not None
        assert resolved.objective_config.mode == "multi_output"
        assert resolved.objective_config.n_w == 4
        assert resolved.objective_config.risk_type is None


def test_nsgaii_factory_still_infers_kronecker_n_w_objective() -> None:
    bundle = _make_bundle(n_w=4)
    config = replace(
        AcquisitionConfig(name="nsgaii"),
        acqf_factory=dict,
    )

    resolved = engine._resolve_objective_config_n_w_from_input_transform(
        acq_config=config,
        bundle=bundle,
    )

    assert resolved.objective_config is not None
    assert resolved.objective_config.mode == "multi_output"
    assert resolved.objective_config.n_w == 4


def test_explicit_kronecker_objective_config_is_preserved() -> None:
    bundle = _make_bundle(n_w=4)
    explicit = ObjectiveConfig(
        mode="multi_output",
        n_w=8,
        risk_type="cvar",
        alpha=0.9,
    )

    resolved = engine._resolve_objective_config_n_w_from_input_transform(
        acq_config=AcquisitionConfig(
            name="ehvi",
            objective_config=explicit,
        ),
        bundle=bundle,
    )

    assert resolved.objective_config is explicit
