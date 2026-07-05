from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.api import (
    AcquisitionConfig,
    InputTransformConfig,
    ModelBundle,
    ModelConfig,
    ObjectiveConfig,
)
import bochan.api.engine as engine


def _make_bundle(
    *,
    n_w: int = 4,
    perturbation: bool = True,
    n_outputs: int = 1,
    task_type: str = "regression",
) -> ModelBundle:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.zeros(3, n_outputs, dtype=torch.double)
    model_config = ModelConfig(
        task_type=task_type,
        model_type="base",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=perturbation,
            n_w=n_w,
        ),
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(num_outputs=n_outputs),
        train_X=train_X,
        train_Y=train_Y,
        model_config=model_config,
        task_type=task_type,
        model_type="base",
        metadata={"multi_output": n_outputs > 1},
    )


def _resolve(
    config: AcquisitionConfig,
    bundle: ModelBundle,
) -> AcquisitionConfig:
    return engine._resolve_objective_config_n_w_from_input_transform(
        acq_config=config,
        bundle=bundle,
    )


def test_missing_objective_config_uses_input_transform_n_w() -> None:
    resolved = _resolve(
        AcquisitionConfig(name="qEI"),
        _make_bundle(n_w=4),
    )

    assert resolved.objective_config is not None
    assert resolved.objective_config.n_w == 4
    assert resolved.objective_config.risk_type is None
    assert resolved.objective_config.mode == "auto"
    assert resolved.objective_config.output is None


def test_existing_objective_config_without_n_w_uses_input_transform_n_w() -> None:
    resolved = _resolve(
        AcquisitionConfig(
            name="qEI",
            objective_config=ObjectiveConfig(
                mode="scalar",
                output=0,
                risk_type=None,
                n_w=None,
            ),
        ),
        _make_bundle(n_w=6),
    )

    assert resolved.objective_config is not None
    assert resolved.objective_config.n_w == 6
    assert resolved.objective_config.output == 0


def test_explicit_objective_n_w_takes_precedence() -> None:
    objective_config = ObjectiveConfig(n_w=8, risk_type=None)
    resolved = _resolve(
        AcquisitionConfig(name="qEI", objective_config=objective_config),
        _make_bundle(n_w=4),
    )

    assert resolved.objective_config is objective_config
    assert resolved.objective_config.n_w == 8


def test_missing_objective_config_is_not_created_without_perturbation() -> None:
    config = AcquisitionConfig(name="qEI")
    resolved = _resolve(config, _make_bundle(perturbation=False))

    assert resolved is config
    assert resolved.objective_config is None


def test_multi_output_vector_acquisitions_use_perturbation_objective() -> None:
    bundle = _make_bundle(n_outputs=2, n_w=4)

    for name in ("ehvi", "nehvi", "nparego", "nsgaii"):
        resolved = _resolve(AcquisitionConfig(name=name), bundle)

        assert resolved.objective_config is not None
        assert resolved.objective_config.mode == "multi_output"
        assert resolved.objective_config.n_w == 4
        assert resolved.objective_config.risk_type is None


def test_multi_output_score_acquisition_does_not_receive_mc_objective() -> None:
    config = AcquisitionConfig(name="bald")
    resolved = _resolve(config, _make_bundle(n_outputs=2, n_w=4))

    assert resolved is config
    assert resolved.objective_config is None


def test_explicit_objective_is_not_replaced() -> None:
    objective = object()
    config = AcquisitionConfig(name="qEI", objective=objective)
    resolved = _resolve(config, _make_bundle(n_w=4))

    assert resolved.objective is objective
    assert resolved.objective_config is None
