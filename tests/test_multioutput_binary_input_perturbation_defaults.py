from __future__ import annotations

from types import SimpleNamespace

import torch

import bochan.api as api_module
from bochan.acquisition.objective import (
    MultiOutputBinaryClassificationInputPerturbationObjective,
)
from bochan.api import (
    AcquisitionConfig,
    InputTransformConfig,
    ModelConfig,
    ObjectiveConfig,
)
from bochan.api.factory import build_objective


def _make_binary_bundle(
    *,
    n_w: int = 16,
    multi_output: bool = True,
):
    model_config = ModelConfig(
        task_type="binary",
        model_type="base",
        input_transform_config=InputTransformConfig(
            normalize=True,
            perturbation=True,
            n_w=n_w,
        ),
        outcome_transform=False,
    )
    return SimpleNamespace(
        task_type="binary",
        model_type="base",
        model=SimpleNamespace(num_outputs=2 if multi_output else 1),
        model_config=model_config,
        metadata={"multi_output": multi_output},
    )


def test_pareto_binary_acquisitions_infer_multi_output_objective() -> None:
    bundle = _make_binary_bundle(n_w=16)

    for name in ("ehvi", "nehvi", "nparego"):
        resolved = api_module._resolve_objective_config_n_w_with_default(
            acq_config=AcquisitionConfig(name=name),
            bundle=bundle,
        )

        assert resolved.objective_config is not None
        assert resolved.objective_config.mode == "multi_output"
        assert resolved.objective_config.n_w == 16
        assert resolved.objective_config.risk_type is None


def test_nsgaii_infers_multi_output_objective_with_factory() -> None:
    bundle = _make_binary_bundle(n_w=16)
    config = AcquisitionConfig(
        name="nsgaii",
        acqf_factory=dict,
    )

    resolved = api_module._resolve_objective_config_n_w_with_default(
        acq_config=config,
        bundle=bundle,
    )

    assert resolved.objective_config is not None
    assert resolved.objective_config.mode == "multi_output"
    assert resolved.objective_config.n_w == 16


def test_score_based_binary_acquisitions_do_not_receive_mc_objective() -> None:
    bundle = _make_binary_bundle(n_w=16)

    for name in ("bald", "entropy", "variance", "straddle", "icu"):
        resolved = api_module._resolve_objective_config_n_w_with_default(
            acq_config=AcquisitionConfig(name=name),
            bundle=bundle,
        )

        assert resolved.objective_config is None


def test_explicit_binary_objective_config_is_preserved() -> None:
    bundle = _make_binary_bundle(n_w=16)
    explicit = ObjectiveConfig(
        mode="multi_output",
        n_w=4,
        risk_type="cvar",
        alpha=0.9,
    )

    resolved = api_module._resolve_objective_config_n_w_with_default(
        acq_config=AcquisitionConfig(
            name="ehvi",
            objective_config=explicit,
        ),
        bundle=bundle,
    )

    assert resolved.objective_config is explicit


def test_inferred_binary_objective_aggregates_48_rows_to_q3() -> None:
    bundle = _make_binary_bundle(n_w=16)
    resolved = api_module._resolve_objective_config_n_w_with_default(
        acq_config=AcquisitionConfig(name="ehvi"),
        bundle=bundle,
    )
    objective = build_objective(
        bundle=bundle,
        config=resolved,
    )

    assert isinstance(
        objective,
        MultiOutputBinaryClassificationInputPerturbationObjective,
    )

    samples = torch.arange(
        5 * 2 * 48 * 2,
        dtype=torch.double,
    ).reshape(5, 2, 48, 2)
    X = torch.rand(2, 3, 5, dtype=torch.double)

    values = objective(samples=samples, X=X)
    expected = samples.reshape(5, 2, 3, 16, 2).mean(dim=-2)

    assert values.shape == torch.Size([5, 2, 3, 2])
    assert torch.allclose(values, expected)
