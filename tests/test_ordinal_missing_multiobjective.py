from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bochan.acquisition.ordinal.bayesian_optimization as ordinal_bo
from bochan.api.automatic_multiobjective import (
    make_default_ref_point,
    make_partitioning,
    observed_multiobjective_values,
)
from bochan.api.configs import (
    AcquisitionConfig,
    DataContext,
    ModelBundle,
    ModelConfig,
)


def _ordinal_bundle(train_Y: torch.Tensor) -> ModelBundle:
    train_X = torch.linspace(0.0, 1.0, train_Y.shape[0], dtype=torch.double).unsqueeze(-1)
    model_config = ModelConfig(
        task_type="ordinal",
        model_type="multitask",
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(num_outputs=train_Y.shape[-1]),
        train_X=train_X,
        train_Y=train_Y,
        model_config=model_config,
        task_type="ordinal",
        model_type="multitask",
        metadata={"multi_output": True},
    )


def _ordinal_config() -> AcquisitionConfig:
    return AcquisitionConfig(
        name="ehvi",
        acqf_kwargs={
            "utility_values": [
                torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
                torch.tensor([0.0, 2.0, 4.0], dtype=torch.double),
            ]
        },
    )


def test_observed_ordinal_utility_uses_only_complete_rows() -> None:
    train_Y = torch.tensor(
        [
            [0.0, float("nan")],
            [1.0, 2.0],
            [float("nan"), 1.0],
            [2.0, 0.0],
        ],
        dtype=torch.double,
    )

    values = observed_multiobjective_values(
        _ordinal_bundle(train_Y),
        _ordinal_config(),
        DataContext(),
    )

    expected = torch.tensor(
        [
            [1.0, 4.0],
            [2.0, 0.0],
        ],
        dtype=torch.double,
    )
    torch.testing.assert_close(values, expected)


def test_filtered_ordinal_values_build_finite_ehvi_context() -> None:
    train_Y = torch.tensor(
        [
            [0.0, float("nan")],
            [1.0, 2.0],
            [2.0, 1.0],
            [float("nan"), 0.0],
        ],
        dtype=torch.double,
    )
    values = observed_multiobjective_values(
        _ordinal_bundle(train_Y),
        _ordinal_config(),
        DataContext(),
    )

    ref_point = make_default_ref_point(values)
    partitioning = make_partitioning(ref_point, values)

    assert values.shape == torch.Size([2, 2])
    assert torch.isfinite(values).all()
    assert torch.isfinite(ref_point).all()
    assert partitioning.num_outcomes == 2


def test_no_complete_ordinal_row_has_actionable_error() -> None:
    train_Y = torch.tensor(
        [
            [0.0, float("nan")],
            [float("nan"), 1.0],
        ],
        dtype=torch.double,
    )

    with pytest.raises(
        ValueError,
        match="at least one training row with every objective observed",
    ):
        observed_multiobjective_values(
            _ordinal_bundle(train_Y),
            _ordinal_config(),
            DataContext(),
        )


def test_nparego_prefers_complete_wide_labels_over_long_targets(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_nparego(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        ordinal_bo,
        "_TBatchSafeMultiOutputOrdinalNParEGO",
        fake_nparego,
    )

    train_Y_wide = torch.tensor(
        [
            [0.0, float("nan")],
            [1.0, 2.0],
            [float("nan"), 1.0],
            [2.0, 0.0],
        ],
        dtype=torch.double,
    )
    model = SimpleNamespace(
        num_tasks=2,
        num_outputs=2,
        train_Y_wide=train_Y_wide,
        train_targets=torch.tensor([0.0, 1.0, 2.0, 1.0, 2.0, 0.0]),
    )
    utility_values = [
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
    ]

    ordinal_bo.qMultiOutputOrdinalNParEGO(
        model=model,
        X_baseline=torch.linspace(0.0, 1.0, 4, dtype=torch.double).unsqueeze(-1),
        ref_point=torch.tensor([-0.1, -0.1], dtype=torch.double),
        utility_values=utility_values,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.double,
    )
    torch.testing.assert_close(captured["train_Y"], expected)
