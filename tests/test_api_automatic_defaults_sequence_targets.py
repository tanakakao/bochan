from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig
import bochan.api.engine_defaults as engine_defaults
from bochan.api.automatic_multiobjective import observed_multiobjective_values


class _NeedsEHVIContext:
    def __init__(self, model, ref_point, partitioning) -> None:
        self.model = model
        self.ref_point = ref_point
        self.partitioning = partitioning


class _ListReturningHybridObjective(MCMultiOutputObjective):
    def forward(self, samples, X=None):
        return [samples[..., 0], -samples[..., 1]]


def _make_sequence_target_bundle() -> tuple[ModelBundle, torch.Tensor]:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    original_Y = torch.tensor(
        [[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]],
        dtype=torch.double,
    )
    config = ModelConfig(
        task_type="regression",
        model_type="base",
        outcome_transform=False,
    )
    sub_bundles = [
        ModelBundle(
            model=SimpleNamespace(),
            train_X=train_X,
            train_Y=original_Y[:, i : i + 1],
            model_config=config,
            task_type="regression",
            model_type="base",
        )
        for i in range(original_Y.shape[-1])
    ]
    bundle = ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=(
            torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double),
            torch.tensor([10.0, 11.0, 12.0], dtype=torch.double),
        ),
        model_config=config,
        task_type="regression",
        model_type="base",
        metadata={"sub_bundles": sub_bundles, "multi_output": True},
    )
    return bundle, original_Y


def test_observed_multiobjective_values_combine_sub_bundle_targets() -> None:
    bundle, original_Y = _make_sequence_target_bundle()

    values = observed_multiobjective_values(
        bundle,
        AcquisitionConfig(name="ehi", acqf_cls=_NeedsEHVIContext),
        DataContext(),
    )

    torch.testing.assert_close(values, original_Y)


def test_observed_values_normalize_list_returning_hybrid_objective() -> None:
    bundle, original_Y = _make_sequence_target_bundle()

    values = observed_multiobjective_values(
        bundle,
        AcquisitionConfig(
            name="nparego",
            acqf_cls=_NeedsEHVIContext,
            objective=_ListReturningHybridObjective(),
        ),
        DataContext(),
    )

    expected = torch.stack([original_Y[:, 0], -original_Y[:, 1]], dim=-1)
    torch.testing.assert_close(values, expected)


def test_ehvi_defaults_support_model_list_target_sequences(monkeypatch) -> None:
    bundle, original_Y = _make_sequence_target_bundle()
    sentinel = object()
    captured = {}

    def fake_partitioning(ref_point, values):
        captured["ref_point"] = ref_point
        captured["values"] = values
        return sentinel

    monkeypatch.setattr(engine_defaults, "make_partitioning", fake_partitioning)

    context = engine_defaults.resolve_acquisition_data_context(
        bundle,
        AcquisitionConfig(name="ehi", acqf_cls=_NeedsEHVIContext),
        DataContext(),
    )

    torch.testing.assert_close(
        context.ref_point,
        torch.tensor([-0.1, 1.9], dtype=torch.double),
    )
    torch.testing.assert_close(captured["values"], original_Y)
    assert context.partitioning is sentinel
