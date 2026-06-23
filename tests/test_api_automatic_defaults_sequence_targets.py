from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig
from bochan.api import engine_defaults
from bochan.api.automatic_multiobjective import observed_multiobjective_values


class _NeedsEHVIContext:
    def __init__(self, model, ref_point, partitioning) -> None:
        self.model = model
        self.ref_point = ref_point
        self.partitioning = partitioning


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
        # ModelListGP-like parent targets. These deliberately differ from the
        # original observations to verify that sub-bundle train_Y is preferred.
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
