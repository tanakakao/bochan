from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)

import bochan.api.information_acquisition_defaults as info_defaults
from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig


def test_explicit_hvkg_ref_point_overrides_context(monkeypatch) -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        dtype=torch.double,
    )
    bundle = ModelBundle(
        model=SimpleNamespace(),
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
    explicit_ref = torch.tensor([-0.2, -0.3], dtype=torch.double)
    context_ref = torch.tensor([-9.0, -9.0], dtype=torch.double)

    def fake_constructor(**kwargs):
        assert kwargs["ref_point"] is explicit_ref
        return {
            "model": kwargs["model"],
            "ref_point": kwargs["ref_point"],
            "current_value": torch.tensor(0.1, dtype=torch.double),
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, context = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="hvkg",
            acqf_cls=qHypervolumeKnowledgeGradient,
            acqf_kwargs={"ref_point": explicit_ref},
        ),
        DataContext(
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            ref_point=context_ref,
        ),
    )

    assert resolved.acqf_kwargs["ref_point"] is explicit_ref
    assert context.ref_point is None
