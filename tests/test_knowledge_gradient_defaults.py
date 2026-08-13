from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

import bochan.api.information_acquisition_defaults as info_defaults
from bochan.api import AcquisitionConfig, DataContext, ModelBundle, ModelConfig, OptimizeConfig
from bochan.api.registry.acquisition import resolve_acqf_cls


def _make_bundle(
    *,
    train_Y: torch.Tensor | None = None,
    task_type: str = "regression",
) -> ModelBundle:
    train_X = torch.tensor(
        [[0.0, 0.2], [0.5, 0.7], [1.0, 0.4]],
        dtype=torch.double,
    )
    if train_Y is None:
        train_Y = torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.double)
    config = ModelConfig(
        task_type=task_type,
        model_type="base",
        outcome_transform=False,
    )
    return ModelBundle(
        model=SimpleNamespace(),
        train_X=train_X,
        train_Y=train_Y,
        model_config=config,
        task_type=task_type,
        model_type="base",
    )


def test_kg_aliases_resolve_to_botorch_qkg() -> None:
    for name in ("kg", "qkg", "knowledgegradient", "qknowledgegradient"):
        resolved = resolve_acqf_cls(
            name,
            task_type="regression",
            multi_output=False,
        )
        assert resolved is qKnowledgeGradient


def test_kg_short_alias_rejects_classification() -> None:
    with pytest.raises(ValueError, match="regression / hybrid"):
        resolve_acqf_cls(
            "kg",
            task_type="binary",
            multi_output=False,
        )


def test_kg_generates_current_value_from_bounds(monkeypatch) -> None:
    bundle = _make_bundle()
    current_value = torch.tensor(1.25, dtype=torch.double)
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "objective": kwargs["objective"],
            "posterior_transform": kwargs["posterior_transform"],
            "num_fantasies": kwargs["num_fantasies"],
            "current_value": current_value,
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(name="kg", acqf_cls=qKnowledgeGradient),
        DataContext(
            bounds=torch.tensor([[0.0, -1.0], [1.0, 2.0]], dtype=torch.double),
            extra={"kg_num_fantasies": 32},
        ),
    )

    assert resolved.acqf_kwargs["current_value"] is current_value
    assert resolved.acqf_kwargs["num_fantasies"] == 32
    assert captured["bounds"] == [(0.0, 1.0), (-1.0, 2.0)]
    assert captured["with_current_value"] is True
    assert captured["num_fantasies"] == 32
    assert captured["objective"] is None
    assert captured["posterior_transform"] is None


def test_kg_preserves_explicit_current_value_without_bounds(monkeypatch) -> None:
    bundle = _make_bundle()
    current_value = torch.tensor(0.75, dtype=torch.double)
    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: pytest.fail("KG constructor must not run"),
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="kg",
            acqf_cls=qKnowledgeGradient,
            acqf_kwargs={"current_value": current_value},
        ),
        DataContext(),
    )

    assert resolved.acqf_kwargs["current_value"] is current_value
    assert resolved.acqf_kwargs["num_fantasies"] == 64


def test_kg_explicit_num_fantasies_takes_precedence() -> None:
    bundle = _make_bundle()
    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="kg",
            acqf_cls=qKnowledgeGradient,
            acqf_kwargs={
                "current_value": torch.tensor(0.5),
                "num_fantasies": 11,
            },
        ),
        DataContext(extra={"kg_num_fantasies": 99}),
    )
    assert resolved.acqf_kwargs["num_fantasies"] == 11


def test_kg_multi_output_requires_scalar_terminal_objective() -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    with pytest.raises(ValueError, match="scalar terminal objective"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name="kg", acqf_cls=qKnowledgeGradient),
            DataContext(),
        )


def test_kg_multi_output_accepts_explicit_objective(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    objective = object()
    current_value = torch.tensor(0.9)
    captured = {}

    def fake_constructor(**kwargs):
        captured.update(kwargs)
        return {
            "model": kwargs["model"],
            "objective": kwargs["objective"],
            "posterior_transform": kwargs["posterior_transform"],
            "num_fantasies": kwargs["num_fantasies"],
            "current_value": current_value,
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="kg",
            acqf_cls=qKnowledgeGradient,
            objective=objective,
        ),
        DataContext(bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
    )

    assert resolved.objective is objective
    assert captured["objective"] is objective
    assert resolved.acqf_kwargs["current_value"] is current_value


def test_kg_multi_output_accepts_posterior_transform(monkeypatch) -> None:
    bundle = _make_bundle(
        train_Y=torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.5]],
            dtype=torch.double,
        )
    )
    posterior_transform = object()
    current_value = torch.tensor(0.8)

    def fake_constructor(**kwargs):
        assert kwargs["posterior_transform"] is posterior_transform
        return {
            "model": kwargs["model"],
            "objective": None,
            "posterior_transform": posterior_transform,
            "num_fantasies": kwargs["num_fantasies"],
            "current_value": current_value,
        }

    monkeypatch.setattr(
        info_defaults,
        "_get_botorch_input_constructor",
        lambda cls: fake_constructor,
    )

    resolved, _ = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="kg",
            acqf_cls=qKnowledgeGradient,
            acqf_kwargs={"posterior_transform": posterior_transform},
        ),
        DataContext(bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]])),
    )

    assert resolved.acqf_kwargs["posterior_transform"] is posterior_transform
    assert resolved.acqf_kwargs["current_value"] is current_value


def test_kg_auto_current_value_rejects_pending_points() -> None:
    bundle = _make_bundle()
    with pytest.raises(ValueError, match="does not condition on X_pending"):
        info_defaults.resolve_information_acquisition_defaults(
            bundle,
            AcquisitionConfig(name="kg", acqf_cls=qKnowledgeGradient),
            DataContext(
                bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                X_pending=torch.tensor([[0.25, 0.75]]),
            ),
        )


def test_kg_pending_points_are_allowed_with_explicit_current_value() -> None:
    bundle = _make_bundle()
    current_value = torch.tensor(0.6)
    resolved, context = info_defaults.resolve_information_acquisition_defaults(
        bundle,
        AcquisitionConfig(
            name="kg",
            acqf_cls=qKnowledgeGradient,
            acqf_kwargs={"current_value": current_value},
        ),
        DataContext(X_pending=torch.tensor([[0.25, 0.75]])),
    )
    assert resolved.acqf_kwargs["current_value"] is current_value
    assert context.X_pending is not None


def test_kg_one_shot_optimization_remains_joint() -> None:
    resolved = info_defaults.resolve_information_optimizer_defaults(
        AcquisitionConfig(name="kg", acqf_cls=qKnowledgeGradient),
        OptimizeConfig(q=2, sequential=True),
    )
    assert resolved.sequential is False
