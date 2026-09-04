from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from bochan.models.regression.gaussian.materials.structure import workflow_factory


def test_workflow_mode_normalization() -> None:
    assert workflow_factory.normalize_material_workflow_mode(" model ") == "model_only"
    assert workflow_factory.normalize_material_workflow_mode("rank") == "relax_rank"
    assert workflow_factory.normalize_material_workflow_mode("BO") == "relax_acquisition"
    assert workflow_factory.normalize_material_workflow_mode("al") == "relax_acquisition"

    with pytest.raises(ValueError, match="Unsupported material workflow mode"):
        workflow_factory.normalize_material_workflow_mode("lookahead")


def test_workflow_spec_normalizes_all_axes() -> None:
    spec = workflow_factory.MaterialWorkflowSpec(
        backend="alignn_ff",
        quantity="FORCE",
        model_mode="residual-gp",
        workflow_mode="acquisition",
    )

    assert spec.as_dict() == {
        "backend": "alignn-ff",
        "quantity": "force",
        "model_mode": "residual_gp",
        "workflow_mode": "relax_acquisition",
    }
    assert spec.model_spec.as_dict() == {
        "backend": "alignn-ff",
        "quantity": "force",
        "mode": "residual_gp",
    }


def test_model_only_workflow_defers_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_create_material_model(*args: Any, **kwargs: Any) -> object:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        workflow_factory,
        "create_material_model",
        fake_create_material_model,
    )

    workflow = workflow_factory.create_material_workflow(
        "mace",
        "energy",
        "direct",
    )
    structures = [object(), object()]
    model = workflow.create_model(structures=structures, model_name="small")

    assert model is not None
    assert workflow.ranker is None
    assert workflow.acquisition_selector is None
    assert calls["args"] == ("mace", "energy", "direct")
    kwargs = calls["kwargs"]
    assert kwargs["structures"] is structures
    assert kwargs["model_name"] == "small"
    assert kwargs["train_X"] is None
    assert kwargs["train_Y"] is None


def test_model_only_rejects_relaxer_configuration() -> None:
    with pytest.raises(ValueError, match="model_only workflows do not accept"):
        workflow_factory.create_material_workflow(
            "mace",
            "energy",
            "direct",
            "model_only",
            device="cpu",
        )


def test_relax_rank_dispatches_backend_and_relaxer_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_ranker = SimpleNamespace(name="ranker")

    def fake_create_ranker(backend: str, **kwargs: Any) -> object:
        calls["backend"] = backend
        calls["kwargs"] = kwargs
        return fake_ranker

    monkeypatch.setattr(
        workflow_factory,
        "create_relaxation_ranker",
        fake_create_ranker,
    )

    workflow = workflow_factory.create_material_workflow(
        "chgnet",
        "stress",
        "residual_gp",
        "rank",
        model_name="0.3.0",
    )

    assert workflow.ranker is fake_ranker
    assert workflow.acquisition_selector is None
    assert calls["backend"] == "chgnet"
    assert calls["kwargs"] == {"relaxer": None, "model_name": "0.3.0"}


def test_relax_acquisition_supports_injected_relaxer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    fake_relaxer = SimpleNamespace(relax=lambda *_args, **_kwargs: None)
    fake_selector = SimpleNamespace(name="selector")

    def fake_create_selector(backend: str, **kwargs: Any) -> object:
        calls["backend"] = backend
        calls["kwargs"] = kwargs
        return fake_selector

    monkeypatch.setattr(
        workflow_factory,
        "create_relaxation_acquisition_selector",
        fake_create_selector,
    )

    workflow = workflow_factory.create_material_workflow(
        "m3gnet",
        "force",
        "residual_gp",
        "bo",
        relaxer=fake_relaxer,
    )

    assert workflow.acquisition_selector is fake_selector
    assert workflow.ranker is None
    assert calls["backend"] == "m3gnet"
    assert calls["kwargs"] == {"relaxer": fake_relaxer}


def test_workflow_create_model_forwards_training_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_create_material_model(*args: Any, **kwargs: Any) -> object:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        workflow_factory,
        "create_material_model",
        fake_create_material_model,
    )

    workflow = workflow_factory.MaterialWorkflow(
        spec=workflow_factory.MaterialWorkflowSpec(
            backend="alignn-ff",
            quantity="stress",
            model_mode="residual_gp",
        )
    )
    train_X = torch.zeros(2, 1)
    train_Y = torch.zeros(2, 9)
    structures = [object(), object()]
    structure_graphs = [object(), object()]

    workflow.create_model(
        structures=structures,
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=structure_graphs,
    )

    kwargs = calls["kwargs"]
    assert kwargs["train_X"] is train_X
    assert kwargs["train_Y"] is train_Y
    assert kwargs["structure_graphs"] is structure_graphs
