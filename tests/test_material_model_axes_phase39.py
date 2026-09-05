from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bochan.models.regression.gaussian.materials.model_axes as axes_module
from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskSpec,
    MaterialModelAxesSpec,
    create_material_model_from_axes,
    material_model_axes_capabilities,
    normalize_material_fidelity_mode,
    normalize_material_task_mode,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0.1, 0.2], [0.4, 0.5]], dtype=torch.double)
    Y = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    return X, Y


def test_axes_spec_keeps_output_task_and_fidelity_separate() -> None:
    spec = MaterialModelAxesSpec(
        "roost",
        kind="deep-kernel",
        input_mode="mixed-input",
        output_mode="single-output",
        task_mode="task-index",
        fidelity_mode="single-fidelity",
    )

    assert spec.kind == "dkl"
    assert spec.input_mode == "mixed"
    assert spec.output_mode == "scalar"
    assert spec.task_mode == "explicit"
    assert spec.fidelity_mode == "none"
    assert spec.route == "explicit_task"
    assert spec.implemented is True


def test_axes_spec_rejects_wide_output_with_explicit_task() -> None:
    with pytest.raises(ValueError, match="scalar long-format"):
        MaterialModelAxesSpec(
            "roost",
            output_mode="correlated",
            task_mode="explicit",
        )


def test_fidelity_alias_is_not_normalized_as_task() -> None:
    assert normalize_material_task_mode("task") == "explicit"
    assert normalize_material_fidelity_mode("multifidelity") == "continuous"


def test_standard_route_dispatches_to_wide_output_factory(monkeypatch) -> None:
    X, Y = _data()
    sentinel = SimpleNamespace(route="wide")
    captured: dict[str, object] = {}

    def fake_create(family, train_X, train_Y, train_Yvar=None, /, **kwargs):
        captured.update(family=family, X=train_X, Y=train_Y, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(axes_module, "create_material_surrogate", fake_create)

    model = create_material_model_from_axes(
        "roost",
        X,
        Y,
        kind="gp",
        input_mode="continuous",
        output_mode="scalar",
    )

    assert model is sentinel
    assert captured["family"] == "roost"
    assert captured["kwargs"] == {
        "kind": "gp",
        "input_mode": "continuous",
        "output_mode": "scalar",
    }


def test_explicit_task_route_dispatches_to_task_factory(monkeypatch) -> None:
    X, Y = _data()
    task_X = torch.cat((X, torch.tensor([[0.0], [1.0]], dtype=X.dtype)), dim=-1)
    task_spec = MaterialExplicitTaskSpec(task_feature=-1, all_tasks=(0, 1))
    sentinel = SimpleNamespace(route="task")
    captured: dict[str, object] = {}

    def fake_create(family, train_X, train_Y, train_Yvar=None, /, **kwargs):
        captured.update(family=family, X=train_X, Y=train_Y, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        axes_module,
        "create_registered_material_explicit_task_surrogate",
        fake_create,
    )

    model = create_material_model_from_axes(
        "roost",
        task_X,
        Y,
        kind="dkl",
        input_mode="mixed",
        task_mode="explicit",
        task_spec=task_spec,
        cat_dims=[1],
    )

    assert model is sentinel
    assert captured["family"] == "roost"
    assert captured["kwargs"] == {
        "kind": "dkl",
        "input_mode": "mixed",
        "task_spec": task_spec,
        "cat_dims": [1],
    }


def test_fidelity_route_fails_explicitly_instead_of_using_task_kernel() -> None:
    X, Y = _data()

    with pytest.raises(NotImplementedError, match="Do not encode fidelity as an explicit task id"):
        create_material_model_from_axes(
            "roost",
            X,
            Y,
            fidelity_mode="multifidelity",
        )


def test_task_spec_rejected_when_task_axis_is_disabled() -> None:
    X, Y = _data()

    with pytest.raises(ValueError, match="task_spec"):
        create_material_model_from_axes(
            "roost",
            X,
            Y,
            task_spec=MaterialExplicitTaskSpec(),
        )


def test_axes_capabilities_explain_fidelity_boundary() -> None:
    capabilities = material_model_axes_capabilities("roost")

    assert capabilities["implemented_routes"] == ["wide_output", "explicit_task"]
    assert capabilities["fidelity_route_implemented"] is False
    assert capabilities["axes"]["task_mode"] == ["none", "explicit"]
    assert capabilities["axes"]["fidelity_mode"] == ["none", "continuous"]
    assert "not a task alias" in capabilities["notes"]["fidelity"]
