from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bochan.api.modeling.materials as material_api
from bochan.api import (
    MaterialAPIModelSpec,
    build_model,
    make_material_model_config,
    material_task_fixed_features,
)


def test_material_api_config_routes_through_model_factory(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_create(
        family,
        train_X,
        train_Y,
        train_Yvar=None,
        /,
        **kwargs,
    ):
        calls.update(
            family=family,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            kwargs=kwargs,
        )
        return SimpleNamespace()

    monkeypatch.setattr(material_api, "create_material_model_from_axes", fake_create)
    spec = MaterialAPIModelSpec(family="crabnet", kind="dkl")
    config = make_material_model_config(spec, outcome_transform=False)
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = torch.rand(6, 1, dtype=torch.double)

    bundle = build_model(train_X, train_Y, config)

    assert calls["family"] == "crabnet"
    assert calls["train_X"] is train_X
    assert calls["train_Y"] is train_Y
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["kind"] == "dkl"
    assert kwargs["input_mode"] == "continuous"
    assert kwargs["output_mode"] == "scalar"
    assert kwargs["task_mode"] == "none"
    assert kwargs["fidelity_mode"] == "none"
    assert bundle.model_type == "material:crabnet:wide_output"
    assert bundle.model.material_model_axes["route"] == "wide_output"


def test_material_api_mixed_requires_categories() -> None:
    spec = MaterialAPIModelSpec(family="roost", input_mode="mixed")

    with pytest.raises(ValueError, match="cat_dims"):
        make_material_model_config(spec)

    config = make_material_model_config(spec, cat_dims=[2])
    assert config.input_type == "mixed"
    assert list(config.cat_dims or []) == [2]
    assert config.pass_cat_dims is True


def test_material_api_continuous_rejects_categories() -> None:
    spec = MaterialAPIModelSpec(family="mace", input_mode="continuous")

    with pytest.raises(ValueError, match="omitted"):
        make_material_model_config(spec, cat_dims=[1])


def test_explicit_task_fixed_features_resolve_negative_index() -> None:
    spec = MaterialAPIModelSpec(
        family="chgnet",
        task_mode="explicit",
        task_feature=-1,
        all_tasks=(0, 1, 2),
    )

    fixed = material_task_fixed_features(spec, 2, input_dim=5)

    assert fixed == {4: 2.0}


def test_explicit_task_fixed_features_validate_task_id() -> None:
    spec = MaterialAPIModelSpec(
        family="m3gnet",
        task_mode="explicit",
        task_feature=0,
        all_tasks=(0, 1),
    )

    with pytest.raises(ValueError, match="all_tasks"):
        material_task_fixed_features(spec, 3, input_dim=4)


def test_non_task_model_rejects_target_task() -> None:
    spec = MaterialAPIModelSpec(family="alignn")

    with pytest.raises(ValueError, match="task_mode='explicit'"):
        material_task_fixed_features(spec, 0, input_dim=3)


def test_fidelity_remains_reserved_in_high_level_api() -> None:
    spec = MaterialAPIModelSpec(
        family="crabnet",
        fidelity_mode="continuous",
    )

    with pytest.raises(NotImplementedError, match="reserved but unimplemented"):
        make_material_model_config(spec)


def test_explicit_task_config_carries_task_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create(family, train_X, train_Y, train_Yvar=None, /, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(material_api, "create_material_model_from_axes", fake_create)
    spec = MaterialAPIModelSpec(
        family="mace",
        task_mode="explicit",
        task_feature=0,
        all_tasks=(10, 20),
        output_tasks=(20,),
    )
    config = make_material_model_config(spec, outcome_transform=False)
    train_X = torch.tensor(
        [[10.0, 0.1], [20.0, 0.1], [10.0, 0.8], [20.0, 0.8]],
        dtype=torch.double,
    )
    train_Y = torch.rand(4, 1, dtype=torch.double)

    bundle = build_model(train_X, train_Y, config)

    task_spec = captured["task_spec"]
    assert task_spec.task_feature == 0
    assert task_spec.all_tasks == (10, 20)
    assert task_spec.output_tasks == (20,)
    assert bundle.model.material_model_axes["task_mode"] == "explicit"
