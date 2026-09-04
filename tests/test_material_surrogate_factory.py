from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.models.regression.gaussian.materials import surrogate_factory


@pytest.mark.parametrize(
    ("kind", "input_mode", "output_mode", "variant"),
    (
        ("gp", "continuous", "scalar", "gp"),
        ("dkl", "continuous", "scalar", "dkl"),
        ("gp", "mixed", "scalar", "mixed_gp"),
        ("dkl", "mixed", "scalar", "mixed_dkl"),
        ("gp", "continuous", "correlated", "multitask_gp"),
        ("dkl", "continuous", "correlated", "multitask_dkl"),
        ("gp", "mixed", "correlated", "mixed_multitask_gp"),
        ("dkl", "mixed", "correlated", "mixed_multitask_dkl"),
    ),
)
def test_material_model_variant_matrix(kind, input_mode, output_mode, variant) -> None:
    assert surrogate_factory.material_model_variant(
        kind=kind,
        input_mode=input_mode,
        output_mode=output_mode,
    ) == variant


def test_output_aliases_preserve_historical_multitask_name() -> None:
    assert surrogate_factory.normalize_material_output_mode("multi-output") == "correlated"
    assert surrogate_factory.normalize_material_output_mode("multitask") == "correlated"
    assert (
        surrogate_factory.material_model_variant(
            kind="gp",
            input_mode="mixed",
            output_mode="multitask",
        )
        == "mixed_multitask_gp"
    )


@pytest.mark.parametrize(
    ("family", "count"),
    (
        ("crabnet", 8),
        ("alignn", 8),
        ("chgnet", 8),
        ("m3gnet", 8),
        ("mace", 8),
        ("roost", 8),
    ),
)
def test_registered_family_capability_matrix(family: str, count: int) -> None:
    capability = surrogate_factory.material_surrogate_capabilities(family)
    assert capability["family"] == family
    assert len(capability["configurations"]) == count


def test_roost_accepts_mixed_and_correlated_variants() -> None:
    mixed = surrogate_factory.RegisteredMaterialSurrogateSpec(
        family="roost",
        input_mode="mixed",
    )
    assert mixed.variant == "mixed_gp"

    correlated = surrogate_factory.RegisteredMaterialSurrogateSpec(
        family="roost",
        output_mode="correlated",
    )
    assert correlated.variant == "multitask_gp"

    combined = surrogate_factory.RegisteredMaterialSurrogateSpec(
        family="roost",
        kind="dkl",
        input_mode="mixed",
        output_mode="correlated",
    )
    assert combined.variant == "mixed_multitask_dkl"


def test_factory_resolves_registry_class_and_delegates_kwargs(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs) -> None:
            calls.update(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                kwargs=kwargs,
            )

    registration = SimpleNamespace(
        family="mace",
        domain="structure",
        variants=frozenset({"mixed_multitask_dkl"}),
        supports=lambda variant: variant == "mixed_multitask_dkl",
        resolve_model_class=lambda variant: FakeModel,
    )
    monkeypatch.setattr(surrogate_factory, "get_material_family", lambda family: registration)

    train_X = torch.zeros(3, 2)
    train_Y = torch.zeros(3, 4)
    train_Yvar = torch.ones(3, 4)
    model = surrogate_factory.create_material_surrogate(
        "mace",
        train_X,
        train_Y,
        train_Yvar,
        kind="dkl",
        input_mode="mixed",
        output_mode="multi-output",
        structures=("s0", "s1", "s2"),
        cat_dims=[1],
    )

    assert isinstance(model, FakeModel)
    assert calls["train_X"] is train_X
    assert calls["train_Y"] is train_Y
    assert calls["train_Yvar"] is train_Yvar
    assert calls["kwargs"] == {"structures": ("s0", "s1", "s2"), "cat_dims": [1]}


def test_spec_is_serializable_and_normalized() -> None:
    spec = surrogate_factory.RegisteredMaterialSurrogateSpec(
        family="MACE",
        kind="deep-kernel",
        input_mode="mixed-input",
        output_mode="multi-task",
    )
    assert spec.as_dict() == {
        "family": "mace",
        "domain": "structure",
        "kind": "dkl",
        "input_mode": "mixed",
        "output_mode": "correlated",
        "variant": "mixed_multitask_dkl",
    }
