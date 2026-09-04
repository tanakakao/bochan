from __future__ import annotations

import pytest

from bochan.models.regression.gaussian.materials import (
    create_material_surrogate,
    material_model_variant,
    material_surrogate_capabilities,
)
from bochan.models.regression.gaussian.materials.common import get_material_family
from bochan.models.regression.gaussian.materials.composition import (
    RoostDKLModel,
    RoostGPModel,
    RoostMixedDKLModel,
    RoostMixedGPModel,
    RoostMixedMultiTaskDKLModel,
    RoostMixedMultiTaskGPModel,
    RoostMultiTaskDKLModel,
    RoostMultiTaskGPModel,
)


EXPECTED = {
    "gp": RoostGPModel,
    "dkl": RoostDKLModel,
    "mixed_gp": RoostMixedGPModel,
    "mixed_dkl": RoostMixedDKLModel,
    "multitask_gp": RoostMultiTaskGPModel,
    "multitask_dkl": RoostMultiTaskDKLModel,
    "mixed_multitask_gp": RoostMixedMultiTaskGPModel,
    "mixed_multitask_dkl": RoostMixedMultiTaskDKLModel,
}


def test_roost_registry_exposes_full_matrix() -> None:
    registration = get_material_family("roost")
    assert registration.variants == frozenset(EXPECTED)
    for variant, expected in EXPECTED.items():
        assert registration.resolve_model_class(variant) is expected


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
def test_roost_variant_selection(kind: str, input_mode: str, output_mode: str, variant: str) -> None:
    assert material_model_variant(
        kind=kind,
        input_mode=input_mode,
        output_mode=output_mode,
    ) == variant


def test_roost_capabilities_match_full_matrix() -> None:
    capabilities = material_surrogate_capabilities("roost")
    assert capabilities["family"] == "roost"
    assert capabilities["domain"] == "composition"
    assert set(capabilities["variants"]) == EXPECTED.keys()


def test_factory_resolves_roost_mixed_correlated_class_before_construction(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeModel:
        def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs) -> None:
            calls.update(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                kwargs=kwargs,
            )

    registration = get_material_family("roost")
    monkeypatch.setattr(registration.__class__, "resolve_model_class", lambda self, variant: FakeModel)

    sentinel_x = object()
    sentinel_y = object()
    model = create_material_surrogate(
        "roost",
        sentinel_x,
        sentinel_y,
        kind="dkl",
        input_mode="mixed",
        output_mode="correlated",
        cat_dims=[3],
        element_ids="ids",
        composition_indices=[0, 1],
    )

    assert isinstance(model, FakeModel)
    assert calls["train_X"] is sentinel_x
    assert calls["train_Y"] is sentinel_y
    assert calls["kwargs"] == {
        "cat_dims": [3],
        "element_ids": "ids",
        "composition_indices": [0, 1],
    }
