"""Contract tests for public material residual ``model_type`` routing."""

from __future__ import annotations

import pickle

import pytest

from bochan.api import (
    DEFAULT_MODEL_REGISTRY,
    ModelConfig,
    material_residual_model_types,
    register_material_residual_model_types,
    resolve_model_cls,
)

_FAMILIES = ("chgnet", "m3gnet", "mace")
_NORMAL_VARIANTS = ("residual_gp", "multitask_residual_gp")
_MIXED_VARIANTS = ("mixed_residual_gp", "mixed_multitask_residual_gp")


def _expected_class_name(family: str, variant: str) -> str:
    prefix = {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE"}[family]
    suffix = {
        "residual_gp": "ResidualGPModel",
        "mixed_residual_gp": "MixedResidualGPModel",
        "multitask_residual_gp": "MultiTaskResidualGPModel",
        "mixed_multitask_residual_gp": "MixedMultiTaskResidualGPModel",
    }[variant]
    return f"{prefix}{suffix}"


def test_public_material_residual_model_type_names_are_stable() -> None:
    expected_normal = tuple(
        f"{family}_{variant}"
        for family in _FAMILIES
        for variant in _NORMAL_VARIANTS
    )
    expected_mixed = tuple(
        f"{family}_{variant}"
        for family in _FAMILIES
        for variant in _MIXED_VARIANTS
    )

    assert material_residual_model_types(input_type="normal") == expected_normal
    assert material_residual_model_types(input_type="mixed") == expected_mixed
    assert material_residual_model_types() == tuple(
        f"{family}_{variant}"
        for family in _FAMILIES
        for variant in (*_NORMAL_VARIANTS, *_MIXED_VARIANTS)
    )


def test_registration_is_idempotent_and_preserves_existing_model_types() -> None:
    tree = DEFAULT_MODEL_REGISTRY.raw()
    base_before = tree["normal"]["regression"]["base"]
    crabnet_before = tree["normal"]["regression"]["crabnet_gp"]

    register_material_residual_model_types()
    register_material_residual_model_types()

    assert tree["normal"]["regression"]["base"] == base_before
    assert tree["normal"]["regression"]["crabnet_gp"] == crabnet_before


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("variant", _NORMAL_VARIANTS)
def test_normal_regression_model_types_resolve_lazily(family: str, variant: str) -> None:
    model_type = f"{family}_{variant}"
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type=model_type,
            input_type="normal",
            outcome_transform=False,
        )
    )

    assert model_cls.__name__ == _expected_class_name(family, variant)
    assert model_cls.__module__.startswith(
        "bochan.models.regression.gaussian.materials.structure"
    )
    assert pickle.loads(pickle.dumps(model_cls)) is model_cls


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("variant", _MIXED_VARIANTS)
def test_mixed_regression_model_types_resolve_lazily(family: str, variant: str) -> None:
    model_type = f"{family}_{variant}"
    model_cls = resolve_model_cls(
        ModelConfig(
            task_type="regression",
            model_type=model_type,
            input_type="mixed",
            cat_dims=(2,),
            outcome_transform=False,
        )
    )

    assert model_cls.__name__ == _expected_class_name(family, variant)
    assert pickle.loads(pickle.dumps(model_cls)) is model_cls


@pytest.mark.parametrize("family", _FAMILIES)
def test_correlated_residual_variants_are_available_for_multi_objective(family: str) -> None:
    normal_cls = resolve_model_cls(
        ModelConfig(
            task_type="multi_objective",
            model_type=f"{family}_multitask_residual_gp",
            input_type="normal",
            outcome_transform=False,
        )
    )
    mixed_cls = resolve_model_cls(
        ModelConfig(
            task_type="multi_objective",
            model_type=f"{family}_mixed_multitask_residual_gp",
            input_type="mixed",
            cat_dims=(2,),
            outcome_transform=False,
        )
    )

    assert normal_cls.__name__ == _expected_class_name(family, "multitask_residual_gp")
    assert mixed_cls.__name__ == _expected_class_name(
        family, "mixed_multitask_residual_gp"
    )


def test_scalar_residual_is_not_registered_as_multi_objective() -> None:
    with pytest.raises(ValueError, match="Unknown model setting"):
        resolve_model_cls(
            ModelConfig(
                task_type="multi_objective",
                model_type="mace_residual_gp",
                input_type="normal",
                outcome_transform=False,
            )
        )


def test_invalid_material_input_type_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_type"):
        material_residual_model_types(input_type="categorical")
