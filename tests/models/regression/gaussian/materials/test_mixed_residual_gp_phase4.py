"""Contracts for mixed-input pretrained residual Gaussian models."""

from __future__ import annotations

import inspect

from bochan.models.regression.gaussian.materials import get_material_family
from bochan.models.regression.gaussian.materials.common.residual import ResidualMaterialGPModel
from bochan.models.regression.gaussian.materials.structure import (
    CHGNetMixedResidualGPModel,
    M3GNetMixedResidualGPModel,
    MACEMixedResidualGPModel,
)


def test_structure_residual_families_register_mixed_residual_variant() -> None:
    expected = {
        "chgnet": CHGNetMixedResidualGPModel,
        "m3gnet": M3GNetMixedResidualGPModel,
        "mace": MACEMixedResidualGPModel,
    }
    for family, model_class in expected.items():
        registration = get_material_family(family)
        assert registration.supports("mixed_residual_gp")
        assert registration.resolve_model_class("mixed_residual_gp") is model_class
        assert registration.pretrained.supports_residual_gp is True


def test_non_direct_families_do_not_advertise_mixed_residual() -> None:
    for family in ("crabnet", "roost", "alignn"):
        registration = get_material_family(family)
        assert not registration.supports("mixed_residual_gp")


def test_mixed_residual_models_keep_explicit_categorical_contract() -> None:
    for model_class in (
        CHGNetMixedResidualGPModel,
        M3GNetMixedResidualGPModel,
        MACEMixedResidualGPModel,
    ):
        assert issubclass(model_class, ResidualMaterialGPModel)
        signature = inspect.signature(model_class.__init__)
        assert "cat_dims" in signature.parameters
        assert signature.parameters["cat_dims"].default is inspect.Parameter.empty
        assert "train_Yvar" in signature.parameters
        assert "structures" in signature.parameters
