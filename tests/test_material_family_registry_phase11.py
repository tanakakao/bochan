from __future__ import annotations

import importlib
import sys

import pytest

from bochan.models.regression.gaussian.materials.common import (
    MATERIAL_FAMILY_REGISTRY,
    get_material_family,
    list_material_families,
)


def test_registry_contains_expected_material_families() -> None:
    assert tuple(MATERIAL_FAMILY_REGISTRY) == (
        "crabnet",
        "roost",
        "alignn",
        "chgnet",
        "m3gnet",
        "mace",
    )
    assert list_material_families(domain="composition") == ("crabnet", "roost")
    assert list_material_families(domain="structure") == (
        "alignn",
        "chgnet",
        "m3gnet",
        "mace",
    )


def test_registry_matches_current_variant_matrix() -> None:
    full_matrix = {
        "gp",
        "dkl",
        "mixed_gp",
        "mixed_dkl",
        "multitask_gp",
        "multitask_dkl",
        "mixed_multitask_gp",
        "mixed_multitask_dkl",
    }
    for family in ("crabnet", "alignn", "chgnet", "m3gnet", "mace"):
        assert get_material_family(family).variants == full_matrix

    assert get_material_family("roost").variants == {"gp", "dkl"}


def test_registry_pretrained_metadata_is_conservative() -> None:
    for registration in MATERIAL_FAMILY_REGISTRY.values():
        capabilities = registration.pretrained.capabilities
        assert capabilities.representation is True
        assert capabilities.fine_tuning is True
        assert capabilities.direct_prediction is False
        assert capabilities.residual_gp is False
        assert registration.supports("residual_gp") is False

    assert get_material_family("mace").pretrained.default_model_name == "medium-mpa-0"
    assert get_material_family("chgnet").pretrained.default_model_name == "0.3.0"
    assert (
        get_material_family("m3gnet").pretrained.default_model_name
        == "M3GNet-PES-MatPES-PBE-2025.2"
    )


def test_registry_resolves_canonical_model_classes_lazily() -> None:
    crabnet = get_material_family("crabnet")
    resolved = crabnet.resolve_model_class("gp")

    from bochan.models.regression.gaussian.materials.composition import CrabNetGPModel

    assert resolved is CrabNetGPModel

    mace = get_material_family("mace")
    resolved_mace = mace.resolve_model_class("mixed_multitask_dkl")

    from bochan.models.regression.gaussian.materials.structure import (
        MACEMixedMultiTaskDKLModel,
    )

    assert resolved_mace is MACEMixedMultiTaskDKLModel


def test_registry_import_does_not_eagerly_load_concrete_namespaces(monkeypatch) -> None:
    structure_name = "bochan.models.regression.gaussian.materials.structure"
    composition_name = "bochan.models.regression.gaussian.materials.composition"
    monkeypatch.delitem(sys.modules, structure_name, raising=False)
    monkeypatch.delitem(sys.modules, composition_name, raising=False)

    registry = importlib.reload(
        sys.modules["bochan.models.regression.gaussian.materials.common.registry"]
    )

    assert registry.MATERIAL_FAMILY_REGISTRY
    assert structure_name not in sys.modules
    assert composition_name not in sys.modules


def test_registry_validates_lookup_and_variant_errors() -> None:
    assert get_material_family(" MACE ").family == "mace"

    with pytest.raises(KeyError, match="Unknown material family"):
        get_material_family("unknown")
    with pytest.raises(ValueError, match="does not support variant"):
        get_material_family("roost").model_path("mixed_gp")
    with pytest.raises(ValueError, match="domain"):
        list_material_families(domain="image")
