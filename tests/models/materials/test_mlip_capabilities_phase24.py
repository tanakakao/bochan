from __future__ import annotations

import pytest

from bochan.models.regression.gaussian.materials.structure.capabilities import (
    get_material_backend_capabilities,
    get_material_capability_catalog,
    list_material_backend_capabilities,
)


def test_all_public_backends_have_full_phase24_capabilities() -> None:
    capabilities = list_material_backend_capabilities()
    assert tuple(item.backend for item in capabilities) == (
        "mace",
        "chgnet",
        "m3gnet",
        "alignn-ff",
    )
    for item in capabilities:
        assert item.direct_quantities == ("energy", "force", "stress")
        assert item.residual_quantities == ("energy", "force", "stress")
        assert item.model_modes == ("direct", "residual_gp")
        assert item.workflow_modes == (
            "model_only",
            "relax_rank",
            "relax_acquisition",
        )
        assert item.supports_relaxation
        assert item.supports_relax_rank
        assert item.supports_relax_acquisition
        assert item.force_fixed_topology
        assert item.stress_components == 9


def test_backend_aliases_are_normalized() -> None:
    capability = get_material_backend_capabilities(" ALIGNN_FF ")
    assert capability.backend == "alignn-ff"
    assert capability.residual_requires_structure_graphs


def test_supports_normalizes_configuration_aliases() -> None:
    capability = get_material_backend_capabilities("mace")
    assert capability.supports(
        quantity=" FORCE ",
        model_mode="baseline",
        workflow_mode="bo",
    )
    assert capability.supports(
        quantity="stress",
        model_mode="residual-gp",
        workflow_mode="rank",
    )


def test_mace_requirements_for_force_residual() -> None:
    capability = get_material_backend_capabilities("mace")
    assert capability.requirements(quantity="force", model_mode="residual_gp") == (
        "structures",
        "train_X",
        "train_Y",
        "fixed_atom_count",
    )


def test_alignn_ff_residual_requires_structure_graphs() -> None:
    capability = get_material_backend_capabilities("alignn-ff")
    assert capability.requirements(quantity="energy", model_mode="residual_gp") == (
        "structures",
        "train_X",
        "train_Y",
        "structure_graphs",
    )


def test_direct_energy_has_only_structure_requirement() -> None:
    capability = get_material_backend_capabilities("chgnet")
    assert capability.requirements(quantity="energy", model_mode="direct") == (
        "structures",
    )


def test_capability_catalog_is_json_ready() -> None:
    catalog = get_material_capability_catalog()
    assert catalog["quantities"] == ["energy", "force", "stress"]
    assert catalog["model_modes"] == ["direct", "residual_gp"]
    assert catalog["workflow_modes"] == [
        "model_only",
        "relax_rank",
        "relax_acquisition",
    ]
    assert len(catalog["backends"]) == 4
    alignn = catalog["backends"][-1]
    assert alignn["backend"] == "alignn-ff"
    assert alignn["residual_requires_structure_graphs"] is True


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported material backend"):
        get_material_backend_capabilities("unknown")
