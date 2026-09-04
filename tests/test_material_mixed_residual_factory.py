from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.models.regression.gaussian.materials.structure import capabilities
from bochan.models.regression.gaussian.materials.structure import property_factory


def test_residual_factory_routes_cat_dims_to_mixed_creator(monkeypatch) -> None:
    sentinel = object()
    calls: dict[str, object] = {}

    def creator(
        backend,
        quantity,
        train_X,
        train_Y,
        train_Yvar,
        /,
        *,
        structures,
        cat_dims,
        **kwargs,
    ):
        calls.update(
            backend=backend,
            quantity=quantity,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            structures=structures,
            cat_dims=cat_dims,
            kwargs=kwargs,
        )
        return sentinel

    mixed_module = SimpleNamespace(create_mixed_material_residual_gp=creator)
    monkeypatch.setattr(property_factory, "_load", lambda name: mixed_module)

    train_X = torch.tensor([[0.0, 1.0, 300.0], [1.0, 0.0, 500.0]])
    train_Y = torch.tensor([[1.0], [2.0]])
    result = property_factory.create_material_residual_gp(
        "MACE",
        "Energy",
        train_X,
        train_Y,
        structures=("s0", "s1"),
        cat_dims=[1],
        marker="kept",
    )

    assert result is sentinel
    assert calls["backend"] == "mace"
    assert calls["quantity"] == "energy"
    assert calls["structures"] == ("s0", "s1")
    assert calls["cat_dims"] == [1]
    assert calls["kwargs"] == {"marker": "kept"}


def test_residual_factory_without_cat_dims_keeps_existing_path(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeResidual:
        def __init__(
            self,
            train_X,
            train_Y,
            train_Yvar,
            /,
            *,
            structures,
            **kwargs,
        ) -> None:
            calls.update(
                train_X=train_X,
                train_Y=train_Y,
                train_Yvar=train_Yvar,
                structures=structures,
                kwargs=kwargs,
            )

    monkeypatch.setattr(
        property_factory,
        "_load",
        lambda name: SimpleNamespace(MACEResidualGPModel=FakeResidual),
    )

    train_X = torch.zeros(2, 2)
    train_Y = torch.zeros(2, 1)
    result = property_factory.create_material_residual_gp(
        "mace",
        "energy",
        train_X,
        train_Y,
        structures=("s0", "s1"),
        marker="old-path",
    )

    assert isinstance(result, FakeResidual)
    assert calls["kwargs"] == {"marker": "old-path"}


def test_alignn_ff_mixed_residual_still_requires_structure_graphs() -> None:
    train_X = torch.zeros(2, 2)
    train_Y = torch.zeros(2, 1)

    with pytest.raises(ValueError, match="structure_graphs"):
        property_factory.create_material_residual_gp(
            "alignn-ff",
            "energy",
            train_X,
            train_Y,
            structures=("s0", "s1"),
            cat_dims=[1],
        )


@pytest.mark.parametrize("backend", ("mace", "chgnet", "m3gnet", "alignn-ff"))
def test_all_mlip_backends_advertise_mixed_and_correlated_residuals(backend: str) -> None:
    capability = capabilities.get_material_backend_capabilities(backend)

    assert capability.supports_mixed_residual
    assert capability.supports_correlated_multioutput_residual
    assert capability.residual_input_modes == ("continuous", "mixed")
    assert capability.residual_scalar_quantities == ("energy",)
    assert capability.residual_correlated_multioutput_quantities == ("force", "stress")
    assert capability.supports(
        quantity="energy",
        model_mode="residual_gp",
        residual_input_mode="mixed",
    )
    assert not capability.supports(
        quantity="energy",
        model_mode="direct",
        residual_input_mode="mixed",
    )


def test_mixed_residual_requirements_are_discoverable() -> None:
    mace = capabilities.get_material_backend_capabilities("mace")
    assert mace.requirements(
        quantity="energy",
        model_mode="residual_gp",
        residual_input_mode="mixed",
    ) == ("structures", "train_X", "train_Y", "cat_dims")

    alignn = capabilities.get_material_backend_capabilities("alignn-ff")
    assert alignn.requirements(
        quantity="force",
        model_mode="residual_gp",
        residual_input_mode="mixed",
    ) == (
        "structures",
        "train_X",
        "train_Y",
        "cat_dims",
        "structure_graphs",
        "fixed_atom_count",
    )


def test_residual_input_mode_normalization() -> None:
    assert capabilities.normalize_material_residual_input_mode("continuous") == "continuous"
    assert capabilities.normalize_material_residual_input_mode("numeric") == "continuous"
    assert capabilities.normalize_material_residual_input_mode("mixed") == "mixed"
    assert capabilities.normalize_material_residual_input_mode("mixed-input") == "mixed"

    with pytest.raises(ValueError, match="Unsupported residual input mode"):
        capabilities.normalize_material_residual_input_mode("ordinal")
