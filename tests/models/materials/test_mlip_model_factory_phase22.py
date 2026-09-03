from __future__ import annotations

import pytest
import torch

from bochan.models.regression.gaussian.materials.structure import model_factory


def test_material_model_spec_normalizes_and_serializes() -> None:
    spec = model_factory.MaterialModelSpec(
        backend=" ALIGNN_FF ",
        quantity="FORCE",
        mode="residual-gp",
    )

    assert spec.backend == "alignn-ff"
    assert spec.quantity == "force"
    assert spec.mode == "residual_gp"
    assert spec.as_dict() == {
        "backend": "alignn-ff",
        "quantity": "force",
        "mode": "residual_gp",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("direct", "direct"),
        ("PRETRAINED", "direct"),
        ("baseline", "direct"),
        ("residual_gp", "residual_gp"),
        ("residual-gp", "residual_gp"),
        ("residualgp", "residual_gp"),
    ],
)
def test_normalize_material_model_mode(raw: str, expected: str) -> None:
    assert model_factory.normalize_material_model_mode(raw) == expected


def test_normalize_material_model_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported material model mode"):
        model_factory.normalize_material_model_mode("dkl")


def test_create_material_model_direct_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def create_direct(backend: str, quantity: str, **kwargs: object) -> object:
        calls["direct"] = (backend, quantity, kwargs)
        return "direct-model"

    monkeypatch.setattr(model_factory, "create_direct_material_predictor", create_direct)

    result = model_factory.create_material_model(
        "MACE",
        "energy",
        "pretrained",
        structures=("s0", "s1"),
        model_name="custom-mace",
    )

    assert result == "direct-model"
    assert calls["direct"] == (
        "mace",
        "energy",
        {"structures": ("s0", "s1"), "model_name": "custom-mace"},
    )


def test_direct_mode_rejects_training_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_factory,
        "create_direct_material_predictor",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(ValueError, match="do not accept training tensors"):
        model_factory.create_material_model(
            "chgnet",
            "stress",
            "direct",
            structures=("s0",),
            train_X=torch.zeros(1, 1),
        )


def test_residual_mode_requires_training_tensors() -> None:
    with pytest.raises(ValueError, match="require train_X, train_Y"):
        model_factory.create_material_model(
            "m3gnet",
            "force",
            "residual_gp",
            structures=("s0",),
        )


def test_create_material_model_residual_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def create_residual(
        backend: str,
        quantity: str,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        train_Yvar: torch.Tensor | None,
        **kwargs: object,
    ) -> object:
        calls["residual"] = (
            backend,
            quantity,
            train_X,
            train_Y,
            train_Yvar,
            kwargs,
        )
        return "residual-model"

    monkeypatch.setattr(model_factory, "create_material_residual_gp", create_residual)

    train_X = torch.zeros(2, 1)
    train_Y = torch.zeros(2, 9)
    train_Yvar = torch.full((2, 9), 0.1)
    result = model_factory.create_material_model(
        "alignnff",
        "stress",
        "residualgp",
        structures=("s0", "s1"),
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        structure_graphs=("g0", "g1"),
    )

    assert result == "residual-model"
    backend, quantity, x_arg, y_arg, yvar_arg, kwargs = calls["residual"]
    assert backend == "alignn-ff"
    assert quantity == "stress"
    assert x_arg is train_X
    assert y_arg is train_Y
    assert yvar_arg is train_Yvar
    assert kwargs == {
        "structures": ("s0", "s1"),
        "structure_graphs": ("g0", "g1"),
    }
