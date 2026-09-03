from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.models.regression.gaussian.materials.structure import property_factory


def test_normalize_material_quantity() -> None:
    assert property_factory.normalize_material_quantity(" Energy ") == "energy"
    assert property_factory.normalize_material_quantity("FORCE") == "force"
    assert property_factory.normalize_material_quantity("stress") == "stress"
    with pytest.raises(ValueError, match="Unsupported material quantity"):
        property_factory.normalize_material_quantity("bandgap")


def test_create_mace_direct_predictor_resolves_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, encoder: object, structures: object) -> None:
            calls["predictor"] = (encoder, structures)

    def resolve_encoder(encoder: object, **kwargs: object) -> object:
        calls["resolve"] = (encoder, kwargs)
        return "resolved-encoder"

    fake_energy = SimpleNamespace(
        _DEFAULT_MODEL_NAME="default-mace",
        _resolve_encoder=resolve_encoder,
        MACEDirectEnergyPredictor=FakePredictor,
    )
    monkeypatch.setattr(
        property_factory,
        "_load",
        lambda name: fake_energy if name == ".mace_residual" else None,
    )

    result = property_factory.create_direct_material_predictor(
        "mace",
        "energy",
        structures=("s0", "s1"),
        model_name="custom-mace",
        pooling="sum",
    )

    assert isinstance(result, FakePredictor)
    assert calls["predictor"] == ("resolved-encoder", ("s0", "s1"))
    _, kwargs = calls["resolve"]
    assert kwargs["model_name"] == "custom-mace"
    assert kwargs["pooling"] == "sum"


def test_create_alignn_ff_direct_force_forwards_backend_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeForcePredictor:
        def __init__(self, structures: object, **kwargs: object) -> None:
            calls["args"] = (structures, kwargs)

    fake_module = SimpleNamespace(ALIGNNFFDirectForcePredictor=FakeForcePredictor)
    monkeypatch.setattr(property_factory, "_load", lambda name: fake_module)

    result = property_factory.create_direct_material_predictor(
        "alignn_ff",
        "force",
        structures=("s0",),
        calculator="calculator",
        num_atoms=4,
    )

    assert isinstance(result, FakeForcePredictor)
    assert calls["args"] == (
        ("s0",),
        {"calculator": "calculator", "num_atoms": 4},
    )


@pytest.mark.parametrize(
    ("backend", "quantity", "expected_class"),
    [
        ("mace", "energy", "MACEResidualGPModel"),
        ("mace", "force", "MACEForceResidualGPModel"),
        ("chgnet", "stress", "CHGNetStressResidualGPModel"),
        ("m3gnet", "force", "M3GNetForceResidualGPModel"),
        ("alignn-ff", "energy", "ALIGNNFFEnergyResidualGPModel"),
    ],
)
def test_create_material_residual_gp_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    quantity: str,
    expected_class: str,
) -> None:
    calls: dict[str, object] = {}

    class FakeResidual:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["args"] = args
            calls["kwargs"] = kwargs

    fake_module = SimpleNamespace(**{expected_class: FakeResidual})
    monkeypatch.setattr(property_factory, "_load", lambda name: fake_module)

    train_X = torch.zeros(2, 1)
    train_Y = torch.zeros(2, 1)
    extra = {"structure_graphs": ("g0", "g1")} if backend == "alignn-ff" else {}
    result = property_factory.create_material_residual_gp(
        backend,
        quantity,
        train_X,
        train_Y,
        structures=("s0", "s1"),
        **extra,
    )

    assert isinstance(result, FakeResidual)
    assert calls["args"] == (train_X, train_Y, None)
    assert calls["kwargs"]["structures"] == ("s0", "s1")


def test_alignn_ff_residual_requires_structure_graphs() -> None:
    with pytest.raises(ValueError, match="requires structure_graphs"):
        property_factory.create_material_residual_gp(
            "alignn-ff",
            "stress",
            torch.zeros(1, 1),
            torch.zeros(1, 9),
            structures=("s0",),
        )


def test_direct_predictor_rejects_unused_mace_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_energy = SimpleNamespace(
        _DEFAULT_MODEL_NAME="default-mace",
        _resolve_encoder=lambda *args, **kwargs: "encoder",
        MACEDirectEnergyPredictor=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(property_factory, "_load", lambda name: fake_energy)
    with pytest.raises(TypeError, match="Unsupported direct-predictor arguments"):
        property_factory.create_direct_material_predictor(
            "mace",
            "energy",
            structures=("s0",),
            calculator="not-a-mace-energy-option",
        )
