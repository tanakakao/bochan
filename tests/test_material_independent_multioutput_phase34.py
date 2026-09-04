from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bochan.models.regression.gaussian.materials import surrogate_factory


def test_independent_output_mode_builds_one_scalar_model_per_output(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs) -> None:
            calls.append(
                {
                    "train_X": train_X,
                    "train_Y": train_Y,
                    "train_Yvar": train_Yvar,
                    "kwargs": kwargs,
                }
            )

    class FakeModelList:
        def __init__(self, *models) -> None:
            self.models = models

    registration = SimpleNamespace(
        family="roost",
        domain="composition",
        variants=frozenset({"mixed_dkl"}),
        supports=lambda variant: variant == "mixed_dkl",
        resolve_model_class=lambda variant: FakeModel,
    )
    monkeypatch.setattr(surrogate_factory, "get_material_family", lambda family: registration)
    monkeypatch.setattr(surrogate_factory, "ModelListGP", FakeModelList)

    train_X = torch.arange(15, dtype=torch.float).reshape(3, 5)
    train_Y = torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
        ]
    )
    train_Yvar = torch.full_like(train_Y, 0.25)

    model = surrogate_factory.create_material_surrogate(
        "roost",
        train_X,
        train_Y,
        train_Yvar,
        kind="dkl",
        input_mode="mixed",
        output_mode="independent",
        cat_dims=[4],
        element_ids="ids",
        composition_indices=[0, 1, 2],
    )

    assert isinstance(model, FakeModelList)
    assert len(model.models) == 3
    assert len(calls) == 3
    for index, call in enumerate(calls):
        assert call["train_X"] is train_X
        assert torch.equal(call["train_Y"], train_Y[:, index : index + 1])
        assert torch.equal(call["train_Yvar"], train_Yvar[:, index : index + 1])
        assert call["kwargs"] == {
            "cat_dims": [4],
            "element_ids": "ids",
            "composition_indices": [0, 1, 2],
        }


def test_independent_requires_wide_targets(monkeypatch) -> None:
    registration = SimpleNamespace(
        family="mace",
        domain="structure",
        variants=frozenset({"gp"}),
        supports=lambda variant: variant == "gp",
        resolve_model_class=lambda variant: object,
    )
    monkeypatch.setattr(surrogate_factory, "get_material_family", lambda family: registration)

    with pytest.raises(ValueError, match="at least two outputs"):
        surrogate_factory.create_material_surrogate(
            "mace",
            torch.zeros(3, 2),
            torch.zeros(3, 1),
            output_mode="independent",
        )


def test_independent_requires_matching_train_yvar_shape(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

    registration = SimpleNamespace(
        family="mace",
        domain="structure",
        variants=frozenset({"gp"}),
        supports=lambda variant: variant == "gp",
        resolve_model_class=lambda variant: FakeModel,
    )
    monkeypatch.setattr(surrogate_factory, "get_material_family", lambda family: registration)

    with pytest.raises(ValueError, match="same shape as train_Y"):
        surrogate_factory.create_material_surrogate(
            "mace",
            torch.zeros(3, 2),
            torch.zeros(3, 2),
            torch.zeros(3, 1),
            output_mode="independent",
        )


def test_independent_capability_reuses_scalar_backend_variant() -> None:
    capabilities = surrogate_factory.material_surrogate_capabilities("roost")
    independent = [
        configuration
        for configuration in capabilities["configurations"]
        if configuration["output_mode"] == "independent"
    ]

    assert len(independent) == 4
    assert {configuration["variant"] for configuration in independent} == {
        "gp",
        "dkl",
        "mixed_gp",
        "mixed_dkl",
    }
