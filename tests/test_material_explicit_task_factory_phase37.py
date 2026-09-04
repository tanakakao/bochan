from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.models import MultiTaskGP
from torch import Tensor, nn

import bochan.models.regression.gaussian.materials.explicit_task_factory as factory_module
from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskSpec,
    RegisteredMaterialExplicitTaskSpec,
    RegisteredMaterialFeatureExtractor,
    create_registered_material_explicit_task_surrogate,
    registered_material_explicit_task_capabilities,
)


class _ShiftTransform(nn.Module):
    def forward(self, X: Tensor) -> Tensor:
        return X + 1.0


class _DummyExtractor(nn.Module):
    output_dim = 2

    def __init__(self, *, trainable: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False).double()
        self.linear.weight.data.copy_(torch.eye(2, dtype=torch.double))
        self.linear.weight.requires_grad_(trainable)
        self.last_input: Tensor | None = None

    def forward(self, X: Tensor) -> Tensor:
        self.last_input = X.detach().clone()
        return self.linear(X)


class _DummyBaseModel:
    last_train_X: Tensor | None = None
    trainable = False

    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar=None, **kwargs) -> None:
        type(self).last_train_X = train_X.detach().clone()
        extractor = _DummyExtractor(trainable=self.trainable)
        self.input_transform = _ShiftTransform()
        self.deepkernel = SimpleNamespace(feature_extractor=extractor)
        self.latent_dim = extractor.output_dim


class _DummyGP(_DummyBaseModel):
    trainable = False


class _DummyDKL(_DummyBaseModel):
    trainable = True


class _DummyRegistration:
    family = "dummy"
    domain = "composition"

    def supports(self, variant) -> bool:
        return variant in {"gp", "dkl"}

    def resolve_model_class(self, variant):
        return {"gp": _DummyGP, "dkl": _DummyDKL}[variant]


@pytest.fixture
def dummy_registry(monkeypatch):
    registration = _DummyRegistration()
    monkeypatch.setattr(factory_module, "get_material_family", lambda family: registration)
    return registration


def _training_data() -> tuple[Tensor, Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.2, 0.0],
            [0.0, 0.2, 1.0],
            [0.5, 0.4, 0.0],
            [0.5, 0.4, 1.0],
            [1.0, 0.6, 0.0],
            [1.0, 0.6, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.1], [0.2], [0.4], [0.5], [0.7], [0.8]], dtype=torch.double)
    return train_X, train_Y


def test_registered_spec_normalizes_family_and_kind(dummy_registry) -> None:
    spec = RegisteredMaterialExplicitTaskSpec("DuMmY", kind="deep-kernel")

    assert spec.family == "dummy"
    assert spec.kind == "dkl"
    assert spec.base_variant == "dkl"
    assert spec.as_dict()["task_mode"] == "explicit"


def test_registered_feature_extractor_applies_backend_transform() -> None:
    extractor = _DummyExtractor(trainable=False)
    adapter = RegisteredMaterialFeatureExtractor(
        input_transform=_ShiftTransform(),
        feature_extractor=extractor,
        output_dim=2,
    )
    X = torch.tensor([[1.0, 2.0]], dtype=torch.double)

    output = adapter(X)

    assert torch.allclose(output, torch.tensor([[2.0, 3.0]], dtype=torch.double))
    assert extractor.last_input is not None
    assert torch.allclose(extractor.last_input, torch.tensor([[2.0, 3.0]], dtype=torch.double))


def test_factory_removes_task_column_before_backend_construction(dummy_registry) -> None:
    train_X, train_Y = _training_data()

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        kind="gp",
        task_spec=MaterialExplicitTaskSpec(task_feature=-1, all_tasks=(0, 1)),
    )

    assert isinstance(model, MultiTaskGP)
    assert _DummyGP.last_train_X is not None
    assert _DummyGP.last_train_X.shape == (6, 2)
    assert torch.allclose(_DummyGP.last_train_X, train_X[:, :2])
    assert model.material_family == "dummy"
    assert model.material_gaussian_kind == "gp"
    assert model.material_task_mode == "explicit"


def test_dkl_factory_preserves_trainable_encoder_parameters(dummy_registry) -> None:
    train_X, train_Y = _training_data()

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        kind="dkl",
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
    )

    parameters = list(model.input_transform.feature_extractor.parameters())
    assert parameters
    assert any(parameter.requires_grad for parameter in parameters)


def test_factory_supports_nonterminal_task_column(dummy_registry) -> None:
    train_X, train_Y = _training_data()
    moved = torch.cat((train_X[:, -1:], train_X[:, :2]), dim=-1)

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        moved,
        train_Y,
        task_spec=MaterialExplicitTaskSpec(task_feature=0, all_tasks=(0, 1)),
    )

    assert isinstance(model, MultiTaskGP)
    assert _DummyGP.last_train_X is not None
    assert torch.allclose(_DummyGP.last_train_X, train_X[:, :2])


def test_factory_supports_known_noise(dummy_registry) -> None:
    train_X, train_Y = _training_data()
    train_Yvar = torch.full_like(train_Y, 0.01)

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        train_Yvar,
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
    )

    assert isinstance(model, MultiTaskGP)


def test_real_registry_reports_all_material_families() -> None:
    expected = {"crabnet", "roost", "alignn", "chgnet", "m3gnet", "mace"}

    capabilities = {
        family: registered_material_explicit_task_capabilities(family)
        for family in expected
    }

    for family, capability in capabilities.items():
        assert capability["family"] == family
        assert capability["task_mode"] == "explicit"
        assert capability["input_modes"] == ["continuous", "mixed"]
        assert capability["gaussian_kinds"] == ["gp", "dkl"]
        assert capability["mixed_gaussian_kinds"] == ["gp", "dkl"]
        assert capability["mixed_explicit_task"] is True


def test_factory_rejects_wide_targets(dummy_registry) -> None:
    train_X, train_Y = _training_data()

    with pytest.raises(ValueError, match="scalar"):
        create_registered_material_explicit_task_surrogate(
            "dummy",
            train_X,
            train_Y.repeat(1, 2),
            task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
        )
