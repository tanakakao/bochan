from __future__ import annotations

from types import SimpleNamespace

import torch
from botorch.models import MultiTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import ProductKernel, RBFKernel, ScaleKernel
from torch import Tensor, nn

import bochan.models.regression.gaussian.materials.explicit_task_factory as factory_module
from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskSpec,
    RegisteredMaterialExplicitTaskSpec,
    RegisteredMixedMaterialFeatureExtractor,
    create_registered_material_explicit_task_surrogate,
)


class _IdentityTransform(nn.Module):
    def forward(self, X: Tensor) -> Tensor:
        return X


class _ContinuousExtractor(nn.Module):
    output_dim = 1

    def __init__(self, *, trainable: bool) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.double))
        self.scale.requires_grad_(trainable)

    def forward(self, X: Tensor) -> Tensor:
        return X[..., :1] * self.scale


class _DummyMixedBase:
    trainable = False
    last_train_X: Tensor | None = None

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar=None,
        *,
        cat_dims,
        **kwargs,
    ) -> None:
        type(self).last_train_X = train_X.detach().clone()
        extractor = _ContinuousExtractor(trainable=self.trainable)
        cont_kernel = RBFKernel(ard_num_dims=1, active_dims=[0])
        cat_kernel = CategoricalKernel(ard_num_dims=1, active_dims=[1])
        covar_module = ScaleKernel(cont_kernel + ScaleKernel(cat_kernel))
        self.input_transform = _IdentityTransform()
        self.deepkernel = SimpleNamespace(
            feature_extractor=extractor,
            scale_to_bounds=nn.Identity(),
            covar_module=covar_module,
            ord_dims=[0],
            cat_dims=[1],
            kernel_ord_dims=[0],
            kernel_cat_dims=[1],
            latent_dim=1,
            _preserve_input_layout=True,
        )


class _DummyMixedGP(_DummyMixedBase):
    trainable = False


class _DummyMixedDKL(_DummyMixedBase):
    trainable = True


class _DummyRegistration:
    family = "dummy"
    domain = "composition"

    def supports(self, variant) -> bool:
        return variant in {"mixed_gp", "mixed_dkl"}

    def resolve_model_class(self, variant):
        return {
            "mixed_gp": _DummyMixedGP,
            "mixed_dkl": _DummyMixedDKL,
        }[variant]


def _patch_registry(monkeypatch) -> None:
    registration = _DummyRegistration()
    monkeypatch.setattr(factory_module, "get_material_family", lambda family: registration)


def _training_data() -> tuple[Tensor, Tensor]:
    train_X = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 1.0],
            [0.4, 1.0, 0.0],
            [0.4, 1.0, 1.0],
            [0.8, 0.0, 0.0],
            [0.8, 0.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9], [1.1]], dtype=torch.double)
    return train_X, train_Y


def test_mixed_spec_resolves_mixed_variant(monkeypatch) -> None:
    _patch_registry(monkeypatch)

    spec = RegisteredMaterialExplicitTaskSpec(
        "dummy",
        kind="deep-kernel",
        input_mode="mixed",
    )

    assert spec.kind == "dkl"
    assert spec.input_mode == "mixed"
    assert spec.base_variant == "mixed_dkl"


def test_mixed_adapter_preserves_categorical_coordinate() -> None:
    adapter = RegisteredMixedMaterialFeatureExtractor(
        input_transform=_IdentityTransform(),
        feature_extractor=_ContinuousExtractor(trainable=False),
        scale_to_bounds=nn.Identity(),
        ord_dims=(0,),
        cat_dims=(1,),
        latent_dim=1,
        preserve_input_layout=True,
        kernel_ord_dims=(0,),
        kernel_cat_dims=(1,),
    )
    X = torch.tensor([[2.0, 3.0]], dtype=torch.double)

    transformed = adapter(X)

    assert transformed.shape == torch.Size([1, 2])
    assert torch.allclose(transformed, X)


def test_mixed_factory_returns_multitask_gp_and_preserves_kernel(monkeypatch) -> None:
    _patch_registry(monkeypatch)
    train_X, train_Y = _training_data()

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        input_mode="mixed",
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
        cat_dims=[1],
    )

    assert isinstance(model, MultiTaskGP)
    assert model.material_input_mode == "mixed"
    assert _DummyMixedGP.last_train_X is not None
    assert torch.allclose(_DummyMixedGP.last_train_X, train_X[:, :2])
    # MultiTaskGP composes the provided data covariance with its task kernel.
    # The transferred mixed material covariance is therefore the first factor.
    assert isinstance(model.covar_module, ProductKernel)
    assert isinstance(model.covar_module.kernels[0], ScaleKernel)
    assert isinstance(
        model.input_transform.feature_extractor,
        RegisteredMixedMaterialFeatureExtractor,
    )


def test_mixed_factory_keeps_categorical_values_out_of_continuous_encoder(monkeypatch) -> None:
    _patch_registry(monkeypatch)
    train_X, train_Y = _training_data()
    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        input_mode="mixed",
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
        cat_dims=[1],
    )

    transformed = model.input_transform(train_X)

    assert transformed.shape[-1] == 3
    assert torch.equal(transformed[:, 1], train_X[:, 1])
    assert torch.equal(transformed[:, -1], train_X[:, -1])


def test_mixed_dkl_keeps_trainable_representation_parameters(monkeypatch) -> None:
    _patch_registry(monkeypatch)
    train_X, train_Y = _training_data()

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        kind="dkl",
        input_mode="mixed",
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
        cat_dims=[1],
    )

    params = list(model.input_transform.feature_extractor.parameters())
    assert params
    assert any(parameter.requires_grad for parameter in params)


def test_mixed_factory_supports_known_noise(monkeypatch) -> None:
    _patch_registry(monkeypatch)
    train_X, train_Y = _training_data()

    model = create_registered_material_explicit_task_surrogate(
        "dummy",
        train_X,
        train_Y,
        torch.full_like(train_Y, 0.01),
        input_mode="mixed",
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
        cat_dims=[1],
    )

    assert isinstance(model, MultiTaskGP)
