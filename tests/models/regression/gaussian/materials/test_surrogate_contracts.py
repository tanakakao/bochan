from __future__ import annotations

import pytest
import torch
from torch import nn

from bochan.models.regression.gaussian.deep.deepkernel_configurable import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)
from bochan.models.regression.gaussian.materials import (
    MaterialSurrogateSpec,
    build_material_gaussian_surrogate,
    resolve_material_latent_dim,
)


class _DummyFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.projection(X)


def test_surrogate_spec_validates_kind_and_latent_dim() -> None:
    assert MaterialSurrogateSpec(kind="gp", latent_dim=3).kind == "gp"
    assert MaterialSurrogateSpec(kind="dkl", latent_dim=3).kind == "dkl"

    with pytest.raises(ValueError, match="kind"):
        MaterialSurrogateSpec(kind="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="latent_dim"):
        MaterialSurrogateSpec(latent_dim=0)


def test_resolve_material_latent_dim_uses_feature_extractor_contract() -> None:
    extractor = _DummyFeatureExtractor(input_dim=2, output_dim=4)
    assert resolve_material_latent_dim(extractor) == 4
    assert resolve_material_latent_dim(extractor, 4) == 4

    with pytest.raises(ValueError, match="does not match"):
        resolve_material_latent_dim(extractor, 3)


def test_non_mixed_builder_reuses_configurable_observation_aware_backend() -> None:
    train_X = torch.tensor([[0.0, 0.2], [0.5, 0.4], [1.0, 0.8]], dtype=torch.double)
    train_Y = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.double)
    train_Yvar = torch.full_like(train_Y, 0.01)
    extractor = _DummyFeatureExtractor(input_dim=2, output_dim=2).double()

    model = build_material_gaussian_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        feature_extractor=extractor,
        spec=MaterialSurrogateSpec(kind="gp", latent_dim=2),
        input_transform=None,
        outcome_transform=None,
    )

    assert isinstance(model, DeepKernelGaussianGPModel)
    assert model.deepkernel.feature_extractor is extractor
    assert model.latent_dim == 2
    assert model.train_Yvar is not None


def test_dkl_kind_uses_same_gaussian_backend() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.double)
    extractor = _DummyFeatureExtractor(input_dim=1, output_dim=2).double()

    model = build_material_gaussian_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        feature_extractor=extractor,
        spec=MaterialSurrogateSpec(kind="dkl", latent_dim=2),
        input_transform=None,
        outcome_transform=None,
    )

    assert isinstance(model, DeepKernelGaussianGPModel)
    assert model.deepkernel.feature_extractor is extractor


def test_mixed_builder_preserves_categorical_kernel_boundary() -> None:
    train_X = torch.tensor(
        [[0.0, 0.1], [1.0, 0.4], [0.0, 0.9]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.double)
    extractor = _DummyFeatureExtractor(input_dim=1, output_dim=2).double()

    model = build_material_gaussian_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        feature_extractor=extractor,
        spec=MaterialSurrogateSpec(kind="gp", mixed=True, latent_dim=2),
        cat_dims=(0,),
        input_transform=None,
        outcome_transform=None,
    )

    assert isinstance(model, DeepKernelGaussianMixedGPModel)
    assert model.cat_dims == [0]
    assert model.ord_dims == [1]
    assert model.deepkernel.feature_extractor is extractor


def test_mixed_and_non_mixed_category_contract_is_explicit() -> None:
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    extractor = _DummyFeatureExtractor(input_dim=1, output_dim=1).double()

    with pytest.raises(ValueError, match="cat_dims must not be empty"):
        build_material_gaussian_surrogate(
            train_X=train_X,
            train_Y=train_Y,
            feature_extractor=extractor,
            spec=MaterialSurrogateSpec(mixed=True),
            input_transform=None,
            outcome_transform=None,
        )

    with pytest.raises(ValueError, match="cat_dims must be empty"):
        build_material_gaussian_surrogate(
            train_X=train_X,
            train_Y=train_Y,
            feature_extractor=extractor,
            spec=MaterialSurrogateSpec(mixed=False),
            cat_dims=(0,),
            input_transform=None,
            outcome_transform=None,
        )
