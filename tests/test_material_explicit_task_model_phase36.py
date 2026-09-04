from __future__ import annotations

import pytest
import torch
from botorch.models import MultiTaskGP
from torch import nn

from bochan.models.regression.gaussian.materials import (
    MaterialExplicitTaskFeatureTransform,
    MaterialExplicitTaskSpec,
    build_material_explicit_task_surrogate,
)


class DummyMaterialEncoder(nn.Module):
    output_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(2, dtype=torch.double))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[..., :2] * self.scale


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.0, 0.1, 0.0],
            [0.0, 0.1, 1.0],
            [0.4, 0.6, 0.0],
            [0.4, 0.6, 1.0],
            [0.8, 0.2, 0.0],
            [0.8, 0.2, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.1], [0.3], [0.5], [0.8], [0.9], [1.1]], dtype=torch.double)
    return train_X, train_Y


def test_feature_transform_encodes_material_columns_and_appends_task() -> None:
    encoder = DummyMaterialEncoder()
    transform = MaterialExplicitTaskFeatureTransform(encoder, task_feature=1)
    X = torch.tensor([[2.0, 1.0, 3.0]], dtype=torch.double)

    transformed = transform(X)

    assert transformed.shape == torch.Size([1, 3])
    assert torch.equal(transformed[:, :2], torch.tensor([[2.0, 3.0]], dtype=torch.double))
    assert torch.equal(transformed[:, -1], torch.tensor([1.0], dtype=torch.double))


def test_feature_transform_preserves_q_batch_dimensions() -> None:
    transform = MaterialExplicitTaskFeatureTransform(DummyMaterialEncoder())
    X = torch.rand(3, 4, 3, dtype=torch.double)
    X[..., -1] = 1.0

    transformed = transform(X)

    assert transformed.shape == torch.Size([3, 4, 3])


def test_build_material_explicit_task_surrogate_returns_multitask_gp() -> None:
    train_X, train_Y = _training_data()
    encoder = DummyMaterialEncoder()

    model = build_material_explicit_task_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        feature_extractor=encoder,
        task_spec=MaterialExplicitTaskSpec(all_tasks=(0, 1)),
    )

    assert isinstance(model, MultiTaskGP)
    assert isinstance(model.input_transform, MaterialExplicitTaskFeatureTransform)
    assert model.num_outputs == 2
    assert model.input_transform.feature_extractor is encoder


def test_builder_keeps_encoder_parameters_registered_for_dkl_training() -> None:
    train_X, train_Y = _training_data()
    model = build_material_explicit_task_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        feature_extractor=DummyMaterialEncoder(),
    )

    parameter_names = dict(model.named_parameters())
    assert "input_transform.feature_extractor.scale" in parameter_names


def test_builder_supports_known_noise() -> None:
    train_X, train_Y = _training_data()
    train_Yvar = torch.full_like(train_Y, 0.01)

    model = build_material_explicit_task_surrogate(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        feature_extractor=DummyMaterialEncoder(),
    )

    assert isinstance(model, MultiTaskGP)


def test_builder_accepts_one_dimensional_targets() -> None:
    train_X, train_Y = _training_data()

    model = build_material_explicit_task_surrogate(
        train_X=train_X,
        train_Y=train_Y.squeeze(-1),
        feature_extractor=DummyMaterialEncoder(),
    )

    assert isinstance(model, MultiTaskGP)


def test_builder_rejects_declared_tasks_that_do_not_cover_observations() -> None:
    train_X, train_Y = _training_data()

    with pytest.raises(ValueError, match="subset"):
        build_material_explicit_task_surrogate(
            train_X=train_X,
            train_Y=train_Y,
            feature_extractor=DummyMaterialEncoder(),
            task_spec=MaterialExplicitTaskSpec(all_tasks=(0,)),
        )


def test_feature_transform_rejects_wrong_latent_width() -> None:
    class WrongEncoder(nn.Module):
        output_dim = 3

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return X[..., :2]

    transform = MaterialExplicitTaskFeatureTransform(WrongEncoder())

    with pytest.raises(ValueError, match="latent_dim"):
        transform(torch.zeros(2, 3))
