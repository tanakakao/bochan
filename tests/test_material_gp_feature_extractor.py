from __future__ import annotations

import torch
from torch import Tensor, nn

from bochan.composition import MaterialEncoder
from bochan.models.regression.gaussian.deep import (
    CompositionMaterialInputTransform,
    CrabNetInputTransform,
    MaterialGPFeatureExtractor,
)


class ToyCompositionEncoder(MaterialEncoder):
    """Small composition encoder with independently trainable submodules."""

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.embedding = nn.Linear(2, output_dim)
        self.output_layer = nn.Linear(output_dim, output_dim)
        self.last_element_ids: Tensor | None = None
        self.last_fractions: Tensor | None = None

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        self.last_element_ids = element_ids.detach().clone()
        self.last_fractions = fractions.detach().clone()
        elemental_inputs = torch.stack(
            (
                element_ids.to(dtype=fractions.dtype) / 100.0,
                fractions,
            ),
            dim=-1,
        )
        elemental_features = torch.tanh(self.embedding(elemental_inputs))
        pooled = (elemental_features * fractions.unsqueeze(-1)).sum(dim=-2)
        return self.output_layer(pooled)


def _extractor(*, zero_tolerance: float = 0.0) -> MaterialGPFeatureExtractor:
    return MaterialGPFeatureExtractor(
        material_encoder=ToyCompositionEncoder(),
        element_ids=torch.tensor([26, 27, 28], dtype=torch.long),
        process_dim=2,
        latent_dim=5,
        projection=nn.Identity(),
        zero_tolerance=zero_tolerance,
    ).double()


def test_composition_transform_is_public_and_preserves_q_batch_gradients() -> None:
    transform = CompositionMaterialInputTransform(
        input_dim=4,
        composition_indices=[0, 2],
        n_components=3,
        method="ilr",
        process_bounds=torch.tensor(
            [[800.0, 1.0], [1200.0, 5.0]],
            dtype=torch.double,
        ),
    ).double()
    raw = torch.tensor(
        [
            [[[0.2, 850.0, -0.1, 2.0], [-0.3, 900.0, 0.4, 3.0]]],
            [[[0.1, 1000.0, 0.2, 4.0], [0.5, 1100.0, -0.2, 2.5]]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    packed = transform(raw)
    weighted = packed * packed.new_tensor([1.0, 2.0, 4.0, 0.5, 0.75])
    (gradient,) = torch.autograd.grad(weighted.sum(), raw)

    assert issubclass(CrabNetInputTransform, CompositionMaterialInputTransform)
    assert packed.shape == torch.Size([2, 1, 2, 5])
    torch.testing.assert_close(
        packed[..., :3].sum(dim=-1),
        torch.ones((2, 1, 2), dtype=torch.double),
    )
    assert torch.isfinite(gradient).all()
    assert gradient[..., [0, 2]].abs().sum() > 0
    assert gradient[..., [1, 3]].abs().sum() > 0


def test_material_extractor_preserves_q_batches_and_all_input_gradients() -> None:
    torch.manual_seed(0)
    extractor = _extractor()
    X = torch.tensor(
        [
            [[0.60, 0.30, 0.10, 900.0, 1.0], [0.40, 0.40, 0.20, 950.0, 2.0]],
            [[0.30, 0.50, 0.20, 1000.0, 3.0], [0.20, 0.30, 0.50, 1050.0, 4.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    features = extractor(X)
    (gradient,) = torch.autograd.grad(features.square().sum(), X)

    assert features.shape == torch.Size([2, 2, 5])
    assert torch.isfinite(features).all()
    assert torch.isfinite(gradient).all()
    assert gradient[..., :3].abs().sum() > 0
    assert gradient[..., 3:].abs().sum() > 0


def test_material_extractor_pads_and_renormalizes_near_zero_fractions() -> None:
    extractor = _extractor(zero_tolerance=0.01)
    X = torch.tensor(
        [[0.60, 0.399, 0.001, 900.0, 1.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    features = extractor(X)
    (gradient,) = torch.autograd.grad(features.sum(), X)
    encoder = extractor.material_encoder

    assert isinstance(encoder, ToyCompositionEncoder)
    assert encoder.last_element_ids is not None
    assert encoder.last_fractions is not None
    torch.testing.assert_close(
        encoder.last_element_ids,
        torch.tensor([[26, 27, 0]], dtype=torch.long),
    )
    torch.testing.assert_close(
        encoder.last_fractions,
        torch.tensor([[0.60 / 0.999, 0.399 / 0.999, 0.0]], dtype=torch.double),
    )
    assert gradient[0, :2].abs().sum() > 0
    assert gradient[0, 2] == 0
    assert gradient[0, 3:].abs().sum() > 0


def test_material_extractor_applies_frozen_partial_and_full_mode_policies() -> None:
    extractor = _extractor()
    encoder = extractor.material_encoder
    assert isinstance(encoder, ToyCompositionEncoder)

    assert not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    extractor._configure_encoder_training("frozen")
    extractor.train()
    assert not encoder.training
    assert not encoder.embedding.training
    assert not encoder.output_layer.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    for parameter in encoder.output_layer.parameters():
        parameter.requires_grad_(True)
    extractor._configure_encoder_training("partial", (encoder.output_layer,))
    extractor.train()
    assert not encoder.training
    assert not encoder.embedding.training
    assert encoder.output_layer.training
    assert not any(parameter.requires_grad for parameter in encoder.embedding.parameters())
    assert all(parameter.requires_grad for parameter in encoder.output_layer.parameters())
    extractor.eval()
    assert not encoder.output_layer.training

    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    extractor._configure_encoder_training("full")
    extractor.train()
    assert encoder.training
    assert encoder.embedding.training
    assert encoder.output_layer.training
    assert all(parameter.requires_grad for parameter in encoder.parameters())
