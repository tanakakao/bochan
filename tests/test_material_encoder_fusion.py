from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from bochan.composition import (
    ConcatFusion,
    MaterialEncoder,
    MaterialProcessFusion,
    build_material_process_fusion,
)


class LinearMaterialEncoder(MaterialEncoder):
    """Small concrete encoder used to exercise the public contract."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self._output_dim = int(output_dim)
        self.projection = nn.Linear(input_dim, output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, composition: Tensor) -> Tensor:
        return self.projection(composition)


class DifferenceFusion(MaterialProcessFusion):
    """Custom fusion proving that callers can inject future strategies."""

    @property
    def output_dim(self) -> int:
        return 2

    def forward(
        self,
        material_features: Tensor,
        process_features: Tensor | None = None,
    ) -> Tensor:
        if process_features is None:
            raise ValueError("process_features is required.")
        return material_features - process_features


def test_material_encoder_contract_exposes_output_dim_and_gradients() -> None:
    encoder = LinearMaterialEncoder(input_dim=3, output_dim=2).double()
    composition = torch.rand(4, 3, dtype=torch.double, requires_grad=True)

    material_features = encoder(composition)
    material_features.sum().backward()

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.output_dim == 2
    assert material_features.shape == torch.Size([4, 2])
    assert composition.grad is not None
    assert torch.isfinite(composition.grad).all()


def test_material_encoder_requires_output_and_forward_contracts() -> None:
    with pytest.raises(TypeError, match="abstract"):
        MaterialEncoder()


def test_concat_fusion_preserves_batch_and_q_dimensions() -> None:
    material_features = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    process_features = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
    fusion = ConcatFusion(material_dim=4, process_dim=2)

    fused = fusion(material_features, process_features)
    fused.square().sum().backward()

    assert fusion.output_dim == 6
    assert fused.shape == torch.Size([2, 3, 6])
    assert torch.equal(fused[..., :4], material_features.detach())
    assert torch.equal(fused[..., 4:], process_features.detach())
    assert material_features.grad is not None
    assert process_features.grad is not None


def test_concat_fusion_supports_composition_only_input() -> None:
    material_features = torch.randn(5, 4)
    fusion = ConcatFusion(material_dim=4)

    fused = fusion(material_features)

    assert fusion.output_dim == 4
    assert fused is material_features


@pytest.mark.parametrize(
    ("material_dim", "process_dim", "match"),
    [
        (0, 2, "material_dim must be positive"),
        (3, -1, "process_dim must be non-negative"),
    ],
)
def test_concat_fusion_validates_configured_dimensions(
    material_dim: int,
    process_dim: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ConcatFusion(material_dim=material_dim, process_dim=process_dim)


def test_concat_fusion_validates_input_contract() -> None:
    fusion = ConcatFusion(material_dim=3, process_dim=2)

    with pytest.raises(ValueError, match="process_features is required"):
        fusion(torch.randn(4, 3))
    with pytest.raises(ValueError, match="material_features width"):
        fusion(torch.randn(4, 4), torch.randn(4, 2))
    with pytest.raises(ValueError, match="identical leading dimensions"):
        fusion(torch.randn(4, 3), torch.randn(5, 2))
    with pytest.raises(ValueError, match="same dtype"):
        fusion(
            torch.randn(4, 3, dtype=torch.float32),
            torch.randn(4, 2, dtype=torch.float64),
        )


def test_fusion_builder_supports_concat_and_custom_modules() -> None:
    concat = build_material_process_fusion(
        material_dim=5,
        process_dim=3,
    )
    custom = DifferenceFusion()

    assert isinstance(concat, ConcatFusion)
    assert concat.output_dim == 8
    assert build_material_process_fusion(
        custom,
        material_dim=2,
        process_dim=2,
    ) is custom


def test_fusion_builder_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="fusion must be 'concat'"):
        build_material_process_fusion(
            "attention",  # type: ignore[arg-type]
            material_dim=4,
            process_dim=2,
        )
