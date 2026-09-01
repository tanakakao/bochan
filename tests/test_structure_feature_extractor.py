from __future__ import annotations

import torch
from torch import Tensor, nn

from bochan.composition import MaterialEncoder
from bochan.models.regression.gaussian.deep.structure import (
    _StructureGPFeatureExtractor,
    _validate_structure_model_inputs,
)


class CountingStructureEncoder(MaterialEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 3, bias=False, dtype=torch.double)
        self.calls = 0

    @property
    def output_dim(self) -> int:
        return 3

    def forward(self, structures: list[Tensor]) -> Tensor:
        self.calls += 1
        values = torch.stack(structures)
        return self.projection(values)


def _extractor() -> tuple[_StructureGPFeatureExtractor, CountingStructureEncoder]:
    encoder = CountingStructureEncoder()
    structures = [
        torch.tensor([1.0, 0.2], dtype=torch.double),
        torch.tensor([0.3, 1.0], dtype=torch.double),
    ]
    extractor = _StructureGPFeatureExtractor(
        material_encoder=encoder,
        structure_inputs=structures,
        process_dim=1,
        latent_dim=2,
        fusion="concat",
        projection=None,
        encoder_name="TEST",
    ).double()
    return extractor, encoder


def test_frozen_structure_features_are_cached_and_reused() -> None:
    extractor, encoder = _extractor()
    X = torch.tensor(
        [[0.0, 0.2], [1.0, 0.8], [0.0, 0.4]],
        dtype=torch.double,
    )

    first = extractor(X)
    second = extractor(X)

    assert first.shape == torch.Size([3, 2])
    assert torch.allclose(first, second)
    assert encoder.calls == 1
    assert extractor.material_feature_cache is not None


def test_frozen_structure_cache_is_invalidated_after_encoder_mutation() -> None:
    extractor, encoder = _extractor()
    X = torch.tensor([[0.0, 0.2], [1.0, 0.8]], dtype=torch.double)

    extractor(X)
    assert encoder.calls == 1

    with torch.no_grad():
        encoder.projection.weight.add_(0.1)

    extractor(X)

    assert encoder.calls == 2


def test_trainable_structure_encoder_bypasses_frozen_cache() -> None:
    extractor, encoder = _extractor()
    X = torch.tensor([[0.0, 0.2], [1.0, 0.8]], dtype=torch.double)

    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    extractor._configure_encoder_training("full")

    extractor(X)
    extractor(X)

    assert encoder.calls == 2
    assert extractor.material_feature_cache is None


def test_structure_input_validation_rejects_fractional_selector() -> None:
    X = torch.tensor([[0.5, 1.0]], dtype=torch.double)

    try:
        _validate_structure_model_inputs(
            X,
            num_structures=2,
            input_dim=2,
            encoder_name="TEST",
        )
    except ValueError as error:
        assert "integer-valued structure indices" in str(error)
    else:
        raise AssertionError("fractional structure selectors must be rejected")
