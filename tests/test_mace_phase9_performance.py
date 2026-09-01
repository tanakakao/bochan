"""Phase-9 performance, device, and reproducibility coverage for MACE."""

from __future__ import annotations

import copy
from importlib.metadata import version
from math import ceil
from typing import Any

import pytest
import torch
from torch import Tensor

pytest.importorskip("mace")

from bochan.composition import MACEEncoder
from bochan.models.regression.gaussian.deep import MACEDKLModel
from bochan.serving.fastapi.services.mace_tabular import build_mace_fit_response
from tests.test_mace_phase7_integration import (
    CountingBatchBuilder,
    FakeMACE,
    _catalog,
    _single_output_optimizer,
    _structure,
)


class ForwardCountingFakeMACE(FakeMACE):
    """Record raw model calls while retaining Phase-7 descriptor semantics."""

    def __init__(self, width: int = 2) -> None:
        super().__init__(width=width)
        self.forward_calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def _fake_native_batch_builder(
    structures: list[dict[str, object]],
) -> dict[str, Tensor]:
    builder = CountingBatchBuilder()
    positions: list[Tensor] = []
    membership: list[Tensor] = []
    for index, structure in enumerate(structures):
        batch = builder(structure)
        position = batch["positions"]
        positions.append(position)
        membership.append(
            torch.full((position.shape[0],), index, dtype=torch.long)
        )
    return {
        "positions": torch.cat(positions, dim=0),
        "batch": torch.cat(membership, dim=0),
    }


def test_mace_native_batching_matches_sequential_representations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    raw_model = ForwardCountingFakeMACE()
    sequential_model = copy.deepcopy(raw_model)
    structures = [_structure(5.20 + 0.03 * index) for index in range(5)]

    sequential = MACEEncoder(
        sequential_model,
        batch_builder=CountingBatchBuilder(),
    )
    expected = sequential(structures)

    batched = MACEEncoder(raw_model, batch_size=2)
    monkeypatch.setattr(
        batched,
        "_default_batch_many",
        lambda chunk: _fake_native_batch_builder(list(chunk)),
    )
    actual = batched(structures)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert raw_model.forward_calls == ceil(len(structures) / batched.batch_size)
    assert batched.native_batching_enabled is True


def test_custom_batch_builder_is_coerced_to_native_mace_dtype() -> None:
    encoder = MACEEncoder(
        FakeMACE().double(),
        batch_builder=CountingBatchBuilder(),
    )

    features = encoder([_structure(5.43)])

    assert features.dtype == torch.float64
    assert next(encoder.encoder.parameters()).dtype == torch.float64
    assert torch.isfinite(features).all()


def test_trainable_mace_deduplicates_repeated_structure_encodes() -> None:
    builder = CountingBatchBuilder()
    encoder = MACEEncoder(FakeMACE(), batch_builder=builder)
    structures = tuple(_catalog(3).values())
    train_X = torch.tensor(
        [[0.0, 0.10], [1.0, 0.30], [2.0, 0.50]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    model = MACEDKLModel(
        train_X,
        train_Y,
        structures=structures,
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
    )
    builder.calls = 0

    repeated_X = torch.tensor(
        [[0.0, 0.1], [0.0, 0.2], [1.0, 0.3], [0.0, 0.4], [1.0, 0.5]],
        dtype=torch.double,
    )
    features = model.mace_feature_extractor._material_features(repeated_X)  # noqa: SLF001

    assert builder.calls == 2
    assert features.shape == (5, model.material_encoder.output_dim)
    assert torch.allclose(features[0], features[1])
    assert torch.allclose(features[0], features[3])
    assert torch.allclose(features[2], features[4])

    features.sum().backward()
    assert model.material_encoder.encoder.products[-1].scale.grad is not None


def test_mace_fit_metadata_records_runtime_reproducibility_contract() -> None:
    optimizer = _single_output_optimizer(3)
    metadata = build_mace_fit_response("phase9", optimizer).metadata["mace"]

    assert metadata["mace_torch_version"] == version("mace-torch")
    assert metadata["batch_size"] == 16
    assert metadata["native_batching_enabled"] is False
    assert metadata["encoder_device"] == "cpu"
    assert metadata["encoder_dtype"] == "float32"
    assert metadata["output_models"][0]["mace_torch_version"] == version("mace-torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_custom_mace_batch_builder_follows_cuda_device() -> None:
    encoder = MACEEncoder(
        FakeMACE().cuda(),
        batch_builder=CountingBatchBuilder(),
    ).cuda()

    features = encoder([_structure(5.43), _structure(5.55)])

    assert features.is_cuda
    assert next(encoder.encoder.parameters()).is_cuda
    assert torch.isfinite(features).all()


def test_real_pretrained_mace_native_batch_matches_single_structure_batches() -> None:
    encoder = MACEEncoder(model_name="medium-mpa-0", batch_size=3)
    structures = list(_catalog(3).values())

    batched = encoder(structures).detach()
    encoder._batch_size = 1  # noqa: SLF001 - verify the same raw model with another chunk size
    single_structure_batches = encoder(structures).detach()

    assert batched.shape == single_structure_batches.shape
    assert torch.allclose(batched, single_structure_batches, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(batched).all()
