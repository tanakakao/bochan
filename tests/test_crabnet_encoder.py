from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import bochan.composition.encoders.crabnet as crabnet_module
from bochan.composition import CrabNetEncoder, MaterialEncoder


class FakeCrabNet(nn.Module):
    """Small differentiable stand-in for the optional upstream encoder."""

    offset: Tensor

    def __init__(self, d_model: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.arange(1, d_model + 1, dtype=torch.float32))
        self.register_buffer("offset", torch.zeros(d_model))

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        del element_ids
        return fractions.unsqueeze(-1) * self.scale + self.offset


class UndeclaredWidthEncoder(nn.Module):
    """Injected encoder that requires an explicit adapter output width."""

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        del element_ids
        return fractions.unsqueeze(-1)


def _inputs(dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    element_ids = torch.tensor([[26, 8, 0], [13, 8, 0]], dtype=torch.long)
    fractions = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]], dtype=dtype)
    return element_ids, fractions


def test_public_composition_import_does_not_import_optional_crabnet() -> None:
    assert CrabNetEncoder is crabnet_module.CrabNetEncoder
    assert "crabnet.kingcrab" not in sys.modules


def test_constructing_upstream_encoder_has_clear_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_crabnet(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(crabnet_module, "import_module", missing_crabnet)

    with pytest.raises(ImportError, match=r"bochan\[materials\]"):
        CrabNetEncoder(
            d_model=8,
            num_layers=1,
            num_heads=2,
            dim_feedforward=16,
            pe_resolution=32,
            ple_resolution=32,
        )


def test_injected_encoder_returns_element_and_pooled_embeddings() -> None:
    encoder = CrabNetEncoder(FakeCrabNet())
    element_ids, fractions = _inputs()

    element_embeddings = encoder.element_embeddings(element_ids, fractions)
    pooled = encoder(element_ids, fractions)
    expected_elements = fractions.unsqueeze(-1) * torch.tensor([1.0, 2.0, 3.0])
    expected_pooled = expected_elements.sum(dim=-2) / 2

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.output_dim == 3
    assert encoder.initialization == "injected"
    assert encoder.checkpoint_path is None
    assert element_embeddings.shape == torch.Size([2, 3, 3])
    assert torch.allclose(element_embeddings, expected_elements)
    assert torch.allclose(pooled, expected_pooled)


def test_encoder_preserves_batch_and_q_dimensions_and_input_gradients() -> None:
    encoder = CrabNetEncoder(FakeCrabNet()).double()
    element_ids = torch.tensor(
        [
            [[26, 8, 0], [13, 8, 0]],
            [[6, 8, 0], [14, 8, 0]],
        ],
        dtype=torch.long,
    )
    fractions = torch.tensor(
        [
            [[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]],
            [[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    element_embeddings = encoder.element_embeddings(element_ids, fractions)
    pooled = encoder(element_ids, fractions)
    pooled.square().sum().backward()

    assert element_embeddings.shape == torch.Size([2, 2, 3, 3])
    assert pooled.shape == torch.Size([2, 2, 3])
    assert fractions.grad is not None
    assert torch.isfinite(fractions.grad).all()


def test_encoder_requires_an_explicit_width_when_injection_cannot_declare_it() -> None:
    with pytest.raises(ValueError, match="output_dim is required"):
        CrabNetEncoder(UndeclaredWidthEncoder())

    encoder = CrabNetEncoder(UndeclaredWidthEncoder(), output_dim=1)
    assert encoder.output_dim == 1


def test_encoder_rejects_a_conflicting_explicit_width() -> None:
    with pytest.raises(ValueError, match="does not match"):
        CrabNetEncoder(FakeCrabNet(d_model=3), output_dim=4)


def test_encoder_validates_composition_inputs() -> None:
    encoder = CrabNetEncoder(FakeCrabNet())
    element_ids, fractions = _inputs()

    with pytest.raises(ValueError, match="identical shapes"):
        encoder(element_ids[:, :2], fractions)
    with pytest.raises(TypeError, match="torch.long"):
        encoder(element_ids.to(dtype=torch.int32), fractions)
    with pytest.raises(TypeError, match="floating-point"):
        encoder(element_ids, fractions.to(dtype=torch.int64))
    with pytest.raises(ValueError, match="sum to one"):
        encoder(element_ids, fractions * 0.5)

    invalid_padding = fractions.clone()
    invalid_padding[:, -1] = 0.1
    with pytest.raises(ValueError, match="Padding slots"):
        encoder(element_ids, invalid_padding)

    zero_fraction = fractions.clone()
    zero_fraction[0] = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive fractions"):
        encoder(element_ids, zero_fraction)

    no_elements = torch.zeros_like(element_ids)
    with pytest.raises(ValueError, match="at least one non-padding"):
        encoder(no_elements, fractions)


def test_to_dtype_controls_encoder_and_input_contract() -> None:
    encoder = CrabNetEncoder(FakeCrabNet()).to(device="cpu", dtype=torch.double)
    element_ids, fractions = _inputs(dtype=torch.double)

    embeddings = encoder(element_ids, fractions)

    assert embeddings.device.type == "cpu"
    assert embeddings.dtype == torch.double
    assert encoder.encoder.scale.dtype == torch.double
    with pytest.raises(ValueError, match="same dtype as the encoder"):
        encoder(element_ids, fractions.float())


def test_upstream_style_checkpoint_loads_only_encoder_weights(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "crabnet.pth"
    torch.save(
        {
            "weights": {
                "encoder.scale": torch.tensor([2.0, 3.0, 4.0]),
                "encoder.offset": torch.tensor([0.1, 0.2, 0.3]),
                "output_nn.fc_out.weight": torch.ones(1, 3),
            },
            "scaler_state": {"mean": torch.tensor(0.0)},
            "model_name": "test",
        },
        checkpoint_path,
    )

    encoder = CrabNetEncoder(FakeCrabNet(), checkpoint=checkpoint_path)

    assert encoder.initialization == "checkpoint"
    assert encoder.checkpoint_path == str(checkpoint_path)
    assert torch.equal(encoder.encoder.scale, torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(encoder.encoder.offset, torch.tensor([0.1, 0.2, 0.3]))


def test_checkpoint_supports_adapter_state_dict_and_non_strict_loading() -> None:
    source = CrabNetEncoder(FakeCrabNet())
    with torch.no_grad():
        source.encoder.scale.fill_(7)
        source.encoder.offset.fill_(0.5)

    restored = CrabNetEncoder(FakeCrabNet(), checkpoint=source.state_dict())
    partial = CrabNetEncoder(
        FakeCrabNet(),
        checkpoint={"state_dict": {"model.encoder.scale": torch.full((3,), 9.0)}},
        strict_checkpoint=False,
    )

    assert torch.equal(restored.encoder.scale, source.encoder.scale)
    assert torch.equal(restored.encoder.offset, source.encoder.offset)
    assert torch.equal(partial.encoder.scale, torch.full((3,), 9.0))
    assert torch.equal(partial.encoder.offset, torch.zeros(3))


def test_checkpoint_rejects_missing_or_unrelated_encoder_weights() -> None:
    with pytest.raises(ValueError, match="no weights matching"):
        CrabNetEncoder(FakeCrabNet(), checkpoint={"weights": {"output_nn.weight": torch.ones(1)}})

    with pytest.raises(RuntimeError, match="Missing key"):
        CrabNetEncoder(FakeCrabNet(), checkpoint={"weights": {"encoder.scale": torch.ones(3)}})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"d_model": 7}, "d_model must be even"),
        ({"d_model": 10, "num_heads": 4}, "divisible by num_heads"),
        ({"dropout": 1.0}, "interval"),
        ({"pe_resolution": 0}, "pe_resolution must be a positive integer"),
    ],
)
def test_upstream_configuration_is_validated_before_optional_import(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        CrabNetEncoder(**kwargs)  # type: ignore[arg-type]


def test_real_crabnet_encoder_runs_on_cpu_when_materials_extra_is_installed() -> None:
    pytest.importorskip("crabnet.kingcrab")
    encoder = CrabNetEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        pe_resolution=32,
        ple_resolution=32,
    ).double()
    encoder.eval()
    element_ids, fractions = _inputs(dtype=torch.double)
    fractions.requires_grad_(True)

    element_embeddings = encoder.element_embeddings(element_ids, fractions)
    pooled = encoder(element_ids, fractions)
    pooled.sum().backward()

    assert encoder.initialization == "random"
    assert element_embeddings.shape == torch.Size([2, 3, 8])
    assert pooled.shape == torch.Size([2, 8])
    assert pooled.dtype == torch.double
    assert fractions.grad is not None
    assert torch.isfinite(fractions.grad).all()
