from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import bochan.composition.encoders.roost as roost_module
from bochan.composition import MaterialEncoder, RoostEncoder, RoostGraph, build_roost_graph


class FakeMaterialNetwork(nn.Module):
    """Small weighted pooling network with upstream-style state names."""

    offset: Tensor

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, output_dim + 1, dtype=torch.float32))
        self.register_buffer("offset", torch.zeros(output_dim))

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del self_idx, nbr_idx
        node_features = elem_fea * elem_weights * self.scale
        n_materials = int(cry_elem_idx[-1].item()) + 1
        pooled = elem_fea.new_zeros((n_materials, elem_fea.shape[-1]))
        pooled.index_add_(0, cry_elem_idx, node_features)
        return pooled + self.offset


class FakeRoostBackbone(nn.Module):
    """Differentiable stand-in for Aviary's embedding and descriptor."""

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.elem_embedding = nn.Embedding(119, output_dim)
        self.material_nn = FakeMaterialNetwork(output_dim)
        with torch.no_grad():
            values = torch.arange(119, dtype=torch.float32).unsqueeze(-1)
            self.elem_embedding.weight.copy_(values / 100 + torch.arange(output_dim))

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        embedded = self.elem_embedding(elem_fea)
        return self.material_nn(elem_weights, embedded, self_idx, nbr_idx, cry_elem_idx)


class UndeclaredWidthEncoder(nn.Module):
    """Injected graph encoder that requires an explicit output width."""

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del elem_fea, self_idx, nbr_idx
        output = elem_weights.new_zeros((int(cry_elem_idx[-1].item()) + 1, 1))
        output.index_add_(0, cry_elem_idx, elem_weights)
        return output


def _inputs(dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    element_ids = torch.tensor([[26, 8, 0], [13, 8, 0]], dtype=torch.long)
    fractions = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]], dtype=dtype)
    return element_ids, fractions


def test_public_composition_import_does_not_import_optional_aviary() -> None:
    assert RoostEncoder is roost_module.RoostEncoder
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bochan.composition; "
                "assert 'aviary.roost.model' not in sys.modules; "
                "assert 'aviary.utils' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_constructing_upstream_encoder_has_exact_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_aviary(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(roost_module, "import_module", missing_aviary)

    with pytest.raises(ImportError) as error:
        RoostEncoder(
            elem_fea_len=8,
            n_graph=1,
            elem_heads=1,
            elem_gate=(8,),
            elem_msg=(8,),
            cry_heads=1,
            cry_gate=(8,),
            cry_msg=(8,),
        )

    assert str(error.value) == "Roost support requires the optional Roost/materials dependency."


def test_torch_graph_builder_matches_aviary_batch_contract() -> None:
    element_ids = torch.tensor([[26, 8, 0], [13, 8, 1]], dtype=torch.long)
    fractions = torch.tensor([[0.25, 0.75, 0.0], [0.4, 0.5, 0.1]], dtype=torch.double)

    graph = build_roost_graph(element_ids, fractions)

    assert isinstance(graph, RoostGraph)
    assert graph.leading_shape == torch.Size([2])
    assert graph.num_materials == 2
    torch.testing.assert_close(
        graph.elem_weights,
        torch.tensor([[0.25], [0.75], [0.4], [0.5], [0.1]], dtype=torch.double),
    )
    torch.testing.assert_close(graph.elem_fea, torch.tensor([26, 8, 13, 8, 1]))
    torch.testing.assert_close(
        graph.self_idx,
        torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]),
    )
    torch.testing.assert_close(
        graph.nbr_idx,
        torch.tensor([0, 1, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4]),
    )
    torch.testing.assert_close(graph.cry_elem_idx, torch.tensor([0, 0, 1, 1, 1]))


def test_graph_builder_drops_exact_zero_fraction_elements() -> None:
    element_ids = torch.tensor([[26, 8, 14], [13, 8, 0]], dtype=torch.long)
    fractions = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.6, 0.0]])

    graph = build_roost_graph(element_ids, fractions)

    torch.testing.assert_close(graph.elem_fea, torch.tensor([26, 13, 8]))
    torch.testing.assert_close(graph.elem_weights, torch.tensor([[1.0], [0.4], [0.6]]))
    torch.testing.assert_close(graph.self_idx, torch.tensor([0, 1, 1, 2, 2]))
    torch.testing.assert_close(graph.nbr_idx, torch.tensor([0, 1, 2, 1, 2]))
    torch.testing.assert_close(graph.cry_elem_idx, torch.tensor([0, 1, 1]))


def test_torch_graph_builder_matches_current_aviary_collate_contract() -> None:
    data_module = pytest.importorskip("aviary.roost.data")
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {
            "material_id": ["a", "b"],
            "composition": ["FeO3", "Al2O3"],
            "target": [0.0, 1.0],
        }
    )
    dataset = data_module.CompositionData(frame, {"target": "regression"})
    upstream_inputs, *_ = data_module.collate_batch([dataset[0], dataset[1]])
    graph = build_roost_graph(
        torch.tensor([[26, 8], [13, 8]], dtype=torch.long),
        torch.tensor([[0.25, 0.75], [0.4, 0.6]]),
    )

    for actual, expected in zip(graph.model_inputs(), upstream_inputs, strict=True):
        torch.testing.assert_close(actual, expected)


def test_encoder_preserves_q_batches_and_composition_gradients() -> None:
    encoder = RoostEncoder(FakeRoostBackbone()).double()
    element_ids = torch.tensor(
        [
            [[26, 8, 0], [13, 8, 0]],
            [[6, 8, 0], [14, 8, 1]],
        ],
        dtype=torch.long,
    )
    fractions = torch.tensor(
        [
            [[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]],
            [[1.0, 0.0, 0.0], [0.3, 0.6, 0.1]],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    embeddings = encoder(element_ids, fractions)
    embeddings.square().sum().backward()

    assert isinstance(encoder, MaterialEncoder)
    assert encoder.output_dim == 3
    assert encoder.initialization == "injected"
    assert encoder.checkpoint_path is None
    assert embeddings.shape == torch.Size([2, 2, 3])
    assert embeddings.device.type == "cpu"
    assert embeddings.dtype == torch.double
    assert torch.isfinite(embeddings).all()
    assert fractions.grad is not None
    assert torch.isfinite(fractions.grad).all()
    assert fractions.grad[..., :2].abs().sum() > 0


def test_encoder_handles_unbatched_and_zero_fraction_compositions() -> None:
    encoder = RoostEncoder(FakeRoostBackbone())
    element_ids = torch.tensor([26, 8, 14], dtype=torch.long)
    fractions = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)

    embeddings = encoder(element_ids, fractions)
    embeddings.sum().backward()

    assert embeddings.shape == torch.Size([3])
    assert torch.isfinite(embeddings).all()
    assert fractions.grad is not None
    assert torch.isfinite(fractions.grad).all()


def test_encoder_requires_explicit_width_when_injection_does_not_declare_it() -> None:
    with pytest.raises(ValueError, match="output_dim is required"):
        RoostEncoder(UndeclaredWidthEncoder())

    encoder = RoostEncoder(UndeclaredWidthEncoder(), output_dim=1)
    element_ids, fractions = _inputs()

    assert encoder.output_dim == 1
    assert encoder(element_ids, fractions).shape == torch.Size([2, 1])


def test_encoder_rejects_conflicting_explicit_width() -> None:
    with pytest.raises(ValueError, match="does not match"):
        RoostEncoder(FakeRoostBackbone(output_dim=3), output_dim=4)


def test_graph_builder_validates_composition_inputs() -> None:
    element_ids, fractions = _inputs()

    with pytest.raises(ValueError, match="identical shapes"):
        build_roost_graph(element_ids[:, :2], fractions)
    with pytest.raises(TypeError, match="torch.long"):
        build_roost_graph(element_ids.to(dtype=torch.int32), fractions)
    with pytest.raises(TypeError, match="floating-point"):
        build_roost_graph(element_ids, fractions.to(dtype=torch.int64))
    with pytest.raises(ValueError, match="sum to one"):
        build_roost_graph(element_ids, fractions * 0.5)

    invalid_padding = fractions.clone()
    invalid_padding[:, -1] = 0.1
    with pytest.raises(ValueError, match="Padding slots"):
        build_roost_graph(element_ids, invalid_padding)

    no_positive_fractions = torch.zeros_like(fractions)
    with pytest.raises(ValueError, match="positive-fraction"):
        build_roost_graph(element_ids, no_positive_fractions)


def test_to_dtype_controls_encoder_and_input_contract() -> None:
    encoder = RoostEncoder(FakeRoostBackbone()).to(device="cpu", dtype=torch.double)
    element_ids, fractions = _inputs(dtype=torch.double)

    embeddings = encoder(element_ids, fractions)

    assert embeddings.device.type == "cpu"
    assert embeddings.dtype == torch.double
    assert encoder.encoder.material_nn.scale.dtype == torch.double
    with pytest.raises(ValueError, match="same dtype as the encoder"):
        encoder(element_ids, fractions.float())


def test_encoder_validates_installed_element_table_range() -> None:
    encoder = RoostEncoder(FakeRoostBackbone())
    element_ids = torch.tensor([[119, 0]], dtype=torch.long)
    fractions = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="outside the installed Aviary table"):
        encoder(element_ids, fractions)


def test_upstream_style_and_adapter_checkpoints_restore_backbone(tmp_path: Path) -> None:
    source = RoostEncoder(FakeRoostBackbone())
    with torch.no_grad():
        source.encoder.elem_embedding.weight.fill_(0.25)
        source.encoder.material_nn.scale.fill_(7)
        source.encoder.material_nn.offset.fill_(0.5)

    checkpoint_path = tmp_path / "roost.pth"
    torch.save(
        {
            "model_params": {"elem_fea_len": 3},
            "state_dict": {
                **source.encoder.state_dict(),
                "trunk_nn.res_fcs.0.weight": torch.ones(1),
                "output_nns.0.fc_out.weight": torch.ones(1),
            },
        },
        checkpoint_path,
    )

    restored_path = RoostEncoder(FakeRoostBackbone(), checkpoint=checkpoint_path)
    restored_adapter = RoostEncoder(FakeRoostBackbone(), checkpoint=source.state_dict())
    element_ids, fractions = _inputs()

    assert restored_path.initialization == "checkpoint"
    assert restored_path.checkpoint_path == str(checkpoint_path)
    assert restored_adapter.initialization == "checkpoint"
    assert restored_adapter.checkpoint_path is None
    for expected, actual in zip(
        source.encoder.state_dict().values(),
        restored_path.encoder.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(
        source.encoder.state_dict().values(),
        restored_adapter.encoder.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(restored_path(element_ids, fractions), source(element_ids, fractions))
    torch.testing.assert_close(restored_adapter(element_ids, fractions), source(element_ids, fractions))


def test_checkpoint_rejects_partial_unrelated_and_ambiguous_weights() -> None:
    encoder = FakeRoostBackbone()

    with pytest.raises(RuntimeError, match="Missing key"):
        RoostEncoder(
            FakeRoostBackbone(),
            checkpoint={"state_dict": {"material_nn.scale": encoder.material_nn.scale.detach().clone()}},
        )
    with pytest.raises(ValueError, match="no weights matching"):
        RoostEncoder(FakeRoostBackbone(), checkpoint={"state_dict": {"output_nn.weight": torch.ones(1)}})
    with pytest.raises(ValueError, match="duplicate Roost encoder key"):
        RoostEncoder(
            FakeRoostBackbone(),
            checkpoint={
                "state_dict": {
                    **encoder.state_dict(),
                    "encoder.material_nn.scale": encoder.material_nn.scale.detach().clone(),
                }
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"elem_fea_len": 1}, "elem_fea_len"),
        ({"n_graph": 0}, "n_graph"),
        ({"elem_heads": 0}, "elem_heads"),
        ({"cry_gate": (0,)}, "cry_gate"),
        ({"elem_embedding": ""}, "elem_embedding"),
    ],
)
def test_upstream_configuration_is_validated_before_optional_import(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        RoostEncoder(**kwargs)  # type: ignore[arg-type]


def test_real_aviary_roost_encoder_runs_on_cpu_when_materials_extra_is_installed() -> None:
    pytest.importorskip("aviary.roost.model")
    encoder = RoostEncoder(
        elem_fea_len=8,
        n_graph=1,
        elem_heads=1,
        elem_gate=(8,),
        elem_msg=(8,),
        cry_heads=1,
        cry_gate=(8,),
        cry_msg=(8,),
    ).double()
    encoder.eval()
    element_ids = torch.tensor([[26, 8, 0], [13, 8, 14]], dtype=torch.long)
    fractions = torch.tensor(
        [[0.25, 0.75, 0.0], [0.4, 0.6, 0.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    embeddings = encoder(element_ids, fractions)
    embeddings.sum().backward()

    assert encoder.initialization == "random"
    assert embeddings.shape == torch.Size([2, 8])
    assert embeddings.device.type == "cpu"
    assert embeddings.dtype == torch.double
    assert torch.isfinite(embeddings).all()
    assert fractions.grad is not None
    assert torch.isfinite(fractions.grad).all()
