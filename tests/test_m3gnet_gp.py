from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import M3GNetDKLModel, M3GNetGPModel

pytest.importorskip("pymatgen")


class FakeM3GNetGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
        self.pbc_offset = torch.zeros((2, 3), dtype=torch.float32)
        self.pos = torch.empty_like(self.frac_coords)
        self.pbc_offshift = torch.empty_like(self.pbc_offset)

    def to(self, device: torch.device | str) -> FakeM3GNetGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pos = self.pos.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        return self


class FakeM3GNetConverter:
    def __init__(self) -> None:
        self.calls = 0

    def get_graph(self, structure: Any) -> tuple[FakeM3GNetGraph, Tensor, list[float]]:
        self.calls += 1
        graph = FakeM3GNetGraph(structure)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        return graph, lattice, [0.0, 0.0]


class FakeM3GNet(nn.Module):
    """Small extensive M3GNet stand-in with message-passing blocks."""

    def __init__(self, output_dim: int = 4, n_blocks: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.is_intensive = False
        self.include_state = False
        self.element_types = ("Si",)
        self.cutoff = 5.0
        self.n_blocks = n_blocks
        self.embedding = nn.Linear(3, output_dim)
        self.graph_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(n_blocks)
        )
        self.readout = nn.Linear(output_dim, 1)
        self.final_layer = nn.Linear(output_dim, 1)
        self.feature_dict: dict[str, Any] = {}

    def forward(self, g: FakeM3GNetGraph, state_attr: Tensor | None = None) -> Tensor:
        assert state_attr is None
        node_features = torch.tanh(self.embedding(g.pos))
        feature_dict: dict[str, Any] = {}
        for index, layer in enumerate(self.graph_layers, start=1):
            node_features = node_features + torch.tanh(layer(node_features))
            feature_dict[f"gc_{index}"] = {"node_feat": node_features}
        feature_dict["readout"] = self.readout(node_features)
        feature_dict["final"] = self.final_layer(node_features).sum()
        self.feature_dict = feature_dict
        return feature_dict["final"]


def _structure(scale: float) -> dict[str, object]:
    return {
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
        "cartesian": False,
    }


def _structures() -> list[dict[str, object]]:
    return [
        _structure(5.20),
        _structure(5.35),
        _structure(5.50),
        _structure(5.65),
    ]


def _data(*, with_process: bool = True) -> tuple[Tensor, Tensor]:
    structure = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.double).unsqueeze(-1)
    if not with_process:
        return structure, (0.25 * structure[:, 0]).unsqueeze(-1)

    process = torch.tensor(
        [
            [900.0, 1.0],
            [950.0, 2.0],
            [1000.0, 3.0],
            [1050.0, 4.0],
            [1100.0, 2.0],
            [1150.0, 3.0],
            [1200.0, 5.0],
            [1250.0, 4.0],
        ],
        dtype=torch.double,
    )
    train_X = torch.cat((structure, process), dim=-1)
    train_Y = (
        0.25 * structure[:, 0]
        + 0.001 * process[:, 0]
        + 0.04 * process[:, 1]
    ).unsqueeze(-1)
    return train_X, train_Y


def _wrapped_encoder() -> M3GNetEncoder:
    return M3GNetEncoder(
        FakeM3GNet(),
        graph_converter=FakeM3GNetConverter(),
    )


def _gp_model(*, with_process: bool = True) -> M3GNetGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return M3GNetGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(*, trainable_encoder_layers: int | str = 1) -> M3GNetDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=True)
    return M3GNetDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_m3gnet_gp_posterior_preserves_batch_q_shape() -> None:
    model = _gp_model()
    test_X = torch.tensor(
        [
            [[0.0, 925.0, 1.5], [1.0, 975.0, 2.5]],
            [[2.0, 1025.0, 3.5], [3.0, 1075.0, 4.5]],
        ],
        dtype=torch.double,
    )

    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([4]))

    assert model.num_structures == 4
    assert model.process_dim == 2
    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([4, 2, 2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_m3gnet_gp_uses_frozen_structure_feature_cache() -> None:
    model = _gp_model()
    converter = model.material_encoder.graph_converter
    assert isinstance(converter, FakeM3GNetConverter)
    assert model.structure_feature_cache_enabled

    test_X = torch.tensor(
        [[0.0, 925.0, 1.5], [2.0, 1025.0, 3.5]],
        dtype=torch.double,
    )
    model.posterior(test_X)
    calls_after_first_posterior = converter.calls
    model.posterior(test_X)

    assert converter.calls == calls_after_first_posterior
    assert model.m3gnet_feature_extractor.material_feature_cache is not None


def test_process_normalization_preserves_structure_index() -> None:
    model = _gp_model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[:, 0], train_X[:, 0])
    assert (transformed[:, 1:] >= 0).all()
    assert (transformed[:, 1:] <= 1).all()


def test_frozen_m3gnet_preserves_process_gradients_and_native_dtype() -> None:
    model = _gp_model()
    model.train()
    upstream = model.material_encoder.encoder

    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert next(upstream.parameters()).dtype == torch.float32
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    test_X = torch.tensor(
        [[1.0, 1000.0, 2.0], [2.0, 1100.0, 3.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    model.posterior(test_X).rsample().sum().backward()

    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert torch.allclose(test_X.grad[:, 0], torch.zeros_like(test_X.grad[:, 0]))
    assert test_X.grad[:, 1:].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_qlogei_optimizes_process_with_fixed_structure_index() -> None:
    model = _gp_model()
    train_X, train_Y = _data()
    model.eval()
    acquisition = qLogExpectedImprovement(model=model, best_f=train_Y.max())
    bounds = torch.tensor(
        [[0.0, 900.0, 1.0], [3.0, 1250.0, 5.0]],
        dtype=torch.double,
    )

    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features={0: 2.0},
    )

    assert candidate.shape == torch.Size([1, 3])
    assert candidate[0, 0].item() == 2.0
    assert 900.0 <= candidate[0, 1].item() <= 1250.0
    assert 1.0 <= candidate[0, 2].item() <= 5.0
    assert torch.isfinite(value)
    assert torch.equal(model.train_X_original, train_X)


def test_dkl_partial_unfreeze_selects_final_graph_layers() -> None:
    model = _dkl_model(trainable_encoder_layers=2)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeM3GNet)

    assert model.trainable_encoder_layers == 2
    assert not any(parameter.requires_grad for parameter in upstream.embedding.parameters())
    assert not any(parameter.requires_grad for parameter in upstream.graph_layers[0].parameters())
    assert all(
        parameter.requires_grad
        for layer in upstream.graph_layers[-2:]
        for parameter in layer.parameters()
    )
    assert not any(parameter.requires_grad for parameter in upstream.readout.parameters())
    assert not any(parameter.requires_grad for parameter in upstream.final_layer.parameters())
    assert not model.structure_feature_cache_enabled

    model.train()
    assert not model.material_encoder.training
    assert not upstream.graph_layers[0].training
    assert all(layer.training for layer in upstream.graph_layers[-2:])


def test_dkl_all_unfreezes_representation_backbone_but_not_property_head() -> None:
    model = _dkl_model(trainable_encoder_layers="all")
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeM3GNet)

    assert all(parameter.requires_grad for parameter in upstream.embedding.parameters())
    assert all(
        parameter.requires_grad
        for layer in upstream.graph_layers
        for parameter in layer.parameters()
    )
    assert not any(parameter.requires_grad for parameter in upstream.readout.parameters())
    assert not any(parameter.requires_grad for parameter in upstream.final_layer.parameters())


def test_dkl_fit_updates_selected_graph_layer_only() -> None:
    model = _dkl_model(trainable_encoder_layers=1)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeM3GNet)
    selected = upstream.graph_layers[-1]
    frozen = upstream.graph_layers[-2]
    selected_before = selected.weight.detach().clone()
    frozen_before = frozen.weight.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert not torch.equal(selected.weight, selected_before)
    assert torch.equal(frozen.weight, frozen_before)


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_m3gnet_structure_indices_are_validated(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        M3GNetGPModel(
            train_X=train_X,
            train_Y=train_Y,
            structures=_structures(),
            encoder=_wrapped_encoder(),
            latent_dim=3,
            outcome_transform=None,
        )


def test_real_pretrained_m3gnet_gp_returns_finite_posterior_on_cpu() -> None:
    pytest.importorskip("matgl")
    structures = [_structure(5.40), _structure(5.55)]
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.1], [0.2]], dtype=torch.double)
    model = M3GNetGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=structures,
        latent_dim=3,
        outcome_transform=None,
    )
    model.eval()

    posterior = model.posterior(torch.tensor([[0.0], [1.0]], dtype=torch.double))

    assert model.material_encoder.representation_mode == "mean_node"
    assert next(model.material_encoder.encoder.parameters()).dtype == torch.float32
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
