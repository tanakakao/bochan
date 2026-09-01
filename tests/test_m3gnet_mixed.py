from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import (
    M3GNetMixedDKLModel,
    M3GNetMixedGPModel,
)

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
    return [_structure(scale) for scale in (5.20, 5.35, 5.50, 5.65)]


def _data() -> tuple[Tensor, Tensor]:
    # [structure_index, temperature, furnace, pressure, atmosphere]
    train_X = torch.tensor(
        [
            [0.0, 900.0, 0.0, 1.0, 0.0],
            [1.0, 950.0, 1.0, 2.0, 1.0],
            [2.0, 1000.0, 0.0, 3.0, 2.0],
            [3.0, 1050.0, 1.0, 4.0, 0.0],
            [0.0, 1100.0, 0.0, 2.0, 1.0],
            [1.0, 1150.0, 1.0, 3.0, 2.0],
            [2.0, 1200.0, 0.0, 5.0, 0.0],
            [3.0, 1250.0, 1.0, 4.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = (
        0.20 * train_X[:, 0]
        + 0.001 * train_X[:, 1]
        + 0.08 * train_X[:, 2]
        + 0.05 * train_X[:, 3]
        + 0.03 * train_X[:, 4]
    ).unsqueeze(-1)
    return train_X, train_Y


def _wrapped_encoder() -> M3GNetEncoder:
    return M3GNetEncoder(
        FakeM3GNet(),
        graph_converter=FakeM3GNetConverter(),
    )


def _gp_model() -> M3GNetMixedGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return M3GNetMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(trainable_encoder_layers: int | str = 1) -> M3GNetMixedDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return M3GNetMixedDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_m3gnet_mixed_gp_separates_structure_numeric_and_categories() -> None:
    model = _gp_model()
    upstream = model.material_encoder.encoder

    assert model.num_structures == 4
    assert model.process_dim == 2
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 2
    assert model.cat_dims == [2, 4]
    assert model.deepkernel.ord_dims == [0, 1, 3]
    assert model.deepkernel.cat_dims == [2, 4]
    assert model.m3gnet_feature_extractor.input_dim == 3
    assert model.structure_feature_cache_enabled
    assert next(upstream.parameters()).dtype == torch.float32
    assert next(model.projection.parameters()).dtype == torch.float64


def test_m3gnet_mixed_gp_normalizes_only_numeric_process_columns() -> None:
    model = _gp_model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    torch.testing.assert_close(transformed[:, 0], train_X[:, 0])
    torch.testing.assert_close(transformed[:, 2], train_X[:, 2])
    torch.testing.assert_close(transformed[:, 4], train_X[:, 4])
    assert (transformed[:, [1, 3]] >= 0).all()
    assert (transformed[:, [1, 3]] <= 1).all()


def test_m3gnet_mixed_gp_posterior_and_numeric_gradients() -> None:
    model = _gp_model()
    test_X = torch.tensor(
        [
            [1.0, 1000.0, 0.0, 2.0, 1.0],
            [2.0, 1100.0, 1.0, 3.0, 2.0],
        ],
        dtype=torch.double,
        requires_grad=True,
    )

    posterior = model.posterior(test_X)
    posterior.rsample().sum().backward()

    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    torch.testing.assert_close(test_X.grad[:, 0], torch.zeros_like(test_X.grad[:, 0]))
    assert test_X.grad[:, [1, 3]].abs().sum() > 0
    torch.testing.assert_close(
        test_X.grad[:, [2, 4]],
        torch.zeros_like(test_X.grad[:, [2, 4]]),
    )


def test_m3gnet_mixed_gp_supports_batched_q_posterior() -> None:
    model = _gp_model()
    test_X = torch.tensor(
        [
            [
                [0.0, 925.0, 0.0, 1.5, 0.0],
                [1.0, 975.0, 1.0, 2.5, 1.0],
            ],
            [
                [2.0, 1025.0, 0.0, 3.5, 2.0],
                [3.0, 1075.0, 1.0, 4.5, 0.0],
            ],
        ],
        dtype=torch.double,
    )

    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([3]))

    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([3, 2, 2, 1])


def test_optimize_acqf_mixed_enumerates_categories_with_fixed_structure() -> None:
    model = _gp_model()
    _, train_Y = _data()
    model.eval()
    acquisition = qLogExpectedImprovement(model=model, best_f=train_Y.max())
    bounds = torch.tensor(
        [[0.0, 900.0, 0.0, 1.0, 0.0], [3.0, 1250.0, 1.0, 5.0, 2.0]],
        dtype=torch.double,
    )
    fixed_features_list = [
        {0: 2.0, 2: furnace, 4: atmosphere}
        for furnace in (0.0, 1.0)
        for atmosphere in (0.0, 1.0, 2.0)
    ]

    candidate, value = optimize_acqf_mixed(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features_list=fixed_features_list,
    )

    assert candidate.shape == torch.Size([1, 5])
    assert candidate[0, 0].item() == 2.0
    assert candidate[0, 2].item() in {0.0, 1.0}
    assert candidate[0, 4].item() in {0.0, 1.0, 2.0}
    assert 900.0 <= candidate[0, 1].item() <= 1250.0
    assert 1.0 <= candidate[0, 3].item() <= 5.0
    assert torch.isfinite(value)


def test_m3gnet_mixed_gp_supports_categorical_only_process_columns() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]],
        dtype=torch.double,
    )
    train_Y = (0.2 * train_X[:, 0] + 0.1 * train_X[:, 1]).unsqueeze(-1)
    model = M3GNetMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[1],
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        outcome_transform=None,
    )

    assert model.process_dim == 0
    assert model.continuous_process_dims == ()
    assert model.input_transform is None
    assert model.deepkernel.ord_dims == [0]
    posterior = model.posterior(train_X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()


def test_m3gnet_mixed_dkl_partial_unfreezes_final_graph_layers() -> None:
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


def test_m3gnet_mixed_dkl_full_unfreezes_backbone_not_property_head() -> None:
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


def test_m3gnet_mixed_dkl_default_fit_updates_selected_graph_layer_only() -> None:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    model = M3GNetMixedDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structures=_structures(),
        encoder=_wrapped_encoder(),
        latent_dim=3,
        trainable_encoder_layers=1,
    )
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeM3GNet)
    selected = upstream.graph_layers[-1]
    frozen = upstream.graph_layers[-2]
    selected_before = selected.weight.detach().clone()
    frozen_before = frozen.weight.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert not torch.equal(selected.weight, selected_before)
    assert torch.equal(frozen.weight, frozen_before)


def test_m3gnet_mixed_rejects_structure_selector_in_cat_dims() -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="structure-index column"):
        M3GNetMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[0, 2],
            structures=_structures(),
            encoder=_wrapped_encoder(),
            latent_dim=3,
            outcome_transform=None,
        )


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_m3gnet_mixed_validates_structure_indices(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        M3GNetMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2, 4],
            structures=_structures(),
            encoder=_wrapped_encoder(),
            latent_dim=3,
            outcome_transform=None,
        )
