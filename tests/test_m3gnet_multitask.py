from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder
from bochan.models.regression.gaussian.deep import (
    M3GNetMixedMultiTaskDKLModel,
    M3GNetMixedMultiTaskGPModel,
    M3GNetMultiTaskDKLModel,
    M3GNetMultiTaskGPModel,
)

pytest.importorskip("pymatgen")


class FakeGraph:
    def __init__(self, structure: Any) -> None:
        self.frac_coords = torch.as_tensor(
            structure.frac_coords,
            dtype=torch.float32,
        )
        self.pbc_offset = torch.zeros((1, 3), dtype=torch.float32)
        self.pbc_offshift = torch.zeros((1, 3), dtype=torch.float32)
        self.pos = self.frac_coords.clone()

    def to(self, device: Any) -> FakeGraph:
        self.frac_coords = self.frac_coords.to(device)
        self.pbc_offset = self.pbc_offset.to(device)
        self.pbc_offshift = self.pbc_offshift.to(device)
        self.pos = self.pos.to(device)
        return self


class FakeGraphConverter:
    def get_graph(self, structure: Any) -> tuple[FakeGraph, Tensor, None]:
        lattice = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
        return FakeGraph(structure), lattice, None


class FakeM3GNet(nn.Module):
    def __init__(self, output_dim: int = 4, n_blocks: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.is_intensive = True
        self.include_state = False
        self.embedding = nn.Linear(3, output_dim)
        self.graph_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(n_blocks)
        )
        self.final_layer = nn.Linear(output_dim, 1)
        self.feature_dict: dict[str, Tensor] = {}

    def forward(self, g: FakeGraph, state_attr: Tensor | None = None) -> Tensor:
        assert state_attr is None
        features = torch.tanh(self.embedding(g.frac_coords.mean(dim=0)))
        for layer in self.graph_layers:
            features = features + torch.tanh(layer(features))
        self.feature_dict = {"readout": features.unsqueeze(0)}
        return self.final_layer(features).squeeze(-1)


def _material_encoder() -> M3GNetEncoder:
    return M3GNetEncoder(
        encoder=FakeM3GNet(),
        graph_converter=FakeGraphConverter(),
    )


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


def _structures() -> tuple[dict[str, object], ...]:
    return (_structure(5.20), _structure(5.35), _structure(5.50))


def _train_data(*, mixed: bool = False) -> tuple[Tensor, Tensor]:
    if mixed:
        X = torch.tensor(
            [
                [0.0, 900.0, 0.0, 0.8],
                [1.0, 950.0, 1.0, 1.0],
                [2.0, 1000.0, 0.0, 1.2],
                [0.0, 1050.0, 1.0, 1.4],
                [1.0, 1100.0, 0.0, 1.6],
                [2.0, 1150.0, 1.0, 1.8],
            ],
            dtype=torch.double,
        )
    else:
        X = torch.tensor(
            [
                [0.0, 900.0, 0.8],
                [1.0, 950.0, 1.0],
                [2.0, 1000.0, 1.2],
                [0.0, 1050.0, 1.4],
                [1.0, 1100.0, 1.6],
                [2.0, 1150.0, 1.8],
            ],
            dtype=torch.double,
        )
    Y = torch.tensor(
        [
            [100.0, 2.1],
            [115.0, 2.4],
            [123.0, 2.2],
            [132.0, 2.7],
            [141.0, 2.6],
            [150.0, 3.0],
        ],
        dtype=torch.double,
    )
    return X, Y


def test_m3gnet_multitask_gp_returns_correlated_wide_posterior() -> None:
    torch.manual_seed(0)
    X, Y = _train_data()
    model = M3GNetMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=_material_encoder(),
        latent_dim=3,
    ).double()

    assert model.num_outputs == 2
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.task_covar_module is model.deepkernel.covar_module.task_covar_module
    assert model.structure_feature_cache_enabled
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())

    posterior = model.posterior(X[:2])
    subset = model.posterior(X[:2], output_indices=[1])
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert subset.mean.shape == torch.Size([2, 1])
    torch.testing.assert_close(subset.mean, posterior.mean[..., 1:2])
    torch.testing.assert_close(subset.variance, posterior.variance[..., 1:2])


def test_m3gnet_multitask_process_gradients_remain_available() -> None:
    X, Y = _train_data()
    model = M3GNetMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=_material_encoder(),
        latent_dim=3,
    ).double()
    query = X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(query)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), query)

    assert torch.isfinite(gradient).all()
    assert gradient[:, 1:].abs().sum() > 0


def test_m3gnet_mixed_multitask_keeps_categories_outside_structure_branch() -> None:
    X, Y = _train_data(mixed=True)
    model = M3GNetMixedMultiTaskGPModel(
        X,
        Y,
        cat_dims=[2],
        structures=_structures(),
        encoder=_material_encoder(),
        latent_dim=3,
    ).double()

    assert model.cat_dims == [2]
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 1
    assert model.process_dim == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.posterior(X[:2]).mean.shape == torch.Size([2, 2])


def test_m3gnet_multitask_dkl_partial_and_full_training_policies() -> None:
    X, Y = _train_data()
    partial = M3GNetMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=_material_encoder(),
        latent_dim=3,
        trainable_encoder_layers=1,
    ).double()
    full = M3GNetMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=_material_encoder(),
        latent_dim=3,
        trainable_encoder_layers="all",
    ).double()

    assert partial.trainable_encoder_layers == 1
    assert full.trainable_encoder_layers == "all"
    assert not partial.structure_feature_cache_enabled
    assert not full.structure_feature_cache_enabled
    assert any(parameter.requires_grad for parameter in partial.material_encoder.parameters())
    assert any(parameter.requires_grad for parameter in full.material_encoder.parameters())
    assert not partial.material_encoder.encoder.final_layer.weight.requires_grad
    assert not full.material_encoder.encoder.final_layer.weight.requires_grad

    layers = partial.material_encoder.encoder.graph_layers
    assert not layers[0].weight.requires_grad
    assert not layers[1].weight.requires_grad
    assert layers[2].weight.requires_grad


def test_m3gnet_mixed_multitask_constructor_preserves_common_train_yvar_position() -> None:
    for cls in (M3GNetMixedMultiTaskGPModel, M3GNetMixedMultiTaskDKLModel):
        parameters = list(inspect.signature(cls.__init__).parameters.values())
        assert [parameter.name for parameter in parameters[:4]] == [
            "self",
            "train_X",
            "train_Y",
            "train_Yvar",
        ]
        assert (
            inspect.signature(cls.__init__).parameters["cat_dims"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
