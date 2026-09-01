from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import (
    CHGNetMixedMultiTaskDKLModel,
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskDKLModel,
    CHGNetMultiTaskGPModel,
)

pytest.importorskip("pymatgen")


class FakeCrystalGraph:
    def __init__(self, structure: Any) -> None:
        self.lattice = torch.tensor(
            [
                float(structure.lattice.a) / 10.0,
                float(len(structure)) / 10.0,
                float(structure.frac_coords.sum()) / max(len(structure), 1),
            ],
            dtype=torch.float32,
        )

    def to(self, device: str = "cpu") -> FakeCrystalGraph:
        self.lattice = self.lattice.to(device)
        return self


class FakeGraphConverter:
    def __call__(self, structure: Any) -> FakeCrystalGraph:
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    def __init__(self, output_dim: int = 4, n_conv: int = 3) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.atom_embedding = nn.Linear(3, output_dim)
        self.atom_conv_layers = nn.ModuleList(
            nn.Linear(output_dim, output_dim) for _ in range(n_conv)
        )
        self.mlp = nn.Linear(output_dim, 1)
        self.graph_converter = FakeGraphConverter()

    def forward(
        self,
        graphs: Sequence[FakeCrystalGraph],
        *,
        task: str = "e",
        return_crystal_feas: bool = False,
    ) -> dict[str, Tensor]:
        assert task == "e"
        features = torch.stack([graph.lattice for graph in graphs])
        features = torch.tanh(self.atom_embedding(features))
        for layer in self.atom_conv_layers:
            features = features + torch.tanh(layer(features))
        result = {"e": self.mlp(features).squeeze(-1)}
        if return_crystal_feas:
            result["crystal_fea"] = features
        return result


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


def test_chgnet_multitask_gp_returns_correlated_wide_posterior() -> None:
    torch.manual_seed(0)
    X, Y = _train_data()
    model = CHGNetMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
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


def test_chgnet_multitask_process_gradients_remain_available() -> None:
    X, Y = _train_data()
    model = CHGNetMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
    ).double()
    query = X[:2].detach().clone().requires_grad_(True)
    posterior = model.posterior(query)
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), query)

    assert torch.isfinite(gradient).all()
    assert gradient[:, 1:].abs().sum() > 0


def test_chgnet_mixed_multitask_keeps_categories_outside_structure_branch() -> None:
    X, Y = _train_data(mixed=True)
    model = CHGNetMixedMultiTaskGPModel(
        X,
        Y,
        cat_dims=[2],
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
    ).double()

    assert model.cat_dims == [2]
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 1
    assert model.process_dim == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.posterior(X[:2]).mean.shape == torch.Size([2, 2])


def test_chgnet_multitask_dkl_partial_and_full_training_policies() -> None:
    X, Y = _train_data()
    partial = CHGNetMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers=1,
    ).double()
    full = CHGNetMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers="all",
    ).double()

    assert partial.trainable_encoder_layers == 1
    assert full.trainable_encoder_layers == "all"
    assert not partial.structure_feature_cache_enabled
    assert not full.structure_feature_cache_enabled
    assert any(parameter.requires_grad for parameter in partial.material_encoder.parameters())
    assert any(parameter.requires_grad for parameter in full.material_encoder.parameters())
    assert not partial.material_encoder.encoder.mlp.weight.requires_grad
    assert not full.material_encoder.encoder.mlp.weight.requires_grad


def test_chgnet_mixed_multitask_constructor_preserves_common_train_yvar_position() -> None:
    for cls in (CHGNetMixedMultiTaskGPModel, CHGNetMixedMultiTaskDKLModel):
        parameters = list(inspect.signature(cls.__init__).parameters.values())
        assert [parameter.name for parameter in parameters[:4]] == [
            "self",
            "train_X",
            "train_Y",
            "train_Yvar",
        ]
        assert inspect.signature(cls.__init__).parameters["cat_dims"].kind is inspect.Parameter.KEYWORD_ONLY
