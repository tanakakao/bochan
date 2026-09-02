from __future__ import annotations

import inspect

import pytest
import torch
from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

pytest.importorskip("mace")

from bochan.composition import MACEEncoder
from bochan.models.regression.gaussian.deep import (
    MACEMixedMultiTaskDKLModel,
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskDKLModel,
    MACEMultiTaskGPModel,
)


class FakeDescriptorLinear(nn.Linear):
    def __init__(self, width: int) -> None:
        super().__init__(width, width, bias=False)
        self.irreps_out = f"{width}x0e + {width}x1o"


class FakeProduct(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = FakeDescriptorLinear(width)
        self.scale = nn.Parameter(torch.ones(()))


class FakeMACE(nn.Module):
    """Small differentiable MACE stand-in with interaction/product blocks."""

    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.register_buffer("atomic_numbers", torch.tensor([14], dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(5.0, dtype=torch.float32))
        self.register_buffer("num_interactions", torch.tensor(2, dtype=torch.int64))
        self.heads = ["Default"]
        self.node_embedding = nn.Linear(3, width, bias=False)
        self.radial_embedding = nn.Linear(1, width, bias=False)
        self.spherical_harmonics = nn.Identity()
        self.interactions = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(2)]
        )
        self.products = nn.ModuleList([FakeProduct(width) for _ in range(2)])
        self.readouts = nn.ModuleList([nn.Linear(width, 1) for _ in range(2)])

    def forward(
        self,
        data: dict[str, Tensor],
        *,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> dict[str, Tensor]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        positions = data["positions"]
        first = self.products[0].scale * torch.tanh(self.node_embedding(positions))
        equivariant = torch.cat([positions, positions], dim=-1)
        final = self.products[1].scale * torch.tanh(self.interactions[-1](first))
        node_feats = torch.cat([first, equivariant, final], dim=-1)
        return {
            "node_feats": node_feats,
            "energy": self.readouts[-1](final).sum(),
        }


class CountingBatchBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: dict[str, object]) -> dict[str, Tensor]:
        self.calls += 1
        lattice = torch.tensor(structure["lattice_mat"], dtype=torch.float32)
        coords = torch.tensor(structure["coords"], dtype=torch.float32)
        positions = coords if bool(structure.get("cartesian", False)) else coords @ lattice
        return {"positions": positions}


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


def _material_encoder() -> tuple[MACEEncoder, CountingBatchBuilder]:
    builder = CountingBatchBuilder()
    return MACEEncoder(FakeMACE(), batch_builder=builder), builder


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


def test_mace_multitask_gp_returns_correlated_wide_posterior() -> None:
    torch.manual_seed(0)
    X, Y = _train_data()
    encoder, _ = _material_encoder()
    model = MACEMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
    ).double()

    assert model.num_outputs == 2
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert model.task_covar_module is model.deepkernel.covar_module.task_covar_module
    assert model.structure_feature_cache_enabled
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert next(model.material_encoder.encoder.parameters()).dtype == torch.float32
    assert next(model.projection.parameters()).dtype == torch.float64

    task_covar = model.task_covar_module.covar_matrix.to_dense()
    assert task_covar.shape == torch.Size([2, 2])
    assert torch.isfinite(task_covar).all()

    posterior = model.posterior(X[:2])
    subset = model.posterior(X[:2], output_indices=[1])
    batched = model.posterior(X[:2].unsqueeze(0))
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert subset.mean.shape == torch.Size([2, 1])
    assert batched.mean.shape == torch.Size([1, 2, 2])
    torch.testing.assert_close(subset.mean, posterior.mean[..., 1:2])
    torch.testing.assert_close(subset.variance, posterior.variance[..., 1:2])


def test_mace_multitask_process_gradients_and_frozen_cache_remain_available() -> None:
    X, Y = _train_data()
    encoder, builder = _material_encoder()
    model = MACEMultiTaskGPModel(
        X,
        Y,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        outcome_transform=None,
    ).double()
    query = X[:2].detach().clone().requires_grad_(True)

    posterior = model.posterior(query)
    calls_after_first = builder.calls
    model.posterior(query.detach())
    (gradient,) = torch.autograd.grad(posterior.mean.sum(), query)

    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, 0], torch.zeros_like(gradient[:, 0]))
    assert gradient[:, 1:].abs().sum() > 0
    assert builder.calls == calls_after_first
    assert model.mace_feature_extractor.material_feature_cache is not None


def test_mace_mixed_multitask_keeps_categories_outside_structure_branch() -> None:
    X, Y = _train_data(mixed=True)
    encoder, _ = _material_encoder()
    model = MACEMixedMultiTaskGPModel(
        X,
        Y,
        cat_dims=[2],
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
    ).double()

    assert model.cat_dims == [2]
    assert model.deepkernel.ord_dims == [0, 1, 3]
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 1
    assert model.process_dim == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mace_multitask_dkl_partial_and_full_training_policies() -> None:
    X, Y = _train_data()
    partial_encoder, _ = _material_encoder()
    full_encoder, _ = _material_encoder()
    partial = MACEMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=partial_encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
    ).double()
    full = MACEMultiTaskDKLModel(
        X,
        Y,
        structures=_structures(),
        encoder=full_encoder,
        latent_dim=3,
        trainable_encoder_layers="all",
    ).double()

    partial_upstream = partial.material_encoder.encoder
    full_upstream = full.material_encoder.encoder
    assert isinstance(partial_upstream, FakeMACE)
    assert isinstance(full_upstream, FakeMACE)
    assert partial.trainable_encoder_layers == 1
    assert full.trainable_encoder_layers == "all"
    assert not partial.structure_feature_cache_enabled
    assert not full.structure_feature_cache_enabled

    assert not partial_upstream.node_embedding.weight.requires_grad
    assert not partial_upstream.interactions[0].weight.requires_grad
    assert partial_upstream.interactions[-1].weight.requires_grad
    assert not partial_upstream.products[0].scale.requires_grad
    assert partial_upstream.products[-1].scale.requires_grad
    assert not any(parameter.requires_grad for parameter in partial_upstream.readouts.parameters())

    assert full_upstream.node_embedding.weight.requires_grad
    assert full_upstream.radial_embedding.weight.requires_grad
    assert all(parameter.requires_grad for parameter in full_upstream.interactions.parameters())
    assert all(parameter.requires_grad for parameter in full_upstream.products.parameters())
    assert not any(parameter.requires_grad for parameter in full_upstream.readouts.parameters())


def test_mace_mixed_multitask_dkl_uses_same_partial_training_policy() -> None:
    X, Y = _train_data(mixed=True)
    encoder, _ = _material_encoder()
    model = MACEMixedMultiTaskDKLModel(
        X,
        Y,
        cat_dims=[2],
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=1,
    ).double()
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeMACE)

    assert model.trainable_encoder_layers == 1
    assert model.continuous_process_dims == (1, 3)
    assert not model.structure_feature_cache_enabled
    assert upstream.interactions[-1].weight.requires_grad
    assert upstream.products[-1].scale.requires_grad
    assert not any(parameter.requires_grad for parameter in upstream.readouts.parameters())


def test_mace_multitask_rejects_single_output_and_accepts_train_yvar() -> None:
    X, Y = _train_data()
    encoder, _ = _material_encoder()
    with pytest.raises(ValueError, match="wide train_Y"):
        MACEMultiTaskGPModel(
            X,
            Y[:, :1],
            structures=_structures(),
            encoder=encoder,
        )

    train_Yvar = torch.full_like(Y, 0.01)
    encoder, _ = _material_encoder()
    model = MACEMultiTaskGPModel(
        X,
        Y,
        train_Yvar,
        structures=_structures(),
        encoder=encoder,
        outcome_transform=None,
    ).double()

    assert model.train_Yvar is not None
    torch.testing.assert_close(model.train_Yvar, train_Yvar)
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mace_mixed_multitask_rejects_structure_selector_in_cat_dims() -> None:
    X, Y = _train_data(mixed=True)
    encoder, _ = _material_encoder()
    with pytest.raises(ValueError, match="structure-index column"):
        MACEMixedMultiTaskGPModel(
            X,
            Y,
            cat_dims=[0, 2],
            structures=_structures(),
            encoder=encoder,
        )


def test_mace_mixed_multitask_constructor_preserves_common_train_yvar_position() -> None:
    for cls in (MACEMixedMultiTaskGPModel, MACEMixedMultiTaskDKLModel):
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


def test_real_pretrained_mace_multitask_gp_returns_finite_posterior_on_cpu() -> None:
    pytest.importorskip("ase")
    structures = (_structure(5.40), _structure(5.55))
    X = torch.tensor(
        [
            [0.0, 900.0],
            [1.0, 950.0],
            [0.0, 1000.0],
            [1.0, 1050.0],
        ],
        dtype=torch.double,
    )
    Y = torch.tensor(
        [
            [100.0, 2.0],
            [112.0, 2.3],
            [121.0, 2.2],
            [134.0, 2.6],
        ],
        dtype=torch.double,
    )
    model = MACEMultiTaskGPModel(
        X,
        Y,
        structures=structures,
        latent_dim=3,
        outcome_transform=None,
    )
    model.eval()

    posterior = model.posterior(X[:2])
    mace_dtype = next(model.material_encoder.encoder.parameters()).dtype
    structure_features = model.material_encoder([structures[0]])

    assert model.material_encoder.model_name == "medium-mpa-0"
    assert model.num_tasks == 2
    assert isinstance(model.deepkernel.covar_module, MultitaskKernel)
    assert mace_dtype in {torch.float32, torch.float64}
    assert structure_features.dtype == mace_dtype
    assert posterior.mean.dtype == torch.double
    assert posterior.mean.shape == torch.Size([2, 2])
    assert posterior.variance.shape == torch.Size([2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
