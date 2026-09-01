from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from torch import Tensor, nn

pytest.importorskip("mace")

from bochan.composition import MACEEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import MACEDKLModel, MACEGPModel


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
        self.interactions = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(2)])
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


def _wrapped_encoder() -> tuple[MACEEncoder, CountingBatchBuilder]:
    builder = CountingBatchBuilder()
    return MACEEncoder(FakeMACE(), batch_builder=builder), builder


def _gp_model(*, with_process: bool = True) -> tuple[MACEGPModel, CountingBatchBuilder]:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    encoder, builder = _wrapped_encoder()
    model = MACEGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        outcome_transform=None,
    )
    return model, builder


def _dkl_model(*, trainable_encoder_layers: int | str = 1) -> MACEDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=True)
    encoder, _ = _wrapped_encoder()
    return MACEDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=encoder,
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_mace_gp_posterior_preserves_batch_q_shape() -> None:
    model, _ = _gp_model()
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


def test_mace_gp_uses_frozen_structure_feature_cache() -> None:
    model, builder = _gp_model()
    assert model.structure_feature_cache_enabled

    test_X = torch.tensor(
        [[0.0, 925.0, 1.5], [2.0, 1025.0, 3.5]],
        dtype=torch.double,
    )
    model.posterior(test_X)
    calls_after_first_posterior = builder.calls
    model.posterior(test_X)

    assert builder.calls == calls_after_first_posterior
    assert model.mace_feature_extractor.material_feature_cache is not None


def test_process_normalization_preserves_structure_index() -> None:
    model, _ = _gp_model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[:, 0], train_X[:, 0])
    assert (transformed[:, 1:] >= 0).all()
    assert (transformed[:, 1:] <= 1).all()


def test_frozen_mace_preserves_process_gradients_and_native_dtype() -> None:
    model, _ = _gp_model()
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
    model, _ = _gp_model()
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


def test_dkl_partial_unfreezes_final_interaction_product_pair() -> None:
    model = _dkl_model(trainable_encoder_layers=1)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeMACE)

    assert model.trainable_encoder_layers == 1
    assert not any(parameter.requires_grad for parameter in upstream.node_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in upstream.interactions[0].parameters())
    assert all(parameter.requires_grad for parameter in upstream.interactions[-1].parameters())
    assert not any(parameter.requires_grad for parameter in upstream.products[0].parameters())
    assert all(parameter.requires_grad for parameter in upstream.products[-1].parameters())
    assert not any(parameter.requires_grad for parameter in upstream.readouts.parameters())
    assert not model.structure_feature_cache_enabled

    model.train()
    assert not model.material_encoder.training
    assert not upstream.interactions[0].training
    assert upstream.interactions[-1].training
    assert not upstream.products[0].training
    assert upstream.products[-1].training


def test_dkl_all_unfreezes_representation_backbone_but_not_energy_readouts() -> None:
    model = _dkl_model(trainable_encoder_layers="all")
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeMACE)

    assert all(parameter.requires_grad for parameter in upstream.node_embedding.parameters())
    assert all(parameter.requires_grad for parameter in upstream.radial_embedding.parameters())
    assert all(
        parameter.requires_grad
        for module in (upstream.interactions, upstream.products)
        for parameter in module.parameters()
    )
    assert not any(parameter.requires_grad for parameter in upstream.readouts.parameters())
    assert not model.structure_feature_cache_enabled


def test_dkl_fit_updates_selected_pair_only() -> None:
    model = _dkl_model(trainable_encoder_layers=1)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeMACE)
    selected_interaction = upstream.interactions[-1]
    frozen_interaction = upstream.interactions[0]
    selected_product = upstream.products[-1]
    frozen_product = upstream.products[0]
    selected_interaction_before = selected_interaction.weight.detach().clone()
    frozen_interaction_before = frozen_interaction.weight.detach().clone()
    selected_product_before = selected_product.scale.detach().clone()
    frozen_product_before = frozen_product.scale.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert not torch.equal(selected_interaction.weight, selected_interaction_before)
    assert torch.equal(frozen_interaction.weight, frozen_interaction_before)
    assert not torch.equal(selected_product.scale, selected_product_before)
    assert torch.equal(frozen_product.scale, frozen_product_before)


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_mace_structure_indices_are_validated(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index
    encoder, _ = _wrapped_encoder()

    with pytest.raises(ValueError):
        MACEGPModel(
            train_X=train_X,
            train_Y=train_Y,
            structures=_structures(),
            encoder=encoder,
            latent_dim=3,
            outcome_transform=None,
        )


def test_partial_dkl_layers_cannot_exceed_interaction_product_pairs() -> None:
    with pytest.raises(ValueError, match="interaction/product pairs"):
        _dkl_model(trainable_encoder_layers=3)


def test_real_pretrained_mace_gp_returns_finite_posterior_on_cpu() -> None:
    pytest.importorskip("ase")
    structures = [_structure(5.40), _structure(5.55)]
    train_X = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.1], [0.2]], dtype=torch.double)
    model = MACEGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=structures,
        latent_dim=3,
        outcome_transform=None,
    )
    model.eval()

    posterior = model.posterior(torch.tensor([[0.0], [1.0]], dtype=torch.double))

    assert model.material_encoder.model_name == "medium-mpa-0"
    assert model.material_encoder.pooling == "mean"
    assert next(model.material_encoder.encoder.parameters()).dtype == torch.float32
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
