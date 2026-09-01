from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from torch import Tensor, nn

from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import CHGNetDKLModel, CHGNetGPModel

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
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: Any) -> FakeCrystalGraph:
        self.calls += 1
        return FakeCrystalGraph(structure)


class FakeCHGNet(nn.Module):
    """Small differentiable backbone matching CHGNet's crystal feature contract."""

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


def _structure(scale: float, element: str = "Si") -> dict[str, object]:
    return {
        "lattice_mat": [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        "coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "elements": [element, element],
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


def _gp_model(*, with_process: bool = True) -> CHGNetGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return CHGNetGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(*, trainable_encoder_layers: int | str = 1) -> CHGNetDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=True)
    return CHGNetDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_chgnet_gp_posterior_preserves_batch_q_shape() -> None:
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


def test_chgnet_gp_uses_frozen_structure_feature_cache() -> None:
    model = _gp_model()
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeCHGNet)
    assert model.structure_feature_cache_enabled

    test_X = torch.tensor(
        [[0.0, 925.0, 1.5], [2.0, 1025.0, 3.5]],
        dtype=torch.double,
    )
    model.posterior(test_X)
    calls_after_first_posterior = upstream.graph_converter.calls
    model.posterior(test_X)

    assert upstream.graph_converter.calls == calls_after_first_posterior
    assert model.chgnet_feature_extractor.material_feature_cache is not None


def test_process_normalization_preserves_structure_index() -> None:
    model = _gp_model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[:, 0], train_X[:, 0])
    assert (transformed[:, 1:] >= 0).all()
    assert (transformed[:, 1:] <= 1).all()


def test_frozen_chgnet_preserves_process_gradients_and_zero_structure_gradient() -> None:
    model = _gp_model()
    model.train()

    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
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


def test_dkl_partial_unfreeze_selects_final_atom_conv_blocks() -> None:
    model = _dkl_model(trainable_encoder_layers=2)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeCHGNet)

    assert model.trainable_encoder_layers == 2
    assert not any(parameter.requires_grad for parameter in upstream.atom_embedding.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in upstream.atom_conv_layers[0].parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in upstream.atom_conv_layers[-2:]
        for parameter in layer.parameters()
    )
    assert not any(parameter.requires_grad for parameter in upstream.mlp.parameters())
    assert not model.structure_feature_cache_enabled

    model.train()
    assert not model.material_encoder.training
    assert not upstream.atom_conv_layers[0].training
    assert all(layer.training for layer in upstream.atom_conv_layers[-2:])


def test_dkl_all_unfreezes_representation_backbone_but_not_property_head() -> None:
    model = _dkl_model(trainable_encoder_layers="all")
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeCHGNet)

    assert all(parameter.requires_grad for parameter in upstream.atom_embedding.parameters())
    assert all(
        parameter.requires_grad
        for layer in upstream.atom_conv_layers
        for parameter in layer.parameters()
    )
    assert not any(parameter.requires_grad for parameter in upstream.mlp.parameters())


def test_dkl_fit_updates_selected_atom_conv_block_only() -> None:
    model = _dkl_model(trainable_encoder_layers=1)
    upstream = model.material_encoder.encoder
    assert isinstance(upstream, FakeCHGNet)
    selected = upstream.atom_conv_layers[-1]
    frozen = upstream.atom_conv_layers[-2]
    selected_before = selected.weight.detach().clone()
    frozen_before = frozen.weight.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert not torch.equal(selected.weight, selected_before)
    assert torch.equal(frozen.weight, frozen_before)


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_chgnet_structure_indices_are_validated(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        CHGNetGPModel(
            train_X=train_X,
            train_Y=train_Y,
            structures=_structures(),
            encoder=FakeCHGNet(),
            latent_dim=3,
            outcome_transform=None,
        )
