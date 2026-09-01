from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import (
    CHGNetMixedDKLModel,
    CHGNetMixedGPModel,
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
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, structure: Any) -> FakeCrystalGraph:
        self.calls += 1
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


def _gp_model() -> CHGNetMixedGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return CHGNetMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(trainable_encoder_layers: int | str = 1) -> CHGNetMixedDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return CHGNetMixedDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structures=_structures(),
        encoder=FakeCHGNet(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_chgnet_mixed_gp_separates_structure_numeric_and_categories() -> None:
    model = _gp_model()

    assert model.num_structures == 4
    assert model.process_dim == 2
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 2
    assert model.cat_dims == [2, 4]
    assert model.deepkernel.ord_dims == [0, 1, 3]
    assert model.deepkernel.cat_dims == [2, 4]
    assert model.chgnet_feature_extractor.input_dim == 3
    assert model.structure_feature_cache_enabled


def test_chgnet_mixed_gp_normalizes_only_numeric_process_columns() -> None:
    model = _gp_model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    torch.testing.assert_close(transformed[:, 0], train_X[:, 0])
    torch.testing.assert_close(transformed[:, 2], train_X[:, 2])
    torch.testing.assert_close(transformed[:, 4], train_X[:, 4])
    assert (transformed[:, [1, 3]] >= 0).all()
    assert (transformed[:, [1, 3]] <= 1).all()


def test_chgnet_mixed_gp_posterior_and_numeric_gradients() -> None:
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


def test_chgnet_mixed_gp_supports_batched_q_posterior() -> None:
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


def test_chgnet_mixed_dkl_partial_unfreezes_final_atom_conv_blocks() -> None:
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


def test_chgnet_mixed_dkl_full_unfreezes_backbone_not_property_head() -> None:
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


def test_chgnet_mixed_rejects_structure_selector_in_cat_dims() -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="structure-index column"):
        CHGNetMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[0, 2],
            structures=_structures(),
            encoder=FakeCHGNet(),
            latent_dim=3,
            outcome_transform=None,
        )


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_chgnet_mixed_validates_structure_indices(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        CHGNetMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2, 4],
            structures=_structures(),
            encoder=FakeCHGNet(),
            latent_dim=3,
            outcome_transform=None,
        )
