from __future__ import annotations

import pytest
import torch
from botorch.models.transforms.input import Normalize
from torch import Tensor, nn

from bochan.composition import ALIGNNEncoder
from bochan.models.regression.gaussian.deep import ALIGNNMixedGPModel


class FakeALIGNN(nn.Module):
    """Small injected graph encoder for dependency-free mixed-model tests."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_projection = nn.Linear(3, output_dim)
        self.alignn_layers = nn.ModuleList(
            [nn.Linear(output_dim, output_dim), nn.Linear(output_dim, output_dim)]
        )
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(output_dim, output_dim), nn.Linear(output_dim, output_dim)]
        )

    def encode(self, graph: Tensor) -> Tensor:
        values = torch.tanh(self.input_projection(graph))
        for layer in self.alignn_layers:
            values = torch.tanh(layer(values))
        for layer in self.gcn_layers:
            values = torch.tanh(layer(values))
        return values


def _graphs() -> list[Tensor]:
    return [
        torch.tensor([1.0, 0.2, 0.1], dtype=torch.double),
        torch.tensor([0.3, 1.0, 0.4], dtype=torch.double),
        torch.tensor([0.2, 0.5, 1.0], dtype=torch.double),
        torch.tensor([0.8, 0.7, 0.2], dtype=torch.double),
    ]


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
        0.2 * train_X[:, 0]
        + 0.001 * train_X[:, 1]
        + 0.08 * train_X[:, 2]
        + 0.05 * train_X[:, 3]
        + 0.03 * train_X[:, 4]
    ).unsqueeze(-1)
    return train_X, train_Y


def _model() -> ALIGNNMixedGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return ALIGNNMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        outcome_transform=None,
    )


def test_alignn_mixed_gp_uses_separate_structure_numeric_and_category_contract() -> None:
    model = _model()

    assert model.num_structures == 4
    assert model.process_dim == 2
    assert model.continuous_process_dims == (1, 3)
    assert model.categorical_process_dim == 2
    assert model.cat_dims == [2, 4]
    assert model.deepkernel.ord_dims == [0, 1, 3]
    assert model.deepkernel.cat_dims == [2, 4]
    assert model.alignn_feature_extractor.input_dim == 3


def test_alignn_mixed_gp_normalizes_only_numeric_process_columns() -> None:
    model = _model()
    train_X, _ = _data()
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    torch.testing.assert_close(transformed[:, 0], train_X[:, 0])
    torch.testing.assert_close(transformed[:, 2], train_X[:, 2])
    torch.testing.assert_close(transformed[:, 4], train_X[:, 4])
    assert (transformed[:, 1] >= 0).all()
    assert (transformed[:, 1] <= 1).all()
    assert (transformed[:, 3] >= 0).all()
    assert (transformed[:, 3] <= 1).all()


def test_alignn_mixed_gp_posterior_preserves_batch_q_shape() -> None:
    model = _model()
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
    samples = posterior.rsample(sample_shape=torch.Size([4]))

    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([4, 2, 2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_alignn_mixed_gp_preserves_numeric_process_gradients() -> None:
    model = _model()
    model.train()

    assert not model.material_encoder.training
    assert not any(
        parameter.requires_grad for parameter in model.material_encoder.parameters()
    )

    test_X = torch.tensor(
        [
            [1.0, 1000.0, 0.0, 2.0, 1.0],
            [2.0, 1100.0, 1.0, 3.0, 2.0],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    sample = model.posterior(test_X).rsample()
    sample.sum().backward()

    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    torch.testing.assert_close(
        test_X.grad[:, 0],
        torch.zeros_like(test_X.grad[:, 0]),
    )
    assert test_X.grad[:, [1, 3]].abs().sum() > 0
    torch.testing.assert_close(
        test_X.grad[:, [2, 4]],
        torch.zeros_like(test_X.grad[:, [2, 4]]),
    )
    assert all(
        parameter.grad is None for parameter in model.material_encoder.parameters()
    )


def test_alignn_mixed_gp_wraps_injected_encoder_with_alignn_adapter() -> None:
    model = _model()

    assert isinstance(model.material_encoder, ALIGNNEncoder)
    assert model.material_encoder.initialization == "injected"


def test_alignn_mixed_gp_rejects_structure_selector_in_cat_dims() -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="structure-index column"):
        ALIGNNMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[0, 2],
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            outcome_transform=None,
        )


def test_alignn_mixed_gp_requires_categorical_process_dimension() -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="at least one categorical"):
        ALIGNNMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[],
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            outcome_transform=None,
        )


@pytest.mark.parametrize("bad_index", [-1.0, 4.0, 1.5])
def test_alignn_mixed_gp_validates_structure_indices(bad_index: float) -> None:
    train_X, train_Y = _data()
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        ALIGNNMixedGPModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2, 4],
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            outcome_transform=None,
        )
