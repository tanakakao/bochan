from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from bochan.models.regression.gaussian.deep import ALIGNNMixedDKLModel


class FakeALIGNN(nn.Module):
    """Small injected ALIGNN-like encoder with explicit graph-convolution blocks."""

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


def _model(trainable_encoder_layers: int | str = 1) -> ALIGNNMixedDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data()
    return ALIGNNMixedDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[2, 4],
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,
        outcome_transform=None,
    )


def _parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def test_alignn_mixed_dkl_partial_unfreezes_only_final_graph_blocks() -> None:
    model = _model(trainable_encoder_layers=1)
    encoder = model.material_encoder.encoder
    selected = encoder.gcn_layers[-1]
    selected_ids = _parameter_ids(selected)

    assert model.trainable_encoder_layers == 1
    assert selected_ids
    for parameter in model.material_encoder.parameters():
        assert parameter.requires_grad == (id(parameter) in selected_ids)
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    model.train()
    assert not model.material_encoder.training
    assert selected.training
    assert not encoder.gcn_layers[0].training
    assert not encoder.alignn_layers[-1].training

    model.eval()
    assert not selected.training


def test_alignn_mixed_dkl_full_unfreezes_complete_representation_backbone() -> None:
    model = _model(trainable_encoder_layers="all")

    assert model.trainable_encoder_layers == "all"
    assert all(parameter.requires_grad for parameter in model.material_encoder.parameters())

    model.train()
    assert model.material_encoder.training

    model.eval()
    assert not model.material_encoder.training


def test_alignn_mixed_dkl_keeps_mixed_kernel_contract_and_posterior() -> None:
    model = _model(trainable_encoder_layers=2)

    assert model.cat_dims == [2, 4]
    assert model.deepkernel.ord_dims == [0, 1, 3]
    assert model.deepkernel.cat_dims == [2, 4]
    assert model.continuous_process_dims == (1, 3)
    assert model.process_dim == 2
    assert model.categorical_process_dim == 2

    test_X = torch.tensor(
        [
            [1.0, 1000.0, 0.0, 2.0, 1.0],
            [2.0, 1100.0, 1.0, 3.0, 2.0],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    posterior = model.posterior(test_X)
    loss = posterior.mean.sum()
    loss.backward()

    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
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

    trainable_encoder_parameters = [
        parameter
        for parameter in model.material_encoder.parameters()
        if parameter.requires_grad
    ]
    assert trainable_encoder_parameters
    assert any(parameter.grad is not None for parameter in trainable_encoder_parameters)


def test_alignn_mixed_dkl_supports_batched_q_posterior() -> None:
    model = _model(trainable_encoder_layers=1)
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


@pytest.mark.parametrize("bad_value", [0, -1, True])
def test_alignn_mixed_dkl_rejects_invalid_trainable_layer_count(bad_value: object) -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="trainable_encoder_layers"):
        ALIGNNMixedDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2, 4],
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            trainable_encoder_layers=bad_value,
            outcome_transform=None,
        )


def test_alignn_mixed_dkl_rejects_more_layers_than_encoder_exposes() -> None:
    train_X, train_Y = _data()

    with pytest.raises(ValueError, match="exceeds"):
        ALIGNNMixedDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=[2, 4],
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            trainable_encoder_layers=5,
            outcome_transform=None,
        )
