from __future__ import annotations

import pytest
import torch
from botorch.models.transforms.input import Normalize
from torch import Tensor, nn

from bochan.composition import ALIGNNEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import ALIGNNDKLModel, ALIGNNGPModel


class FakeALIGNN(nn.Module):
    """Small graph encoder that mimics ALIGNN's ordered convolution stacks."""

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
        x = torch.tanh(self.input_projection(graph))
        for layer in self.alignn_layers:
            x = torch.tanh(layer(x))
        for layer in self.gcn_layers:
            x = torch.tanh(layer(x))
        return x


def _graphs() -> list[Tensor]:
    return [
        torch.tensor([1.0, 0.2, 0.1], dtype=torch.double),
        torch.tensor([0.3, 1.0, 0.4], dtype=torch.double),
        torch.tensor([0.2, 0.5, 1.0], dtype=torch.double),
        torch.tensor([0.8, 0.7, 0.2], dtype=torch.double),
    ]


def _data(*, with_process: bool = True) -> tuple[Tensor, Tensor]:
    structure = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.double).unsqueeze(-1)
    if with_process:
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
        train_X = torch.cat([structure, process], dim=-1)
        train_Y = (
            0.2 * structure[:, 0]
            + 0.001 * process[:, 0]
            + 0.05 * process[:, 1]
        ).unsqueeze(-1)
    else:
        train_X = structure
        train_Y = (0.2 * structure[:, 0]).unsqueeze(-1)
    return train_X, train_Y


def _gp_model(*, with_process: bool = True) -> ALIGNNGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(*, trainable_encoder_layers: int | str = 1) -> ALIGNNDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=True)
    return ALIGNNDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_alignn_encoder_injected_graph_batch() -> None:
    encoder = ALIGNNEncoder(encoder=FakeALIGNN()).double()
    features = encoder(_graphs()[:3])

    assert encoder.output_dim == 4
    assert encoder.initialization == "injected"
    assert features.shape == torch.Size([3, 4])
    assert torch.isfinite(features).all()


def test_alignn_gp_posterior_preserves_batch_q_shape() -> None:
    model = _gp_model(with_process=True)
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


def test_alignn_custom_projection_receives_flat_rows_for_q_batch() -> None:
    train_X, train_Y = _data(with_process=True)
    projection = nn.Sequential(
        nn.BatchNorm1d(6),
        nn.Linear(6, 3),
    ).double()
    model = ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_graphs=_graphs(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        projection=projection,
        outcome_transform=None,
    )
    test_X = torch.tensor(
        [
            [[0.0, 925.0, 1.5], [1.0, 975.0, 2.5]],
            [[2.0, 1025.0, 3.5], [3.0, 1075.0, 4.5]],
        ],
        dtype=torch.double,
    )

    model.eval()
    features = model.alignn_feature_extractor(model.transform_inputs(test_X))
    posterior = model.posterior(test_X)

    assert features.shape == torch.Size([2, 2, 3])
    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert torch.isfinite(features).all()
    assert torch.isfinite(posterior.mean).all()


def test_alignn_gp_exposes_canonical_raw_training_inputs() -> None:
    model = _gp_model(with_process=True)
    train_X, _ = _data(with_process=True)

    assert torch.equal(model.train_X_original, train_X)
    assert model.train_X_original.data_ptr() != train_X.data_ptr()

    updated_X = train_X.clone()
    updated_X[:, 1] += 5.0
    model.set_train_data(inputs=updated_X, strict=False)

    assert torch.equal(model.train_X_original, updated_X)
    assert model.train_X_original.data_ptr() != updated_X.data_ptr()


def test_process_normalization_preserves_structure_index() -> None:
    model = _gp_model(with_process=True)
    train_X, _ = _data(with_process=True)
    transformed = model.transform_inputs(train_X)

    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[:, 0], train_X[:, 0])
    assert (transformed[:, 1:] >= 0).all()
    assert (transformed[:, 1:] <= 1).all()


def test_frozen_alignn_preserves_process_gradients_and_zero_structure_gradient() -> None:
    model = _gp_model(with_process=True)
    model.train()

    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    test_X = torch.tensor(
        [[1.0, 1000.0, 2.0], [2.0, 1100.0, 3.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    sample = model.posterior(test_X).rsample()
    sample.sum().backward()

    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert torch.allclose(test_X.grad[:, 0], torch.zeros_like(test_X.grad[:, 0]))
    assert test_X.grad[:, 1:].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_dkl_partial_unfreeze_selects_final_graph_conv_blocks() -> None:
    model = _dkl_model(trainable_encoder_layers=2)
    encoder = model.material_encoder.encoder

    assert model.trainable_encoder_layers == 2
    assert not any(parameter.requires_grad for parameter in encoder.input_projection.parameters())
    assert not any(parameter.requires_grad for layer in encoder.alignn_layers for parameter in layer.parameters())
    assert all(parameter.requires_grad for layer in encoder.gcn_layers for parameter in layer.parameters())

    model.train()
    assert not model.material_encoder.training
    assert not any(layer.training for layer in encoder.alignn_layers)
    assert all(layer.training for layer in encoder.gcn_layers)

    model.eval()
    assert not any(layer.training for layer in encoder.gcn_layers)


def test_dkl_fit_updates_selected_encoder_block() -> None:
    model = _dkl_model(trainable_encoder_layers=1)
    encoder = model.material_encoder.encoder
    selected = encoder.gcn_layers[-1]
    frozen = encoder.gcn_layers[-2]
    selected_before = selected.weight.detach().clone()
    frozen_before = frozen.weight.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert not torch.equal(selected.weight, selected_before)
    assert torch.equal(frozen.weight, frozen_before)


@pytest.mark.parametrize(
    "bad_index",
    [-1.0, 4.0, 1.5],
)
def test_alignn_structure_indices_are_validated(bad_index: float) -> None:
    train_X, train_Y = _data(with_process=True)
    train_X = train_X.clone()
    train_X[0, 0] = bad_index

    with pytest.raises(ValueError):
        ALIGNNGPModel(
            train_X=train_X,
            train_Y=train_Y,
            structure_graphs=_graphs(),
            encoder=FakeALIGNN(),
            latent_dim=3,
            outcome_transform=None,
        )
