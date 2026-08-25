from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor, nn

from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import ALIGNNDKLModel, ALIGNNGPModel


class FakeALIGNN(nn.Module):
    """Small ALIGNN-shaped module exposing pooled readout features."""

    def __init__(self, hidden_features: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_features=hidden_features,
            classification=False,
            extra_features=0,
            alignn_layers=2,
        )
        self.atom_embedding = nn.Linear(3, hidden_features)
        self.alignn_layers = nn.ModuleList(
            nn.Linear(hidden_features, hidden_features) for _ in range(2)
        )
        self.gcn_layers = nn.ModuleList(
            nn.Linear(hidden_features, hidden_features) for _ in range(2)
        )
        self.readout = nn.Identity()
        self.fc = nn.Linear(hidden_features, 1)

    def forward(self, structure: Tensor) -> Tensor:
        h = torch.tanh(self.atom_embedding(structure))
        for layer in self.alignn_layers:
            h = torch.tanh(layer(h))
        for layer in self.gcn_layers:
            h = torch.tanh(layer(h))
        h = self.readout(h)
        return self.fc(h)


def _structures() -> tuple[Tensor, ...]:
    return (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, 1.0]),
    )


def _data(*, with_process: bool) -> tuple[Tensor, Tensor]:
    structure_ids = torch.tensor(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [0.0], [2.0], [4.0]],
        dtype=torch.double,
    )
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
        train_X = torch.cat([structure_ids, process], dim=-1)
        train_Y = (
            0.2 * structure_ids[:, 0]
            + 0.001 * process[:, 0]
            + 0.05 * process[:, 1]
        ).unsqueeze(-1)
    else:
        train_X = structure_ids
        train_Y = (0.2 * structure_ids[:, 0]).unsqueeze(-1)
    return train_X, train_Y


def _model(*, with_process: bool) -> ALIGNNGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return ALIGNNGPModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_catalog=_structures(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        outcome_transform=None,
    )


def _dkl_model(
    *,
    with_process: bool,
    trainable_encoder_layers: int | str = 1,
) -> ALIGNNDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return ALIGNNDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        structure_catalog=_structures(),
        encoder=FakeALIGNN(),
        latent_dim=3,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_structure_catalog_posterior_preserves_batch_q_shapes() -> None:
    model = _model(with_process=False)
    test_X = torch.tensor([[[0.0], [1.0]], [[2.0], [4.0]]], dtype=torch.double)

    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([4]))

    assert model.n_structures == 5
    assert model.process_dim == 0
    assert model.latent_dim == 3
    assert model.input_transform is None
    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([4, 2, 2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_process_features_are_normalized_without_touching_structure_ids() -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)
    transformed = model.transform_inputs(train_X)

    assert model.process_dim == 2
    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[..., 0], train_X[..., 0])
    assert (transformed[..., 1:] >= 0).all()
    assert (transformed[..., 1:] <= 1).all()


def test_frozen_alignn_encoder_and_process_gradient_contract() -> None:
    model = _model(with_process=True)
    model.train()

    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    test_X = torch.tensor(
        [[[1.0, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )
    posterior = model.posterior(test_X)
    sample = posterior.rsample()
    sample.sum().backward()

    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert test_X.grad[..., 0].abs().sum() == 0
    assert test_X.grad[..., 1:].abs().sum() > 0


def test_structure_id_must_be_integer_and_in_catalog() -> None:
    train_X, train_Y = _data(with_process=False)
    fractional = train_X.clone()
    fractional[0, 0] = 0.5

    with pytest.raises(ValueError, match="integer-valued structure ids"):
        ALIGNNGPModel(
            train_X=fractional,
            train_Y=train_Y,
            structure_catalog=_structures(),
            encoder=FakeALIGNN(),
            outcome_transform=None,
        )

    invalid = train_X.clone()
    invalid[0, 0] = 5.0
    with pytest.raises(ValueError, match="structure_catalog"):
        ALIGNNGPModel(
            train_X=invalid,
            train_Y=train_Y,
            structure_catalog=_structures(),
            encoder=FakeALIGNN(),
            outcome_transform=None,
        )


def test_dkl_partial_unfreeze_selects_final_message_passing_layers() -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=2)
    encoder = model.material_encoder.encoder

    assert model.trainable_encoder_layers == 2
    assert not any(parameter.requires_grad for parameter in encoder.atom_embedding.parameters())
    assert not any(
        parameter.requires_grad
        for layer in encoder.alignn_layers
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in encoder.gcn_layers
        for parameter in layer.parameters()
    )

    model.train()
    assert not model.material_encoder.training
    assert not any(layer.training for layer in encoder.alignn_layers)
    assert all(layer.training for layer in encoder.gcn_layers)


def test_dkl_fit_updates_selected_alignn_layers_projection_and_gp() -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=1)
    encoder = model.material_encoder.encoder
    frozen_before = encoder.gcn_layers[0].weight.detach().clone()
    trainable_before = encoder.gcn_layers[1].weight.detach().clone()
    projection_before = model.projection.weight.detach().clone()
    gp_before = model.deepkernel.covar_module.raw_outputscale.detach().clone()

    fit_deepkernel_mll(model.make_mll(), num_epochs=3, lr=0.01)

    assert torch.equal(encoder.gcn_layers[0].weight, frozen_before)
    assert not torch.equal(encoder.gcn_layers[1].weight, trainable_before)
    assert not torch.equal(model.projection.weight, projection_before)
    assert not torch.equal(model.deepkernel.covar_module.raw_outputscale, gp_before)


def test_qlogei_keeps_structure_discrete_and_process_differentiable() -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=1)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=17),
    )
    test_X = torch.tensor(
        [[[2.0, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(test_X)
    (gradient,) = torch.autograd.grad(value.sum(), test_X)

    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert gradient[..., 0].abs().sum() == 0
    assert gradient[..., 1:].abs().sum() > 0
