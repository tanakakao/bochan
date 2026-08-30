from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor, nn

import bochan.composition.encoders.roost as roost_module
from bochan.composition import CompositionTransformer, RoostEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import (
    CompositionMaterialInputTransform,
    CrabNetDKLModel,
    RoostDKLModel,
    RoostGPModel,
)


class FakeRoostBackbone(nn.Module):
    """Small differentiable graph encoder with Aviary's Roost call contract."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.elem_embedding = nn.Embedding(119, output_dim)
        self.scale = nn.Parameter(torch.arange(1, output_dim + 1, dtype=torch.float32))
        with torch.no_grad():
            atomic_numbers = torch.arange(119, dtype=torch.float32).unsqueeze(-1)
            channels = torch.arange(output_dim, dtype=torch.float32).unsqueeze(0)
            self.elem_embedding.weight.copy_(atomic_numbers / 100 + channels / 10)

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del self_idx, nbr_idx
        node_features = self.elem_embedding(elem_fea) * elem_weights * self.scale
        num_materials = int(cry_elem_idx[-1].item()) + 1
        pooled = node_features.new_zeros((num_materials, self.output_dim))
        pooled.index_add_(0, cry_elem_idx, node_features)
        return torch.tanh(pooled)


class FakeRoostDescriptor(nn.Module):
    """Differentiable Roost descriptor exposing graphs and crystal pooling."""

    def __init__(self, output_dim: int = 4, n_graphs: int = 3, cry_heads: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Linear(output_dim, output_dim)
        self.graphs = nn.ModuleList(nn.Linear(output_dim, output_dim) for _ in range(n_graphs))
        self.cry_pool = nn.ModuleList(nn.Linear(output_dim, output_dim) for _ in range(cry_heads))

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del self_idx, nbr_idx
        features = torch.tanh(self.embedding(elem_fea))
        for graph in self.graphs:
            features = features + torch.tanh(graph(features))

        num_materials = int(cry_elem_idx[-1].item()) + 1
        normalizer = elem_weights.new_zeros((num_materials, 1))
        normalizer.index_add_(0, cry_elem_idx, elem_weights)
        pooled_heads: list[Tensor] = []
        for pool in self.cry_pool:
            messages = torch.tanh(pool(features)) * elem_weights
            pooled = messages.new_zeros((num_materials, messages.shape[-1]))
            pooled.index_add_(0, cry_elem_idx, messages)
            pooled_heads.append(pooled / normalizer.clamp_min(torch.finfo(messages.dtype).eps))
        return torch.stack(pooled_heads).mean(dim=0)


class LayeredFakeRoostBackbone(nn.Module):
    """Descriptor-only Roost backbone with the upstream module structure."""

    def __init__(self, output_dim: int = 4, n_graphs: int = 3, cry_heads: int = 2) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.elem_embedding = nn.Embedding(119, output_dim)
        self.material_nn = FakeRoostDescriptor(output_dim, n_graphs, cry_heads)

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        return self.material_nn(
            elem_weights,
            self.elem_embedding(elem_fea),
            self_idx,
            nbr_idx,
            cry_elem_idx,
        )


class OpaqueFakeRoostBackbone(nn.Module):
    """Five-tensor backbone without Aviary's partial-training internals."""

    def __init__(self, output_dim: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.elem_embedding = nn.Embedding(119, output_dim)
        self.material_nn = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: Tensor,
        nbr_idx: Tensor,
        cry_elem_idx: Tensor,
    ) -> Tensor:
        del self_idx, nbr_idx
        features = torch.tanh(self.material_nn(self.elem_embedding(elem_fea)))
        weighted = features * elem_weights
        num_materials = int(cry_elem_idx[-1].item()) + 1
        pooled = weighted.new_zeros((num_materials, self.output_dim))
        pooled.index_add_(0, cry_elem_idx, weighted)
        return pooled


def _element_ids() -> Tensor:
    return torch.tensor([26, 27, 28], dtype=torch.long)


def _fractions(dtype: torch.dtype = torch.double) -> Tensor:
    return torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.50, 0.20, 0.30],
            [0.40, 0.40, 0.20],
            [0.30, 0.50, 0.20],
            [0.20, 0.50, 0.30],
            [0.10, 0.60, 0.30],
            [0.70, 0.20, 0.10],
            [0.20, 0.30, 0.50],
        ],
        dtype=dtype,
    )


def _process(dtype: torch.dtype = torch.double) -> Tensor:
    return torch.tensor(
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
        dtype=dtype,
    )


def _data(*, with_process: bool) -> tuple[Tensor, Tensor]:
    fractions = _fractions()
    if with_process:
        process = _process()
        train_X = torch.cat((fractions, process), dim=-1)
        train_Y = (fractions[:, 0] - 0.4 * fractions[:, 2] + 0.001 * process[:, 0] + 0.05 * process[:, 1]).unsqueeze(-1)
    else:
        train_X = fractions
        train_Y = (fractions[:, 0] - 0.4 * fractions[:, 2]).unsqueeze(-1)
    return train_X, train_Y


def _model(*, with_process: bool, latent_dim: int = 3) -> RoostGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return RoostGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        latent_dim=latent_dim,
        outcome_transform=None,
    )


def _dkl_model(
    *,
    with_process: bool,
    encoder_training: str = "partial",
    trainable_encoder_layers: int = 1,
    n_graphs: int = 3,
    latent_dim: int = 4,
) -> RoostDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return RoostDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=LayeredFakeRoostBackbone(n_graphs=n_graphs),
        latent_dim=latent_dim,
        encoder_training=encoder_training,  # type: ignore[arg-type]
        trainable_encoder_layers=trainable_encoder_layers,
        outcome_transform=None,
    )


def test_composition_only_posterior_preserves_batch_q_and_sample_shapes() -> None:
    model = _model(with_process=False)
    fractions = _fractions()
    test_X = torch.stack((fractions[:2], fractions[2:4]), dim=0)

    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([4]))

    assert RoostGPModel.__module__.endswith(".deep.roost")
    assert model.composition_dim == 3
    assert model.process_dim == 0
    assert model.latent_dim == 3
    assert model.fusion.output_dim == model.material_encoder.output_dim
    assert model.input_transform is None
    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert posterior.variance.shape == torch.Size([2, 2, 1])
    assert samples.shape == torch.Size([4, 2, 2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(samples).all()


def test_one_dimensional_targets_are_normalized_to_botorch_shape() -> None:
    train_X, train_Y = _data(with_process=False)

    model = RoostGPModel(
        train_X=train_X,
        train_Y=train_Y.squeeze(-1),
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
    )
    posterior = model.posterior(train_X[:2])

    assert model.train_Y.shape == torch.Size([8, 1])
    assert posterior.mean.shape == torch.Size([2, 1])


def test_process_features_are_normalized_fused_and_differentiable() -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)
    transformed = model.transform_inputs(train_X)

    assert model.process_dim == 2
    assert model.fusion.output_dim == model.material_encoder.output_dim + 2
    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[..., :3], train_X[..., :3])
    assert (transformed[..., 3:] >= 0).all()
    assert (transformed[..., 3:] <= 1).all()

    model.train()
    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    test_X = train_X[:2].clone().requires_grad_(True)
    sample = model.posterior(test_X).rsample()
    (gradient,) = torch.autograd.grad(sample.sum(), test_X)
    composition_gradient = gradient[..., : model.composition_dim]
    tangent_gradient = composition_gradient - composition_gradient.mean(dim=-1, keepdim=True)

    assert torch.isfinite(gradient).all()
    assert tangent_gradient.abs().sum() > 1e-8
    assert gradient[..., model.composition_dim :].abs().sum() > 1e-8
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_dkl_partial_unfreezes_final_message_layers_and_crystal_pool() -> None:
    model = _dkl_model(
        with_process=True,
        encoder_training="partial",
        trainable_encoder_layers=2,
    )
    encoder = model.material_encoder.encoder
    descriptor = encoder.material_nn

    assert model.encoder_training == "partial"
    assert model.trainable_encoder_layers == 2
    assert not any(parameter.requires_grad for parameter in encoder.elem_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in descriptor.embedding.parameters())
    assert not any(parameter.requires_grad for parameter in descriptor.graphs[0].parameters())
    assert all(parameter.requires_grad for parameter in descriptor.graphs[1].parameters())
    assert all(parameter.requires_grad for parameter in descriptor.graphs[2].parameters())
    assert all(parameter.requires_grad for parameter in descriptor.cry_pool.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    model.train()
    assert not model.material_encoder.training
    assert not encoder.training
    assert not encoder.elem_embedding.training
    assert not descriptor.training
    assert not descriptor.embedding.training
    assert not descriptor.graphs[0].training
    assert descriptor.graphs[1].training
    assert descriptor.graphs[2].training
    assert all(pool.training for pool in descriptor.cry_pool)

    model.eval()
    assert not any(graph.training for graph in descriptor.graphs)
    assert not any(pool.training for pool in descriptor.cry_pool)


def test_dkl_full_unfreezes_element_embedding_and_complete_descriptor() -> None:
    model = _dkl_model(
        with_process=False,
        encoder_training="full",
    )
    encoder = model.material_encoder.encoder
    element_before = {name: parameter.detach().clone() for name, parameter in encoder.elem_embedding.named_parameters()}
    descriptor_before = {name: parameter.detach().clone() for name, parameter in encoder.material_nn.named_parameters()}

    assert model.encoder_training == "full"
    assert all(parameter.requires_grad for parameter in encoder.elem_embedding.parameters())
    assert all(parameter.requires_grad for parameter in encoder.material_nn.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    model.train()
    assert model.material_encoder.training
    assert encoder.training
    assert encoder.elem_embedding.training
    assert encoder.material_nn.training
    assert all(graph.training for graph in encoder.material_nn.graphs)
    assert all(pool.training for pool in encoder.material_nn.cry_pool)

    model.eval()
    assert not model.material_encoder.training
    assert not encoder.training

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)

    assert any(
        not torch.equal(dict(encoder.elem_embedding.named_parameters())[name], before)
        for name, before in element_before.items()
    )
    assert any(
        not torch.equal(dict(encoder.material_nn.named_parameters())[name], before)
        for name, before in descriptor_before.items()
    )


def test_dkl_full_supports_injected_backbone_without_partial_training_layout() -> None:
    train_X, train_Y = _data(with_process=False)
    model = RoostDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=OpaqueFakeRoostBackbone(),
        encoder_training="full",
        outcome_transform=None,
    )
    encoder = model.material_encoder.encoder
    test_X = train_X[:2].clone().requires_grad_(True)

    sample = model.posterior(test_X).rsample()
    gradients = torch.autograd.grad(
        sample.sum(),
        (test_X, *encoder.elem_embedding.parameters(), *encoder.material_nn.parameters()),
    )

    assert all(parameter.requires_grad for parameter in encoder.elem_embedding.parameters())
    assert all(parameter.requires_grad for parameter in encoder.material_nn.parameters())
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_dkl_fit_jointly_updates_selected_roost_projection_and_exact_gp() -> None:
    model = _dkl_model(
        with_process=True,
        encoder_training="partial",
        trainable_encoder_layers=1,
    )
    named_encoder_parameters = dict(model.material_encoder.named_parameters())
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in named_encoder_parameters.items()
        if not parameter.requires_grad
    }
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in named_encoder_parameters.items()
        if parameter.requires_grad
    }
    projection_before = {name: parameter.detach().clone() for name, parameter in model.projection.named_parameters()}
    gp_before = {
        name: parameter.detach().clone()
        for name, parameter in model.deepkernel.named_parameters()
        if not name.startswith("feature_extractor.")
    }

    fit_deepkernel_mll(model.make_mll(), num_epochs=3, lr=0.01)

    assert trainable_before
    assert all(
        torch.equal(parameter, frozen_before[name])
        for name, parameter in named_encoder_parameters.items()
        if name in frozen_before
    )
    assert any(not torch.equal(named_encoder_parameters[name], before) for name, before in trainable_before.items())
    assert any(
        not torch.equal(dict(model.projection.named_parameters())[name], before)
        for name, before in projection_before.items()
    )
    assert any(
        not torch.equal(dict(model.deepkernel.named_parameters())[name], before) for name, before in gp_before.items()
    )
    assert all(parameter.grad is None for name, parameter in named_encoder_parameters.items() if name in frozen_before)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in named_encoder_parameters.items()
        if name in trainable_before
    )


def test_dkl_qlogei_preserves_composition_and_process_input_gradients() -> None:
    model = _dkl_model(with_process=True)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=37),
    )
    test_X = torch.tensor(
        [[[0.40, 0.35, 0.25, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(test_X)
    (gradient,) = torch.autograd.grad(value.sum(), test_X)
    composition_gradient = gradient[..., : model.composition_dim]
    tangent_gradient = composition_gradient - composition_gradient.mean(
        dim=-1,
        keepdim=True,
    )

    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert tangent_gradient.abs().sum() > 1e-8
    assert gradient[..., model.composition_dim :].abs().sum() > 1e-8


def test_dkl_zero_fraction_inputs_keep_active_gradients_finite() -> None:
    model = _dkl_model(with_process=False)
    test_X = torch.tensor(
        [[0.7, 0.3, 0.0], [0.0, 0.6, 0.4]],
        dtype=torch.double,
        requires_grad=True,
    )

    projected = model.material_feature_extractor(test_X)
    posterior = model.posterior(test_X)
    posterior.rsample().sum().backward()

    assert projected.shape == torch.Size([2, model.latent_dim])
    assert torch.isfinite(projected).all()
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert test_X.grad[test_X.detach() > 0].abs().sum() > 1e-8


@pytest.mark.parametrize("latent_dim", [2, 5])
def test_dkl_latent_projection_width_is_configurable(latent_dim: int) -> None:
    model = _dkl_model(
        with_process=True,
        latent_dim=latent_dim,
    )
    train_X, _ = _data(with_process=True)

    projected = model.material_feature_extractor(model.transform_inputs(train_X[:2]))

    assert projected.shape == torch.Size([2, latent_dim])
    assert model.latent_dim == latent_dim
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == latent_dim


def test_dkl_save_load_preserves_policy_and_posterior(tmp_path: Path) -> None:
    model = _dkl_model(
        with_process=True,
        encoder_training="partial",
        trainable_encoder_layers=2,
    )
    train_X, _ = _data(with_process=True)
    model.eval()
    reference = model.posterior(train_X[:2])
    path = tmp_path / "roost_dkl.pt"

    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_posterior = restored.posterior(train_X[:2])

    assert isinstance(restored, RoostDKLModel)
    assert restored.encoder_training == "partial"
    assert restored.trainable_encoder_layers == 2
    torch.testing.assert_close(restored_posterior.mean, reference.mean)
    torch.testing.assert_close(restored_posterior.variance, reference.variance)

    restored.train()
    descriptor = restored.material_encoder.encoder.material_nn
    assert not restored.material_encoder.training
    assert not descriptor.graphs[0].training
    assert descriptor.graphs[1].training
    assert descriptor.graphs[2].training
    assert all(pool.training for pool in descriptor.cry_pool)


def test_roost_dkl_matches_crabnet_dkl_canonical_constructor_surface() -> None:
    roost_parameters = inspect.signature(RoostDKLModel).parameters
    crabnet_parameters = inspect.signature(CrabNetDKLModel).parameters
    shared_parameters = {
        "train_X",
        "train_Y",
        "train_Yvar",
        "element_ids",
        "encoder",
        "checkpoint",
        "latent_dim",
        "fusion",
        "projection",
        "strict_checkpoint",
        "trainable_encoder_layers",
        "likelihood",
        "input_transform",
        "outcome_transform",
    }

    assert shared_parameters <= roost_parameters.keys()
    assert shared_parameters <= crabnet_parameters.keys()
    assert roost_parameters["trainable_encoder_layers"].default == 1
    assert crabnet_parameters["trainable_encoder_layers"].default == 1
    assert "encoder_training" in roost_parameters


def test_model_supports_qlogei_qucb_and_qlognei() -> None:
    model = _model(with_process=True)
    train_X, train_Y = _data(with_process=True)
    fit_deepkernel_mll(model.make_mll(), num_epochs=2)
    candidates = train_X[:2].unsqueeze(0)
    acquisitions = [
        qLogExpectedImprovement(model=model, best_f=train_Y.max()),
        qUpperConfidenceBound(model=model, beta=0.2),
        qLogNoisyExpectedImprovement(model=model, X_baseline=train_X),
    ]

    values = [acquisition(candidates) for acquisition in acquisitions]

    assert all(value.shape == torch.Size([1]) for value in values)
    assert all(torch.isfinite(value).all() for value in values)
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_qlogei_gradient_spans_simplex_tangent_and_process_directions() -> None:
    model = _model(with_process=True, latent_dim=4)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=23),
    )
    test_X = torch.tensor(
        [[[0.40, 0.35, 0.25, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(test_X)
    (gradient,) = torch.autograd.grad(value.sum(), test_X)
    composition_gradient = gradient[..., : model.composition_dim]
    simplex_tangent_gradient = composition_gradient - composition_gradient.mean(
        dim=-1,
        keepdim=True,
    )
    process_gradient = gradient[..., model.composition_dim :]

    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert simplex_tangent_gradient.abs().sum() > 1e-8
    assert process_gradient.abs().sum() > 1e-8


def test_optimize_acqf_jointly_optimizes_packed_composition_and_process() -> None:
    model = _model(with_process=True, latent_dim=4)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=29),
    )
    bounds = torch.tensor(
        [
            [0.05, 0.05, 0.05, 850.0, 0.5],
            [0.80, 0.80, 0.80, 1300.0, 6.0],
        ],
        dtype=torch.double,
    )
    composition_constraint = (
        torch.arange(model.composition_dim, device=bounds.device, dtype=torch.long),
        torch.ones(model.composition_dim, device=bounds.device, dtype=bounds.dtype),
        1.0,
    )

    candidates, acq_value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=3,
        num_restarts=10,
        raw_samples=256,
        equality_constraints=[composition_constraint],
        options={"batch_limit": 5, "maxiter": 50},
    )
    posterior = model.posterior(candidates)

    assert candidates.shape == torch.Size([3, 5])
    assert acq_value is not None
    assert acq_value.shape == torch.Size([])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(acq_value)
    assert torch.all(candidates >= bounds[0] - 1e-7)
    assert torch.all(candidates <= bounds[1] + 1e-7)
    torch.testing.assert_close(
        candidates[..., : model.composition_dim].sum(dim=-1),
        torch.ones(3, dtype=candidates.dtype),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


def test_custom_projection_controls_gp_latent_width() -> None:
    train_X, train_Y = _data(with_process=True)
    projection = nn.Linear(6, 5).double()
    model = RoostGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(output_dim=4),
        latent_dim=5,
        projection=projection,
        outcome_transform=None,
    )

    projected = model.material_feature_extractor(model.transform_inputs(train_X[:2]))

    assert model.projection is projection
    assert projected.shape == torch.Size([2, 5])
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == 5


def test_composition_transform_preserves_ilr_and_process_gradients() -> None:
    raw_train_X = torch.tensor(
        [
            [-0.5, -0.2, 900.0, 1.0],
            [-0.3, 0.1, 950.0, 2.0],
            [-0.1, 0.3, 1000.0, 3.0],
            [0.1, -0.3, 1050.0, 4.0],
            [0.2, 0.2, 1100.0, 2.0],
            [0.3, -0.1, 1150.0, 3.0],
            [0.4, 0.4, 1200.0, 5.0],
            [0.6, -0.4, 1250.0, 4.0],
        ],
        dtype=torch.double,
    )
    train_Y = (raw_train_X[:, 0] - raw_train_X[:, 1] + 0.001 * raw_train_X[:, 2]).unsqueeze(-1)
    transform = CompositionMaterialInputTransform(
        input_dim=4,
        composition_indices=[0, 1],
        n_components=3,
        method="ilr",
        process_bounds=torch.tensor([[850.0, 0.5], [1300.0, 6.0]], dtype=torch.double),
    ).double()
    model = RoostGPModel(
        train_X=raw_train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        latent_dim=4,
        input_transform=transform,
        outcome_transform=None,
    )
    test_X = raw_train_X[:2].clone().requires_grad_(True)

    posterior = model.posterior(test_X)
    (gradient,) = torch.autograd.grad(posterior.rsample().sum(), test_X)
    packed = model.transform_inputs(test_X)

    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(gradient).all()
    assert gradient[..., :2].abs().sum() > 1e-8
    assert gradient[..., 2:].abs().sum() > 1e-8
    torch.testing.assert_close(
        packed[..., :3].sum(dim=-1),
        torch.ones(2, dtype=torch.double),
    )


def test_optimize_acqf_in_ilr_coordinates_decodes_formula_candidates() -> None:
    torch.manual_seed(0)
    fractions = _fractions()
    process = _process()
    formulas = [
        "Fe0.6Co0.3Ni0.1",
        "Fe0.5Co0.2Ni0.3",
        "Fe0.4Co0.4Ni0.2",
        "Fe0.3Co0.5Ni0.2",
        "Fe0.2Co0.5Ni0.3",
        "Fe0.1Co0.6Ni0.3",
        "Fe0.7Co0.2Ni0.1",
        "Fe0.2Co0.3Ni0.5",
    ]
    formula_transformer = CompositionTransformer(
        elements=("Fe", "Co", "Ni"),
        representation="ilr",
        prefix="alloy",
        precision=8,
    )
    composition_coordinates = torch.as_tensor(
        formula_transformer.fit_transform(formulas),
        dtype=torch.double,
    )
    train_X = torch.cat((composition_coordinates, process), dim=-1)
    train_Y = (fractions[:, 0] - 0.4 * fractions[:, 2] + 0.001 * process[:, 0] + 0.05 * process[:, 1]).unsqueeze(-1)
    input_transform = CompositionMaterialInputTransform(
        input_dim=4,
        composition_indices=[0, 1],
        n_components=3,
        method="ilr",
        process_bounds=torch.tensor(
            [[850.0, 0.5], [1300.0, 6.0]],
            dtype=torch.double,
        ),
    ).double()
    model = RoostGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        latent_dim=4,
        input_transform=input_transform,
        outcome_transform=None,
    )
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=31),
    )
    coordinate_lower = composition_coordinates.min(dim=0).values - 0.25
    coordinate_upper = composition_coordinates.max(dim=0).values + 0.25
    bounds = torch.stack(
        (
            torch.cat((coordinate_lower, torch.tensor([850.0, 0.5], dtype=torch.double))),
            torch.cat((coordinate_upper, torch.tensor([1300.0, 6.0], dtype=torch.double))),
        )
    )

    candidates, acq_value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=8,
        raw_samples=128,
        options={"batch_limit": 4, "maxiter": 50},
    )
    decoded_formulas = formula_transformer.inverse_transform(candidates[..., :2].detach().cpu().numpy())
    roundtrip_coordinates = torch.as_tensor(
        formula_transformer.transform(decoded_formulas),
        dtype=candidates.dtype,
        device=candidates.device,
    )
    gradient_X = candidates.detach().unsqueeze(0).requires_grad_(True)
    gradient_value = acquisition(gradient_X)
    (gradient,) = torch.autograd.grad(gradient_value.sum(), gradient_X)
    posterior = model.posterior(candidates)

    assert candidates.shape == torch.Size([2, 4])
    assert len(decoded_formulas) == 2
    assert all(formula.startswith("Fe") and "Co" in formula and "Ni" in formula for formula in decoded_formulas)
    torch.testing.assert_close(
        roundtrip_coordinates,
        candidates[..., :2],
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.isfinite(acq_value)
    assert torch.isfinite(gradient).all()
    assert gradient[..., :2].abs().sum() > 1e-8
    assert gradient[..., 2:].abs().sum() > 1e-8
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_zero_fraction_elements_remain_finite() -> None:
    model = _model(with_process=False)
    test_X = torch.tensor(
        [[0.7, 0.3, 0.0], [0.0, 0.6, 0.4]],
        dtype=torch.double,
    )

    projected = model.material_feature_extractor(test_X)
    posterior = model.posterior(test_X)

    assert projected.shape == torch.Size([2, model.latent_dim])
    assert torch.isfinite(projected).all()
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_checkpoint_is_loaded_before_encoder_freezing() -> None:
    source = FakeRoostBackbone()
    with torch.no_grad():
        source.scale.fill_(5.0)
    checkpoint = {
        "state_dict": {f"encoder.{name}": value.detach().clone() for name, value in source.state_dict().items()}
    }
    train_X, train_Y = _data(with_process=False)

    model = RoostGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeRoostBackbone(),
        checkpoint=checkpoint,
        latent_dim=2,
        outcome_transform=None,
    )

    assert model.material_encoder.initialization == "checkpoint"
    torch.testing.assert_close(
        model.material_encoder.encoder.scale,
        torch.full((4,), 5.0, dtype=torch.double),
    )
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


def test_full_model_save_load_restores_posterior_and_metadata(tmp_path: Path) -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)
    model.eval()
    reference = model.posterior(train_X[:2])
    path = tmp_path / "roost_gp.pt"

    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_posterior = restored.posterior(train_X[:2])

    assert isinstance(restored, RoostGPModel)
    assert torch.equal(restored.element_ids, _element_ids())
    assert restored.composition_dim == 3
    assert restored.process_dim == 2
    assert restored.latent_dim == model.latent_dim
    assert restored.material_encoder.initialization == "injected"
    torch.testing.assert_close(restored_posterior.mean, reference.mean)
    torch.testing.assert_close(restored_posterior.variance, reference.variance)

    state_keys = set(restored.state_dict())
    assert "deepkernel.feature_extractor.element_ids" in state_keys
    assert "deepkernel.feature_extractor.material_encoder.encoder.scale" in state_keys
    assert "deepkernel.feature_extractor.projection.weight" in state_keys
    assert "deepkernel.covar_module.raw_outputscale" in state_keys


def test_to_dtype_moves_encoder_projection_gp_and_likelihood() -> None:
    model = _model(with_process=True).float()
    train_X, _ = _data(with_process=True)

    posterior = model.posterior(train_X[:2].float())

    assert posterior.mean.dtype == torch.float32
    assert model.element_ids.dtype == torch.long
    assert model.material_encoder.encoder.scale.dtype == torch.float32
    assert next(model.projection.parameters()).dtype == torch.float32
    assert next(model.deepkernel.covar_module.parameters()).dtype == torch.float32
    assert next(model.likelihood.parameters()).dtype == torch.float32


def test_model_surfaces_optional_aviary_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_aviary(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(roost_module, "import_module", missing_aviary)
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ImportError, match="optional Roost/materials dependency"):
        RoostGPModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
        )


def test_model_rejects_unsupported_or_invalid_inputs() -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="single-output"):
        RoostGPModel(
            train_X=train_X,
            train_Y=torch.cat((train_Y, train_Y), dim=-1),
            element_ids=_element_ids(),
            encoder=FakeRoostBackbone(),
        )
    with pytest.raises(NotImplementedError, match="train_Yvar"):
        RoostGPModel(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, 0.01),
            element_ids=_element_ids(),
            encoder=FakeRoostBackbone(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        RoostGPModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=torch.tensor([26, 26, 28]),
            encoder=FakeRoostBackbone(),
        )

    invalid_fractions = train_X.clone()
    invalid_fractions[0, 0] += 0.2
    with pytest.raises(ValueError, match="sum to one"):
        RoostGPModel(
            train_X=invalid_fractions,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=FakeRoostBackbone(),
        )

    adapter = RoostEncoder(FakeRoostBackbone())
    with pytest.raises(ValueError, match="encoder_output_dim"):
        RoostGPModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=adapter,
            encoder_output_dim=adapter.output_dim + 1,
        )


@pytest.mark.parametrize("trainable_encoder_layers", [0, True, -1])
def test_dkl_rejects_invalid_trainable_message_layer_count(
    trainable_encoder_layers: object,
) -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="positive integer"):
        RoostDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=LayeredFakeRoostBackbone(),
            trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        )


def test_dkl_rejects_invalid_mode_or_missing_roost_descriptor_structure() -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="encoder_training"):
        RoostDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=LayeredFakeRoostBackbone(),
            encoder_training="frozen",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="encoder.material_nn"):
        RoostDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=FakeRoostBackbone(),
        )
    with pytest.raises(ValueError, match="exceeds the number of Roost"):
        RoostDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=LayeredFakeRoostBackbone(n_graphs=2),
            trainable_encoder_layers=3,
        )


def test_real_aviary_roost_gp_runs_on_cpu_when_materials_extra_is_installed() -> None:
    pytest.importorskip("aviary.roost.model")
    material_encoder = RoostEncoder(
        elem_fea_len=8,
        n_graph=1,
        elem_heads=1,
        elem_gate=(8,),
        elem_msg=(8,),
        cry_heads=1,
        cry_gate=(8,),
        cry_msg=(8,),
    ).double()
    train_X, train_Y = _data(with_process=False)
    model = RoostGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=material_encoder,
        latent_dim=4,
        outcome_transform=None,
    )
    test_X = train_X[:2].clone().requires_grad_(True)

    posterior = model.posterior(test_X)
    posterior.mean.sum().backward()

    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert test_X.grad.abs().sum() > 0
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


def test_real_aviary_roost_dkl_partially_unfreezes_and_runs_on_cpu() -> None:
    pytest.importorskip("aviary.roost.model")
    material_encoder = RoostEncoder(
        elem_fea_len=8,
        n_graph=2,
        elem_heads=1,
        elem_gate=(8,),
        elem_msg=(8,),
        cry_heads=1,
        cry_gate=(8,),
        cry_msg=(8,),
    ).double()
    train_X, train_Y = _data(with_process=False)
    model = RoostDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=material_encoder,
        latent_dim=4,
        encoder_training="partial",
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    raw_encoder = model.material_encoder.encoder
    test_X = train_X[:2].clone().requires_grad_(True)

    model.train()
    assert not any(parameter.requires_grad for parameter in raw_encoder.elem_embedding.parameters())
    assert not any(parameter.requires_grad for parameter in raw_encoder.material_nn.embedding.parameters())
    assert not any(parameter.requires_grad for parameter in raw_encoder.material_nn.graphs[0].parameters())
    assert all(parameter.requires_grad for parameter in raw_encoder.material_nn.graphs[1].parameters())
    assert all(parameter.requires_grad for parameter in raw_encoder.material_nn.cry_pool.parameters())
    assert not raw_encoder.material_nn.graphs[0].training
    assert raw_encoder.material_nn.graphs[1].training
    assert all(pool.training for pool in raw_encoder.material_nn.cry_pool)

    posterior = model.posterior(test_X)
    posterior.rsample().sum().backward()

    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert test_X.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in raw_encoder.material_nn.graphs[1].parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in raw_encoder.material_nn.cry_pool.parameters()
    )
