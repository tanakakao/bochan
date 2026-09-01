from __future__ import annotations

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

from bochan.composition import CrabNetEncoder, TorchSimplexTransform
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import CrabNetDKLModel, CrabNetGPModel


class FakeCrabNet(nn.Module):
    """Small differentiable upstream encoder for frozen-feature model tests."""

    def __init__(self, d_model: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.arange(1, d_model + 1, dtype=torch.float32))

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        feature_scale = torch.arange(
            1,
            self.d_model + 1,
            device=fractions.device,
            dtype=fractions.dtype,
        )
        element_signal = element_ids.to(dtype=fractions.dtype).unsqueeze(-1) / (100 * feature_scale)
        return fractions.unsqueeze(-1) * self.scale * (1 + element_signal)


class FakeTransformerStack(nn.Module):
    """Minimal upstream-style Transformer stack for partial-unfreeze tests."""

    def __init__(self, d_model: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(num_layers))


class LayeredFakeCrabNet(nn.Module):
    """Differentiable encoder exposing CrabNet's Transformer layer structure."""

    def __init__(self, d_model: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.transformer_encoder = FakeTransformerStack(d_model, num_layers)

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        element_signal = element_ids.to(dtype=fractions.dtype).unsqueeze(-1) / 100
        features = self.embedding(fractions.unsqueeze(-1) * (1 + element_signal))
        for layer in self.transformer_encoder.layers:
            features = torch.tanh(layer(features))
        return features * fractions.unsqueeze(-1)


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
        train_X = torch.cat([fractions, process], dim=-1)
        train_Y = (fractions[:, 0] - 0.4 * fractions[:, 2] + 0.001 * process[:, 0] + 0.05 * process[:, 1]).unsqueeze(-1)
    else:
        train_X = fractions
        train_Y = (fractions[:, 0] - 0.4 * fractions[:, 2]).unsqueeze(-1)
    return train_X, train_Y


def _model(*, with_process: bool, latent_dim: int = 3) -> CrabNetGPModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return CrabNetGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeCrabNet(),
        latent_dim=latent_dim,
        outcome_transform=None,
    )


def _dkl_model(
    *,
    with_process: bool,
    trainable_encoder_layers: int | str = 1,
) -> CrabNetDKLModel:
    torch.manual_seed(0)
    train_X, train_Y = _data(with_process=with_process)
    return CrabNetDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=LayeredFakeCrabNet(),
        latent_dim=4,
        trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        outcome_transform=None,
    )


def test_composition_only_posterior_preserves_batch_q_and_sample_shapes() -> None:
    model = _model(with_process=False)
    fractions = _fractions()
    test_X = torch.stack([fractions[:2], fractions[2:4]], dim=0)

    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([4]))

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


def test_process_features_are_normalized_separately_and_fused() -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)
    transformed = model.transform_inputs(train_X)

    assert model.process_dim == 2
    assert model.fusion.output_dim == model.material_encoder.output_dim + 2
    assert not hasattr(model, "crabnet_feature_extractor")
    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[..., :3], train_X[..., :3])
    assert (transformed[..., 3:] >= 0).all()
    assert (transformed[..., 3:] <= 1).all()

    same_material = train_X[:1].repeat(2, 1)
    same_material[1, 3:] = train_X[-1, 3:]
    projected = model.material_feature_extractor(model.transform_inputs(same_material))
    assert not torch.allclose(projected[0], projected[1])


def test_encoder_is_frozen_and_input_gradients_are_preserved() -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)

    model.train()
    assert not model.material_encoder.training
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    test_X = train_X[:2].clone().requires_grad_(True)
    sample = model.posterior(test_X).rsample()
    sample.sum().backward()

    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
    assert test_X.grad[..., :3].abs().sum() > 0
    assert test_X.grad[..., 3:].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_dkl_partial_unfreeze_selects_final_transformer_layers_and_modes() -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=2)
    named_parameters = dict(model.material_encoder.named_parameters())
    layers = model.material_encoder.encoder.transformer_encoder.layers

    assert model.trainable_encoder_layers == 2
    assert not named_parameters["encoder.embedding.weight"].requires_grad
    assert not named_parameters["encoder.transformer_encoder.layers.0.weight"].requires_grad
    assert named_parameters["encoder.transformer_encoder.layers.1.weight"].requires_grad
    assert named_parameters["encoder.transformer_encoder.layers.2.weight"].requires_grad
    assert all(parameter.requires_grad for parameter in model.projection.parameters())

    model.train()
    assert not model.material_encoder.training
    assert not layers[0].training
    assert layers[1].training
    assert layers[2].training

    model.eval()
    assert not any(layer.training for layer in layers)


def test_dkl_full_unfreeze_trains_complete_injected_encoder() -> None:
    train_X, train_Y = _data(with_process=False)
    model = CrabNetDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeCrabNet(),
        latent_dim=3,
        trainable_encoder_layers="all",
        outcome_transform=None,
    )

    assert model.trainable_encoder_layers == "all"
    assert all(parameter.requires_grad for parameter in model.material_encoder.parameters())
    encoder_before = model.material_encoder.encoder.scale.detach().clone()

    model.train()
    assert model.material_encoder.training
    model.eval()
    assert not model.material_encoder.training

    fit_deepkernel_mll(model.make_mll(), num_epochs=2, lr=0.01)
    assert not torch.equal(model.material_encoder.encoder.scale, encoder_before)


def test_dkl_fit_jointly_updates_encoder_projection_and_exact_gp() -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=1)
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
    model = _dkl_model(with_process=True, trainable_encoder_layers=1)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=19),
    )
    test_X = torch.tensor(
        [[[0.40, 0.35, 0.25, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )

    value = acquisition(test_X)
    (gradient,) = torch.autograd.grad(value.sum(), test_X)
    posterior = model.posterior(test_X)
    samples = posterior.rsample(sample_shape=torch.Size([3]))

    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert gradient[..., : model.composition_dim].abs().sum() > 1e-8
    assert gradient[..., model.composition_dim :].abs().sum() > 1e-8
    assert posterior.mean.shape == torch.Size([1, 1, 1])
    assert samples.shape == torch.Size([3, 1, 1, 1])
    assert torch.isfinite(samples).all()


def test_dkl_save_load_preserves_policy_and_posterior(tmp_path: Path) -> None:
    model = _dkl_model(with_process=True, trainable_encoder_layers=1)
    train_X, _ = _data(with_process=True)
    model.eval()
    reference = model.posterior(train_X[:2])
    path = tmp_path / "crabnet_dkl.pt"

    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_posterior = restored.posterior(train_X[:2])

    assert isinstance(restored, CrabNetDKLModel)
    assert restored.trainable_encoder_layers == 1
    assert torch.allclose(restored_posterior.mean, reference.mean)
    assert torch.allclose(restored_posterior.variance, reference.variance)
    restored.train()
    layers = restored.material_encoder.encoder.transformer_encoder.layers
    assert not restored.material_encoder.training
    assert not layers[0].training
    assert not layers[1].training
    assert layers[2].training


def test_zero_fraction_elements_are_converted_to_crabnet_padding() -> None:
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


def test_model_checkpoint_is_loaded_before_encoder_freezing() -> None:
    train_X, train_Y = _data(with_process=False)
    model = CrabNetGPModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=FakeCrabNet(),
        checkpoint={"weights": {"encoder.scale": torch.full((4,), 5.0)}},
        latent_dim=2,
        outcome_transform=None,
    )

    assert model.material_encoder.initialization == "checkpoint"
    assert torch.equal(model.material_encoder.encoder.scale, torch.full((4,), 5.0, dtype=torch.double))
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


@pytest.mark.parametrize("latent_dim", [16, 32, 64])
def test_latent_projection_width_is_configurable(latent_dim: int) -> None:
    model = _model(with_process=False, latent_dim=latent_dim)
    train_X, _ = _data(with_process=False)

    projected = model.material_feature_extractor(train_X[:2])

    assert projected.shape == torch.Size([2, latent_dim])
    assert model.deepkernel.covar_module.base_kernel.ard_num_dims == latent_dim


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
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=11),
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


def test_qlogei_gradient_flows_from_ilr_coordinates_through_crabnet_gp() -> None:
    model = _model(with_process=True, latent_dim=4)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=13),
    )
    coordinate_X = torch.tensor(
        [[[0.25, -0.15, 1075.0, 3.0]]],
        dtype=torch.double,
        requires_grad=True,
    )
    inverse_ilr = TorchSimplexTransform(
        n_components=model.composition_dim,
        method="ilr",
    ).to(coordinate_X)
    fractions = inverse_ilr(coordinate_X[..., : inverse_ilr.input_dim])
    model_X = torch.cat(
        [fractions, coordinate_X[..., inverse_ilr.input_dim :]],
        dim=-1,
    )

    value = acquisition(model_X)
    (gradient,) = torch.autograd.grad(value.sum(), coordinate_X)

    assert torch.isfinite(value).all()
    assert torch.isfinite(gradient).all()
    assert gradient[..., : inverse_ilr.input_dim].abs().sum() > 1e-8
    assert gradient[..., inverse_ilr.input_dim :].abs().sum() > 1e-8
    torch.testing.assert_close(
        fractions.sum(dim=-1),
        torch.ones((1, 1), dtype=fractions.dtype),
    )
    assert all(parameter.grad is None for parameter in model.material_encoder.parameters())


def test_optimize_acqf_jointly_optimizes_composition_and_process() -> None:
    model = _model(with_process=True, latent_dim=4)
    _, train_Y = _data(with_process=True)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=17),
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
    assert torch.allclose(
        candidates[..., : model.composition_dim].sum(dim=-1),
        torch.ones(3, dtype=candidates.dtype),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


def test_full_model_save_load_restores_posterior_and_metadata(tmp_path: Path) -> None:
    model = _model(with_process=True)
    train_X, _ = _data(with_process=True)
    model.eval()
    reference = model.posterior(train_X[:2])
    path = tmp_path / "crabnet_gp.pt"

    torch.save(model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    restored_posterior = restored.posterior(train_X[:2])

    assert isinstance(restored, CrabNetGPModel)
    assert torch.equal(restored.element_ids, _element_ids())
    assert restored.composition_dim == 3
    assert restored.process_dim == 2
    assert restored.latent_dim == model.latent_dim
    assert restored.material_encoder.initialization == "injected"
    assert torch.allclose(restored_posterior.mean, reference.mean)
    assert torch.allclose(restored_posterior.variance, reference.variance)

    state_keys = set(restored.state_dict())
    assert "deepkernel.feature_extractor.element_ids" in state_keys
    assert "deepkernel.feature_extractor.material_encoder.encoder.scale" in state_keys
    assert "deepkernel.feature_extractor.projection.weight" in state_keys
    assert "deepkernel.covar_module.raw_outputscale" in state_keys


def test_to_dtype_moves_encoder_projection_fusion_gp_and_likelihood() -> None:
    model = _model(with_process=True).float()
    train_X, _ = _data(with_process=True)
    posterior = model.posterior(train_X[:2].float())

    assert posterior.mean.dtype == torch.float32
    assert model.element_ids.dtype == torch.long
    assert model.material_encoder.encoder.scale.dtype == torch.float32
    assert next(model.projection.parameters()).dtype == torch.float32
    assert next(model.deepkernel.covar_module.parameters()).dtype == torch.float32
    assert next(model.likelihood.parameters()).dtype == torch.float32


def test_model_rejects_unsupported_or_invalid_inputs() -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="single-output"):
        CrabNetGPModel(
            train_X=train_X,
            train_Y=torch.cat([train_Y, train_Y], dim=-1),
            element_ids=_element_ids(),
            encoder=FakeCrabNet(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        CrabNetGPModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=torch.tensor([26, 26, 28]),
            encoder=FakeCrabNet(),
        )

    invalid_fractions = train_X.clone()
    invalid_fractions[0, 0] += 0.2
    with pytest.raises(ValueError, match="sum to one"):
        CrabNetGPModel(
            train_X=invalid_fractions,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=FakeCrabNet(),
        )

    with pytest.raises(ValueError, match="sum to one"):
        CrabNetGPModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=FakeCrabNet(),
            input_transform=Normalize(d=3),
        )


@pytest.mark.parametrize("trainable_encoder_layers", [0, True, "partial"])
def test_dkl_rejects_invalid_unfreeze_configuration(
    trainable_encoder_layers: object,
) -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="positive integer or 'all'"):
        CrabNetDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=LayeredFakeCrabNet(),
            trainable_encoder_layers=trainable_encoder_layers,  # type: ignore[arg-type]
        )


def test_dkl_partial_unfreeze_requires_explicit_transformer_structure() -> None:
    train_X, train_Y = _data(with_process=False)

    with pytest.raises(ValueError, match="transformer_encoder.layers"):
        CrabNetDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=FakeCrabNet(),
            trainable_encoder_layers=1,
        )

    with pytest.raises(ValueError, match="exceeds the number"):
        CrabNetDKLModel(
            train_X=train_X,
            train_Y=train_Y,
            element_ids=_element_ids(),
            encoder=LayeredFakeCrabNet(num_layers=2),
            trainable_encoder_layers=3,
        )


def test_real_crabnet_gp_runs_on_cpu_when_materials_extra_is_installed() -> None:
    pytest.importorskip("crabnet.kingcrab")
    material_encoder = CrabNetEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        pe_resolution=32,
        ple_resolution=32,
    )
    train_X, train_Y = _data(with_process=False)
    model = CrabNetGPModel(
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
    assert not any(parameter.requires_grad for parameter in model.material_encoder.parameters())


def test_real_crabnet_dkl_partially_unfreezes_and_runs_on_cpu() -> None:
    pytest.importorskip("crabnet.kingcrab")
    material_encoder = CrabNetEncoder(
        d_model=8,
        num_layers=2,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        pe_resolution=32,
        ple_resolution=32,
    )
    train_X, train_Y = _data(with_process=False)
    model = CrabNetDKLModel(
        train_X=train_X,
        train_Y=train_Y,
        element_ids=_element_ids(),
        encoder=material_encoder,
        latent_dim=4,
        trainable_encoder_layers=1,
        outcome_transform=None,
    )
    named_parameters = dict(model.material_encoder.named_parameters())

    fit_deepkernel_mll(model.make_mll(), num_epochs=1, lr=0.001)
    posterior = model.posterior(train_X[:2])

    assert any(
        parameter.requires_grad
        for name, parameter in named_parameters.items()
        if "transformer_encoder.layers.1" in name
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in named_parameters.items()
        if "transformer_encoder.layers.0" in name
    )
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
