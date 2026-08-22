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
from torch import Tensor, nn

from bochan.composition import CrabNetEncoder
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.regression.gaussian.deep import CrabNetGPModel


class FakeCrabNet(nn.Module):
    """Small differentiable upstream encoder for frozen-feature model tests."""

    def __init__(self, d_model: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.arange(1, d_model + 1, dtype=torch.float32))

    def forward(self, element_ids: Tensor, fractions: Tensor) -> Tensor:
        del element_ids
        return fractions.unsqueeze(-1) * self.scale


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
    assert isinstance(model.input_transform, Normalize)
    assert torch.equal(transformed[..., :3], train_X[..., :3])
    assert (transformed[..., 3:] >= 0).all()
    assert (transformed[..., 3:] <= 1).all()

    same_material = train_X[:1].repeat(2, 1)
    same_material[1, 3:] = train_X[-1, 3:]
    projected = model.crabnet_feature_extractor(model.transform_inputs(same_material))
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


def test_zero_fraction_elements_are_converted_to_crabnet_padding() -> None:
    model = _model(with_process=False)
    test_X = torch.tensor(
        [[0.7, 0.3, 0.0], [0.0, 0.6, 0.4]],
        dtype=torch.double,
    )

    projected = model.crabnet_feature_extractor(test_X)
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

    projected = model.crabnet_feature_extractor(train_X[:2])

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
    with pytest.raises(NotImplementedError, match="train_Yvar"):
        CrabNetGPModel(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, 0.01),
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
