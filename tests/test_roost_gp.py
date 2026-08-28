from __future__ import annotations

from pathlib import Path

import pytest
import torch
from botorch.models.transforms.input import Normalize
from torch import Tensor, nn

import bochan.composition.encoders.roost as roost_module
from bochan.composition import RoostEncoder
from bochan.models.regression.gaussian.deep import (
    CompositionMaterialInputTransform,
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
