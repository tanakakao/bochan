"""Tests for correlated variational Gamma multi-task regression."""

from __future__ import annotations

import torch
from botorch.sampling.get_sampler import get_sampler

from bochan.api.configs import FitConfig, ModelConfig
from bochan.api.factory import build_model, fit_model
from bochan.api.model_registry import MODEL_REGISTRY
from bochan.models.regression.gamma.base import (
    GammaMultiTaskGPModel,
    WideGammaMultiTaskGPModel,
)
from bochan.models.transforms.outcome import PositiveScaleOutcomeTransform


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small correlated positive data set."""
    x = torch.linspace(0.05, 0.95, 7, dtype=torch.double).unsqueeze(-1)
    y = torch.cat([1.0 + x, 2.0 + 1.5 * x, 0.5 + 0.7 * x], dim=-1)
    return x, y


def test_gamma_multitask_shapes_correlation_sampling_and_gradient() -> None:
    """The public posterior preserves q/task axes and differentiability."""
    train_x, train_y = _data()
    model = WideGammaMultiTaskGPModel(
        train_x,
        train_y,
        rank=2,
        num_latents=2,
        num_inducing_points=8,
        outcome_transform=PositiveScaleOutcomeTransform(validate_positive=True),
    )
    candidate = torch.tensor([[0.2], [0.8]], dtype=torch.double, requires_grad=True)
    posterior = model.posterior(candidate)
    assert posterior.mean.shape == torch.Size([2, 3])
    assert posterior.rsample(torch.Size([4])).shape == torch.Size([4, 2, 3])
    assert torch.isfinite(posterior.mean).all() and (posterior.mean > 0).all()
    assert torch.isfinite(posterior.variance).all() and (posterior.variance >= 0).all()
    assert model.task_covar_matrix.shape == torch.Size([3, 3])
    assert torch.isfinite(model.task_covar_matrix).all()
    posterior.mean.sum().backward()
    assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
    sampler = get_sampler(posterior, sample_shape=torch.Size([3]))
    assert sampler(posterior).shape == torch.Size([3, 2, 3])


def test_wide_gamma_multitask_omits_missing_cells_and_rejects_empty_task() -> None:
    """NaN cells are excluded rather than imputed and task order is retained."""
    train_x, train_y = _data()
    train_y[0, 2] = torch.nan
    train_y[3, 0] = torch.nan
    model = WideGammaMultiTaskGPModel(
        train_x,
        train_y,
        rank=2,
        num_inducing_points=8,
        outcome_transform=PositiveScaleOutcomeTransform(validate_positive=True),
    )
    assert int(model.observed_mask.sum()) == train_y.numel() - 2
    assert model.model.train_targets.numel() == train_y.numel() - 2
    assert model.posterior(train_x[:2]).mean.shape == torch.Size([2, 3])

    train_y[:, 1] = torch.nan
    try:
        WideGammaMultiTaskGPModel(train_x, train_y)
    except ValueError as error:
        assert "empty tasks" in str(error)
    else:
        raise AssertionError("An entirely missing task must be rejected.")


def test_gamma_registry_factory_and_variational_fit() -> None:
    """The high-level factory selects and fits the correlated Gamma model."""
    train_x, train_y = _data()
    assert MODEL_REGISTRY["normal"]["regression"]["gamma_multitask"] is GammaMultiTaskGPModel
    config = ModelConfig(
        task_type="regression",
        model_type="gamma_wide_multitask",
        model_kwargs={"rank": 2, "num_latents": 2, "num_inducing_points": 8},
    )
    bundle = build_model(train_x, train_y, config)
    assert isinstance(bundle.model, WideGammaMultiTaskGPModel)
    fit_model(bundle, FitConfig(num_epochs=2, lr=0.01))
    assert bundle.metadata["mll"] == "VariationalELBO"
    assert bundle.metadata["fit_func"] == "fit_non_gaussian_mll"
    assert torch.isfinite(bundle.model.posterior(train_x[:2]).mean).all()


def test_gamma_multitask_conditioning_and_state_round_trip() -> None:
    """Conditioning preserves the task structure and state restores predictions."""
    train_x, train_y = _data()
    kwargs = {"rank": 2, "num_latents": 2, "num_inducing_points": 8}
    model = WideGammaMultiTaskGPModel(train_x, train_y, **kwargs).eval()
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    expected = model.posterior(candidate).mean.detach()

    restored = WideGammaMultiTaskGPModel(train_x, train_y, **kwargs)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    torch.testing.assert_close(restored.posterior(candidate).mean, expected)

    conditioned = model.condition_on_observations(
        torch.tensor([[1.05]], dtype=torch.double),
        torch.tensor([[2.05, 3.1, 1.2]], dtype=torch.double),
    )
    assert isinstance(conditioned, WideGammaMultiTaskGPModel)
    assert conditioned.train_inputs_raw[0].shape[-2] == train_x.shape[-2] + 1
    torch.testing.assert_close(conditioned.task_covar_matrix, model.task_covar_matrix)
