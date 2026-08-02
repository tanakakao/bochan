from __future__ import annotations

import torch
from torch import nn

import bochan.models.regression.non_gaussian.gamma.robust.gamma_heteroscedastic as gamma_heteroscedastic
from bochan.models.regression.non_gaussian.gamma.robust.gamma_heteroscedastic import (
    HeteroscedasticGammaGPModel,
    HeteroscedasticGammaMixedGPModel,
)


class _StubNoiseModel(nn.Module):
    """Minimal registered module returned instead of fitting an auxiliary GP."""

    def __init__(self, reference: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(reference.new_ones(1))


def _continuous_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.tensor(
        [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
        ],
        dtype=torch.double,
    )
    train_y = 0.5 + train_x[:, :1].square() + train_x[:, 1:2]
    train_yvar = torch.full_like(train_y, 0.05)
    return train_x, train_y, train_yvar


def _mixed_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous_x, train_y, train_yvar = _continuous_data()
    categorical_x = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=continuous_x.dtype)
    train_x = torch.cat([continuous_x, categorical_x], dim=-1)
    train_y = train_y + 0.1 * categorical_x
    return train_x, train_y, train_yvar


def test_heteroscedastic_gamma_registers_noise_model_after_parent_init(monkeypatch) -> None:
    train_x, train_y, train_yvar = _continuous_data()
    noise_model = _StubNoiseModel(train_x)
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_fit_noise_model_single",
        lambda **kwargs: noise_model,
    )

    model = HeteroscedasticGammaGPModel(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        num_inducing_points=3,
    )

    assert model.noise_model is noise_model
    assert model._modules["noise_model"] is noise_model
    assert "noise_model.weight" in model.state_dict()


def test_heteroscedastic_gamma_mixed_registers_noise_model_after_parent_init(monkeypatch) -> None:
    train_x, train_y, train_yvar = _mixed_data()
    noise_model = _StubNoiseModel(train_x)
    monkeypatch.setattr(
        gamma_heteroscedastic,
        "_fit_noise_model_mixed",
        lambda **kwargs: noise_model,
    )

    model = HeteroscedasticGammaMixedGPModel(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        cat_dims=[2],
        num_inducing_points=3,
    )

    assert model.noise_model is noise_model
    assert model._modules["noise_model"] is noise_model
    assert "noise_model.weight" in model.state_dict()
