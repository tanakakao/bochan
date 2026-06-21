from __future__ import annotations

import torch
from botorch.models.transforms.input import Normalize

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig
from bochan.fit import VAEFitResult, fit_vae_gp
from bochan.models.regression.gaussian.high_dim import VAESingleTaskGP

DTYPE = torch.double


def make_regression_data(
    n: int = 16,
    d: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_X = torch.rand(n, d, dtype=DTYPE)
    train_Y = (
        torch.sin(2.0 * torch.pi * train_X[:, :1])
        + 0.25 * train_X[:, 1:2]
        - 0.1 * train_X[:, 2:3].square()
    )
    bounds = torch.stack(
        [torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)]
    )
    return train_X, train_Y, bounds


def test_vae_single_task_gp_joint_fit_and_posterior() -> None:
    train_X, train_Y, bounds = make_regression_data()
    model = VAESingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
        latent_dim=2,
        hidden_dims=[10, 6],
        reconstruction_weight=1.0,
        kl_weight=1e-3,
        gp_weight=1.0,
    )

    encoder_before = [
        parameter.detach().clone() for parameter in model.vae.parameters()
    ]
    result = fit_vae_gp(
        model,
        num_epochs=3,
        lr=0.01,
        clip_grad_norm=10.0,
    )

    assert isinstance(result, VAEFitResult)
    assert len(result.loss_history) == 3
    assert torch.isfinite(torch.tensor(result.loss_history)).all()
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(encoder_before, model.vae.parameters())
    )

    latent = model.encode(train_X)
    reconstruction = model.reconstruct(train_X, raw_space=True)
    posterior = model.posterior(train_X[:4])

    assert latent.shape == torch.Size([train_X.shape[0], 2])
    assert reconstruction.shape == train_X.shape
    assert model.latent_train_input.shape == latent.shape
    assert model.base_model.train_inputs[0].shape == latent.shape
    assert posterior.mean.shape == torch.Size([4, 1])
    assert posterior.variance.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_high_level_api_resolves_and_fits_vae_model() -> None:
    train_X, train_Y, bounds = make_regression_data(n=12, d=4)
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(
            task_type="regression",
            model_type="vae",
            input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
            model_kwargs={
                "latent_dim": 2,
                "hidden_dims": [8],
                "kl_weight": 1e-3,
            },
        ),
        fit_config=FitConfig(num_epochs=2, lr=0.01),
        bounds=bounds,
    )

    optimizer.fit(train_X, train_Y)

    assert isinstance(optimizer.model, VAESingleTaskGP)
    assert isinstance(optimizer.bundle.fit_result, VAEFitResult)
    assert optimizer.bundle.mll is None
    assert optimizer.bundle.metadata["fit_func"] == "fit"
    posterior = optimizer.predict(train_X[:3])
    assert posterior.mean.shape == torch.Size([3, 1])
