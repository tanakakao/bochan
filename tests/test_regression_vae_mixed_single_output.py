from __future__ import annotations

import torch
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf_mixed

from bochan.api import BayesianOptimizer, FitConfig, ModelConfig
from bochan.fit import VAEFitResult, fit_vae_gp
from bochan.models.regression.gaussian.high_dim import VAEGaussianMixedGPModel

DTYPE = torch.double
CAT_DIMS = [1]
CONT_DIMS = [0, 2, 3, 4]


def make_mixed_regression_data(
    n: int = 18,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    continuous_X = torch.rand(n, len(CONT_DIMS), dtype=DTYPE)
    category = torch.randint(0, 3, (n, 1)).to(dtype=DTYPE)
    train_X = torch.cat(
        [continuous_X[:, :1], category, continuous_X[:, 1:]],
        dim=-1,
    )
    train_Y = (
        torch.sin(2.0 * torch.pi * continuous_X[:, :1])
        + 0.3 * continuous_X[:, 1:2]
        + 0.2 * category
    )
    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 1.0, 1.0],
        ],
        dtype=DTYPE,
    )
    return train_X, train_Y, bounds


def make_continuous_only_normalize(bounds: torch.Tensor) -> Normalize:
    return Normalize(
        d=bounds.shape[-1],
        bounds=bounds,
        indices=CONT_DIMS,
    )


def test_vae_mixed_gp_projects_only_continuous_columns() -> None:
    train_X, train_Y, bounds = make_mixed_regression_data()
    model = VAEGaussianMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=CAT_DIMS,
        category_counts={1: 3},
        input_transform=make_continuous_only_normalize(bounds),
        latent_dim=2,
        hidden_dims=[10, 6],
        reconstruction_weight=1.0,
        kl_weight=1e-3,
        gp_weight=1.0,
    )

    result = fit_vae_gp(
        model,
        num_epochs=3,
        lr=0.01,
        clip_grad_norm=10.0,
    )

    assert isinstance(result, VAEFitResult)
    assert len(result.loss_history) == 3
    assert torch.isfinite(torch.tensor(result.loss_history)).all()

    latent = model.encode(train_X)
    projected = model.transform_inputs(train_X)
    reconstruction = model.reconstruct(train_X, raw_space=True)
    decoded = model.decode(
        latent,
        categorical_X=train_X[:, CAT_DIMS],
        raw_space=True,
    )
    posterior = model.posterior(train_X[:4])

    assert latent.shape == torch.Size([train_X.shape[0], 2])
    assert projected.shape == torch.Size([train_X.shape[0], 3])
    assert model.latent_train_input.shape == latent.shape
    assert model.projected_train_input.shape == projected.shape
    assert model.base_model.train_inputs[0].shape == projected.shape
    assert model.latent_cat_dims == [2]
    assert torch.equal(projected[:, -1], train_X[:, 1])
    assert reconstruction.shape == train_X.shape
    assert decoded.shape == train_X.shape
    assert torch.equal(reconstruction[:, 1], train_X[:, 1])
    assert torch.equal(decoded[:, 1], train_X[:, 1])
    assert posterior.mean.shape == torch.Size([4, 1])
    assert posterior.variance.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    acquisition = UpperConfidenceBound(model=model, beta=0.1)
    candidates, acquisition_value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features_list=[{1: 0.0}, {1: 1.0}, {1: 2.0}],
        options={"maxiter": 20},
    )
    assert candidates.shape == torch.Size([1, train_X.shape[-1]])
    assert candidates[0, 1].item() in {0.0, 1.0, 2.0}
    assert torch.isfinite(acquisition_value).all()


def test_high_level_api_resolves_and_fits_mixed_vae_model() -> None:
    train_X, train_Y, bounds = make_mixed_regression_data(n=12)
    optimizer = BayesianOptimizer(
        model_config=ModelConfig(
            task_type="regression",
            model_type="vae",
            cat_dims=CAT_DIMS,
            input_transform=make_continuous_only_normalize(bounds),
            model_kwargs={
                "category_counts": {1: 3},
                "latent_dim": 2,
                "hidden_dims": [8],
                "kl_weight": 1e-3,
            },
        ),
        fit_config=FitConfig(num_epochs=2, lr=0.01),
        bounds=bounds,
    )

    optimizer.fit(train_X, train_Y)

    assert isinstance(optimizer.model, VAEGaussianMixedGPModel)
    assert isinstance(optimizer.bundle.fit_result, VAEFitResult)
    assert optimizer.bundle.input_type == "mixed"
    assert optimizer.bundle.mll is None
    assert optimizer.bundle.metadata["fit_func"] == "fit"
    posterior = optimizer.predict(train_X[:3])
    assert posterior.mean.shape == torch.Size([3, 1])
