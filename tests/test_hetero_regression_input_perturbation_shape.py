from __future__ import annotations

from types import SimpleNamespace

import torch

from bochan.acquisition.regression.bayesian_optimization.hetero_single_output import (
    compute_hetero_regression_best_f,
    hetero_adjust_regression_samples,
)


class _NoiseModel:
    def posterior(self, X: torch.Tensor):
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        mean = torch.zeros(
            *batch_shape,
            q,
            1,
            dtype=X.dtype,
            device=X.device,
        )
        return SimpleNamespace(mean=mean)


class _InputPerturbedHeteroModel:
    def __init__(self, n_w: int = 4) -> None:
        self.n_w = int(n_w)
        self.noise_model = _NoiseModel()

    def posterior(self, X: torch.Tensor):
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        q_perturbed = q * self.n_w
        view_shape = (1,) * len(batch_shape) + (q_perturbed, 1)
        mean = torch.arange(
            q_perturbed,
            dtype=X.dtype,
            device=X.device,
        ).reshape(view_shape)
        mean = mean.expand(*batch_shape, q_perturbed, 1)
        return SimpleNamespace(mean=mean)


def test_hetero_samples_align_noise_with_input_perturbation() -> None:
    model = _InputPerturbedHeteroModel(n_w=4)
    X = torch.zeros(32, 10, 5, dtype=torch.double)
    posterior = model.posterior(X)
    samples = posterior.mean.unsqueeze(0).expand(7, -1, -1, -1).clone()

    robust = hetero_adjust_regression_samples(
        model,
        X,
        samples,
        beta=0.0,
        noise_penalty=1.0,
        posterior=posterior,
    )

    assert robust.shape == torch.Size([7, 32, 40, 1])
    expected = posterior.mean.unsqueeze(0) - 1.0
    torch.testing.assert_close(robust, expected.expand_as(robust))


def test_hetero_best_f_aligns_training_noise_with_input_perturbation() -> None:
    model = _InputPerturbedHeteroModel(n_w=4)
    train_X = torch.zeros(10, 5, dtype=torch.double)

    best_f = compute_hetero_regression_best_f(
        model,
        train_X,
        noise_penalty=1.0,
    )

    torch.testing.assert_close(best_f, torch.tensor(38.0, dtype=torch.double))
