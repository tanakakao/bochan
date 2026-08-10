from __future__ import annotations

import torch
from gpytorch.settings import cholesky_jitter
from linear_operator.utils.errors import NotPSDError
from torch import nn

from bochan.fit.deep.common import fit_deep_full_batch_mll
from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.likelihoods.regression import build_single_task_likelihood
from bochan.models.components.layers.kernel_layers import StableScaleToBounds
from bochan.models.regression.gaussian.deep.deepkernel import DeepKernelGaussianGPModel


class ConstantFeatureExtractor(nn.Module):
    """Return a collapsed representation to reproduce a degenerate fold."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            *X.shape[:-1],
            self.output_dim,
            dtype=X.dtype,
            device=X.device,
        )


class LinearTrainingModel(nn.Module):
    """Small trainable model used to exercise the generic fitting loop."""

    def __init__(self) -> None:
        super().__init__()
        train_X = torch.arange(1, 5, dtype=torch.double).unsqueeze(-1)
        self.train_inputs = (train_X,)
        self.train_targets = 2.0 * train_X.squeeze(-1)
        self.weight = nn.Parameter(torch.tensor(0.5, dtype=torch.double))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.squeeze(-1) * self.weight


class JitterSensitiveMll(nn.Module):
    """Fail until the configured Cholesky jitter reaches a threshold."""

    def __init__(self, model: nn.Module, threshold: float) -> None:
        super().__init__()
        self.model = model
        self.threshold = float(threshold)
        self.seen_jitters: list[float] = []

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        jitter = float(cholesky_jitter.value(dtype=target.dtype))
        self.seen_jitters.append(jitter)
        if jitter < self.threshold:
            raise NotPSDError(f"jitter {jitter} is below {self.threshold}")
        return -torch.mean((output - target).square())


def test_stable_scale_to_bounds_keeps_constant_features_finite() -> None:
    scaler = StableScaleToBounds(-1.0, 1.0).double()
    scaler.train()

    scaled = scaler(torch.ones(8, 3, dtype=torch.double))

    assert torch.isfinite(scaled).all()
    assert scaled.shape == (8, 3)
    assert scaled.min() >= -1.0
    assert scaled.max() <= 1.0


def test_deepkernel_fit_handles_collapsed_fold_representation() -> None:
    train_X = torch.zeros(8, 3, dtype=torch.double)
    train_Y = torch.linspace(-1.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
    )
    model.deepkernel.feature_extractor = ConstantFeatureExtractor(output_dim=3)

    mll = model.make_mll()
    fitted_mll = fit_deepkernel_mll(mll, num_epochs=3)

    assert fitted_mll is mll
    model.eval()
    posterior = model.posterior(train_X[:3])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_deepkernel_fit_keeps_log_normal_noise_prior_in_support() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 3.0)
    alpha = 1e-6
    likelihood = build_single_task_likelihood(
        train_X=train_X,
        train_Y=train_Y,
        alpha=alpha,
    ).to(train_X)

    with torch.no_grad():
        likelihood.noise_covar.raw_noise.fill_(-20.0)

    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        input_transform=None,
        outcome_transform=None,
    )
    mll = model.make_mll()

    fitted_mll = fit_deepkernel_mll(mll, num_epochs=2)

    assert fitted_mll is mll
    assert torch.all(likelihood.noise > alpha)
    prior_log_prob = likelihood.noise_covar.noise_prior.log_prob(likelihood.noise)
    assert torch.isfinite(prior_log_prob).all()


def test_deepkernel_stability_defaults_allow_custom_override() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = train_X.sum(dim=-1, keepdim=True)
    model = DeepKernelGaussianGPModel(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=None,
        outcome_transform=None,
    )

    mll = model.make_mll()
    fitted_mll = fit_deepkernel_mll(
        mll,
        num_epochs=1,
        clip_grad_norm=None,
        psd_jitter_values=(1e-7, 1e-5),
    )

    assert fitted_mll is mll


def test_full_batch_fit_retries_not_psd_with_ascending_bounded_jitter() -> None:
    model = LinearTrainingModel()
    mll = JitterSensitiveMll(model, threshold=1e-3)
    initial_weight = model.weight.detach().clone()

    fitted_mll = fit_deep_full_batch_mll(
        mll,
        num_epochs=1,
        psd_jitter_values=(1e-6, 1e-4, 1e-2),
    )

    assert fitted_mll is mll
    assert mll.seen_jitters == [1e-6, 1e-4, 1e-2]
    assert not torch.equal(model.weight.detach(), initial_weight)
