from __future__ import annotations

import torch
from torch import nn

from bochan.fit.deep.deepkernel import fit_deepkernel_mll
from bochan.models.components.layers.kernel_layers import StableScaleToBounds
from bochan.models.regression.gaussian.deep.deepkernel import DeepKernelGPModel


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
    model = DeepKernelGPModel(
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


def test_deepkernel_stability_defaults_allow_custom_override() -> None:
    train_X = torch.rand(6, 2, dtype=torch.double)
    train_Y = train_X.sum(dim=-1, keepdim=True)
    model = DeepKernelGPModel(
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
