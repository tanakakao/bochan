"""Internal helpers for wide multi-fidelity binary classifiers."""

from __future__ import annotations

import torch
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.transforms.input import InputTransform
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel
from torch import Tensor

from bochan.models.regression.gaussian.multifidelity import (
    FidelityFeatureInputTransform,
    wide_fidelity_to_long,
)


def prepare_fidelity_input_transform(
    transform: InputTransform | None,
    data_dim: int,
) -> InputTransform | None:
    """Wrap a public-space transform while preserving appended fidelity."""
    if transform is None or isinstance(transform, FidelityFeatureInputTransform):
        return transform
    return FidelityFeatureInputTransform(transform, data_dim=data_dim)


def normalize_fidelity_values(
    values,
    X: Tensor,
    num_fidelities: int,
) -> Tensor:
    """Return finite, unique fidelity values on the input dtype and device."""
    result = torch.as_tensor(values, dtype=X.dtype, device=X.device).reshape(-1)
    if result.numel() != int(num_fidelities):
        raise ValueError(
            f"Expected {num_fidelities} fidelity_values, got {result.numel()}."
        )
    if not torch.isfinite(result).all() or torch.unique(result).numel() != result.numel():
        raise ValueError("fidelity_values must be finite and unique.")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError("fidelity_values must be within [0, 1].")
    return result


def validate_binary_targets(train_Y: Tensor) -> None:
    """Validate observed wide targets as binary labels."""
    observed = train_Y[~torch.isnan(train_Y)]
    if observed.numel() == 0:
        raise ValueError("train_Y must contain at least one observed binary label.")
    valid = (observed == 0) | (observed == 1)
    if not bool(valid.all()):
        invalid = torch.unique(observed[~valid]).detach().cpu().tolist()
        raise ValueError(
            "Observed binary targets must be encoded as 0 or 1. "
            f"Invalid values: {invalid}."
        )


def make_default_data_kernel(data_dim: int, ref_X: Tensor) -> Kernel:
    """Build the non-fidelity covariance for continuous design variables."""
    return MaternKernel(
        nu=2.5,
        ard_num_dims=int(data_dim),
        active_dims=tuple(range(int(data_dim))),
        batch_shape=torch.Size(),
    ).to(device=ref_X.device, dtype=ref_X.dtype)


def make_multifidelity_kernel(
    *,
    data_dim: int,
    ref_X: Tensor,
    data_covar_module: Kernel,
    fidelity_covar_module: Kernel | None,
) -> Kernel:
    """Combine design and ordered-fidelity covariance modules."""
    fidelity_kernel = fidelity_covar_module or DownsamplingKernel(
        active_dims=(int(data_dim),),
        batch_shape=torch.Size(),
    )
    fidelity_kernel = fidelity_kernel.to(device=ref_X.device, dtype=ref_X.dtype)
    data_kernel = data_covar_module.to(device=ref_X.device, dtype=ref_X.dtype)
    return ScaleKernel(
        ProductKernel(data_kernel, fidelity_kernel),
        batch_shape=torch.Size(),
    ).to(device=ref_X.device, dtype=ref_X.dtype)


def wide_probability_tensors(
    posterior,
    *,
    public_q: int,
    num_fidelities: int,
) -> tuple[Tensor, Tensor]:
    """Reshape flattened fidelity/perturbation values to ``[..., q*n_w, m]``."""
    mean, variance = posterior.mean, posterior.variance
    if mean.shape != variance.shape or mean.shape[-1] != 1:
        raise RuntimeError(
            "Binary fidelity posterior must expose matching mean/variance with "
            f"a singleton output dimension, got {tuple(mean.shape)} and "
            f"{tuple(variance.shape)}."
        )
    flat_points = int(mean.shape[-2])
    expected = int(public_q) * int(num_fidelities)
    if expected <= 0 or flat_points % expected != 0:
        raise RuntimeError(
            "Wide binary fidelity posterior point count must equal "
            "q * num_fidelities * n_w."
        )
    n_w = flat_points // expected

    def reshape(value: Tensor) -> Tensor:
        squeezed = value.squeeze(-1)
        grid = squeezed.reshape(
            *squeezed.shape[:-1], public_q, num_fidelities, n_w
        )
        return grid.transpose(-1, -2).reshape(
            *squeezed.shape[:-1], public_q * n_w, num_fidelities
        )

    return reshape(mean), reshape(variance)


__all__ = [
    "FidelityFeatureInputTransform",
    "make_default_data_kernel",
    "make_multifidelity_kernel",
    "normalize_fidelity_values",
    "prepare_fidelity_input_transform",
    "validate_binary_targets",
    "wide_fidelity_to_long",
    "wide_probability_tensors",
]
