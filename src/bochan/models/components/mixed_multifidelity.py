"""Kernel helpers for mixed long-format multi-fidelity Gaussian models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel

from bochan.models.components.mixed_kronecker import normalize_mixed_dims


def build_mixed_non_fidelity_kernel(
    *,
    d: int,
    cat_dims: Sequence[int],
    fidelity_dims: Sequence[int],
    batch_shape: torch.Size = torch.Size(),
) -> Kernel:
    """Build ``K_continuous * K_categorical`` excluding fidelity columns.

    The returned kernel is intended to be passed as ``covar_module`` to
    :class:`botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP`
    with ``linear_truncated=False``. BoTorch then multiplies this data kernel
    by its fidelity kernel, yielding the Phase-45 contract
    ``K_continuous * K_categorical * K_fidelity``.
    """

    d = int(d)
    categorical = tuple(normalize_mixed_dims(cat_dims, d))
    fidelity: list[int] = []
    seen: set[int] = set()
    for raw_index in fidelity_dims:
        index = int(raw_index)
        if index < 0:
            index += d
        if index < 0 or index >= d:
            raise ValueError(f"Invalid fidelity dim {raw_index} for input dim {d}.")
        if index in seen:
            raise ValueError(f"Duplicate fidelity dim {raw_index} resolves to feature {index}.")
        seen.add(index)
        fidelity.append(index)

    if set(categorical).intersection(fidelity):
        raise ValueError("Categorical and fidelity features must be disjoint.")

    excluded = set(categorical).union(fidelity)
    continuous = tuple(index for index in range(d) if index not in excluded)
    batch_shape = torch.Size(batch_shape)

    categorical_kernel = CategoricalKernel(
        ard_num_dims=len(categorical),
        active_dims=categorical,
        batch_shape=batch_shape,
        lengthscale_constraint=GreaterThan(1e-6),
    )

    if continuous:
        continuous_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=len(continuous),
            active_dims=continuous,
            batch_shape=batch_shape,
        )
        base_kernel: Kernel = ProductKernel(continuous_kernel, categorical_kernel)
    else:
        base_kernel = categorical_kernel

    return ScaleKernel(base_kernel, batch_shape=batch_shape)


__all__ = ["build_mixed_non_fidelity_kernel"]
