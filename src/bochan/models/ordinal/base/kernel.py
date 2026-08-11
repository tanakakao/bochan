from __future__ import annotations

from typing import Sequence

from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, RBFKernel, ScaleKernel


def _normalize_dims(cat_dims: Sequence[int], d: int) -> list[int]:
    """Normalize possibly-negative categorical feature indices."""
    dims: list[int] = []
    for idx in cat_dims:
        j = idx if idx >= 0 else d + idx
        if j < 0 or j >= d:
            raise ValueError(f"Invalid categorical dim {idx} for input dim {d}.")
        dims.append(int(j))
    return sorted(set(dims))


def _get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    """Return continuous feature indices complementary to ``cat_dims``."""
    cat_set = set(_normalize_dims(cat_dims, d))
    return [i for i in range(d) if i not in cat_set]


def _make_cont_kernel(
    cont_dims: Sequence[int],
    kernel_name: str = "matern52",
) -> Kernel | None:
    """Build the continuous part of the ordinal mixed-input kernel."""
    cont_dims = list(cont_dims)
    if not cont_dims:
        return None

    if kernel_name.lower() == "rbf":
        return ScaleKernel(
            RBFKernel(
                ard_num_dims=len(cont_dims),
                active_dims=tuple(cont_dims),
            )
        )
    if kernel_name.lower() == "matern52":
        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=len(cont_dims),
                active_dims=tuple(cont_dims),
            )
        )
    raise ValueError(f"Unknown continuous kernel: {kernel_name}")


def _make_cat_kernel(cat_dims: Sequence[int]) -> Kernel | None:
    """Build the categorical part of the ordinal mixed-input kernel."""
    cat_dims = list(cat_dims)
    if not cat_dims:
        return None
    return ScaleKernel(CategoricalKernel(active_dims=tuple(cat_dims)))


def build_mixed_ordinal_kernel(
    d: int,
    cat_dims: Sequence[int],
    cont_kernel_name: str = "matern52",
) -> Kernel:
    """Build the canonical mixed kernel for ordinal GP models."""
    cat_dims = _normalize_dims(cat_dims, d)
    cont_dims = _get_cont_dims(d, cat_dims)

    if not cat_dims:
        kernel = _make_cont_kernel(cont_dims, cont_kernel_name)
        if kernel is None:
            raise ValueError("Failed to build continuous kernel.")
        return kernel
    if not cont_dims:
        kernel = _make_cat_kernel(cat_dims)
        if kernel is None:
            raise ValueError("Failed to build categorical kernel.")
        return kernel

    cont_kernel_1 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cont_kernel_2 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cat_kernel_1 = _make_cat_kernel(cat_dims)
    cat_kernel_2 = _make_cat_kernel(cat_dims)
    if any(
        kernel is None
        for kernel in (
            cont_kernel_1,
            cont_kernel_2,
            cat_kernel_1,
            cat_kernel_2,
        )
    ):
        raise RuntimeError("Failed to build mixed ordinal kernel.")

    return cont_kernel_1 + cat_kernel_1 + ProductKernel(
        cont_kernel_2,
        cat_kernel_2,
    )


__all__ = ["build_mixed_ordinal_kernel"]
