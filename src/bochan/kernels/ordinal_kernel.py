from typing import Optional, Sequence
from gpytorch.kernels import (
    MaternKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from botorch.models.kernels.categorical import CategoricalKernel

def _normalize_dims(cat_dims: Sequence[int], d: int) -> list[int]:
    dims: list[int] = []
    for idx in cat_dims:
        j = idx if idx >= 0 else d + idx
        if j < 0 or j >= d:
            raise ValueError(f"Invalid categorical dim {idx} for input dim {d}.")
        dims.append(int(j))
    return sorted(set(dims))


def _get_cont_dims(d: int, cat_dims: Sequence[int]) -> list[int]:
    cat_set = set(_normalize_dims(cat_dims, d))
    return [i for i in range(d) if i not in cat_set]


def _make_cont_kernel(cont_dims: Sequence[int], kernel_name: str = "matern52"):
    cont_dims = list(cont_dims)
    if len(cont_dims) == 0:
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


def _make_cat_kernel(cat_dims: Sequence[int]):
    cat_dims = list(cat_dims)
    if len(cat_dims) == 0:
        return None
    return ScaleKernel(CategoricalKernel(active_dims=tuple(cat_dims)))


def build_mixed_ordinal_kernel(
    d: int,
    cat_dims: Sequence[int],
    cont_kernel_name: str = "matern52",
):
    """
    Build a mixed kernel in the spirit of ``MixedSingleTaskGP``.

    K = K_cont_1 + K_cat_1 + K_cont_2 * K_cat_2
    """
    cat_dims = _normalize_dims(cat_dims, d)
    cont_dims = _get_cont_dims(d, cat_dims)

    if len(cat_dims) == 0:
        return _make_cont_kernel(cont_dims, cont_kernel_name)
    if len(cont_dims) == 0:
        return _make_cat_kernel(cat_dims)

    cont_kernel_1 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cont_kernel_2 = _make_cont_kernel(cont_dims, cont_kernel_name)
    cat_kernel_1 = _make_cat_kernel(cat_dims)
    cat_kernel_2 = _make_cat_kernel(cat_dims)
    return cont_kernel_1 + cat_kernel_1 + ProductKernel(cont_kernel_2, cat_kernel_2)