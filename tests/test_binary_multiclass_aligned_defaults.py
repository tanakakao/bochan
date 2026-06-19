from __future__ import annotations

import inspect

import torch
from gpytorch.kernels import AdditiveKernel, Kernel, MaternKernel, ProductKernel, ScaleKernel
from botorch.models.kernels.categorical import CategoricalKernel

from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    BinaryClassificationMixedGPModel,
)


def _make_binary_data(n: int = 150, d: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_X = torch.rand(n, d, dtype=torch.double)
    score = train_X[:, 0] - 0.5 * train_X[:, 1] + 0.25 * train_X[:, 2]
    train_Y = (score > score.median()).to(dtype=train_X.dtype).unsqueeze(-1)
    return train_X, train_Y


def _walk_kernels(kernel: Kernel):
    yield kernel
    for child in kernel.children():
        if isinstance(child, Kernel):
            yield from _walk_kernels(child)


def test_binary_public_default_num_inducing_points_matches_multiclass() -> None:
    signature = inspect.signature(BinaryClassificationGPModel.__init__)
    assert signature.parameters["num_inducing_points"].default == 128

    train_X, train_Y = _make_binary_data()
    model = BinaryClassificationGPModel(train_X=train_X, train_Y=train_Y)

    inducing_points = model.model.variational_strategy.inducing_points
    assert inducing_points.shape[-2] == 128


def test_binary_public_default_kernel_is_matern_2p5() -> None:
    train_X, train_Y = _make_binary_data()
    model = BinaryClassificationGPModel(train_X=train_X, train_Y=train_Y)

    covar_module = model.model.covar_module
    assert isinstance(covar_module, ScaleKernel)
    assert isinstance(covar_module.base_kernel, MaternKernel)
    assert covar_module.base_kernel.nu == 2.5
    assert covar_module.base_kernel.ard_num_dims == train_X.shape[-1]


def test_binary_mixed_defaults_match_multiclass_structure() -> None:
    signature = inspect.signature(BinaryClassificationMixedGPModel.__init__)
    assert signature.parameters["num_inducing_points"].default == 128

    train_cont, train_Y = _make_binary_data()
    category = torch.randint(0, 3, (train_cont.shape[0], 1)).to(train_cont)
    train_X = torch.cat([train_cont, category], dim=-1)

    model = BinaryClassificationMixedGPModel(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[train_X.shape[-1] - 1],
    )

    inducing_points = model.model.variational_strategy.inducing_points
    assert inducing_points.shape[-2] == 128

    covar_module = model.model.covar_module
    assert isinstance(covar_module, ScaleKernel)
    assert isinstance(covar_module.base_kernel, AdditiveKernel)

    kernels = list(_walk_kernels(covar_module.base_kernel))
    assert sum(isinstance(kernel, MaternKernel) for kernel in kernels) == 2
    assert sum(isinstance(kernel, CategoricalKernel) for kernel in kernels) == 2
    assert any(isinstance(kernel, ProductKernel) for kernel in kernels)

    matern_kernels = [kernel for kernel in kernels if isinstance(kernel, MaternKernel)]
    assert all(kernel.nu == 2.5 for kernel in matern_kernels)
