from __future__ import annotations

import torch
from gpytorch.kernels import MaternKernel, ScaleKernel

from bochan.models.ordinal.base import OrdinalGPModel


def test_public_ordinal_default_kernel_is_matern_2p5() -> None:
    torch.manual_seed(0)
    train_x = torch.rand(18, 2, dtype=torch.double)
    score = train_x[:, 0] + train_x[:, 1]
    q1, q2 = torch.quantile(score, torch.tensor([1 / 3, 2 / 3]))
    train_y = (score > q1).long() + (score > q2).long()
    train_y[:3] = torch.tensor([0, 1, 2])

    model = OrdinalGPModel(
        train_X=train_x,
        train_Y=train_y,
        num_classes=3,
    )
    kernel = model.model.covar_module

    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, MaternKernel)
    assert kernel.base_kernel.nu == 2.5
    assert kernel.base_kernel.ard_num_dims == train_x.shape[-1]
    assert model.model.variational_strategy.inducing_points.shape[-2] == train_x.shape[-2]
