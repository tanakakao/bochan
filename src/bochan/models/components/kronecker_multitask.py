from __future__ import annotations

from typing import Optional

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor


def canonicalize_block_design_targets(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    target_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Validate block-design multi-task targets and return ``[n, m]`` data.

    Args:
        train_X: Shared input locations with shape ``[n, d]``.
        train_Y: Task observations with shape ``[n, m]``.
        target_dtype: Optional output dtype. Binary models generally use the
            input floating-point dtype, while ordinal models use ``torch.long``.

    Returns:
        Canonicalized target tensor on the same device as ``train_X``.
    """
    if train_X.ndim != 2:
        raise ValueError(f"train_X must have shape [n, d], got {tuple(train_X.shape)}.")
    if train_Y.ndim != 2:
        raise ValueError(
            "Kronecker multi-task models require block-design train_Y with "
            f"shape [n, m], got {tuple(train_Y.shape)}."
        )
    if train_Y.shape[0] != train_X.shape[0]:
        raise ValueError(
            "train_X and train_Y must contain the same number of input locations: "
            f"got {train_X.shape[0]} and {train_Y.shape[0]}."
        )
    if train_Y.shape[1] < 2:
        raise ValueError(
            "Kronecker multi-task models require at least two tasks; "
            f"got train_Y.shape[-1]={train_Y.shape[1]}."
        )

    dtype = train_Y.dtype if target_dtype is None else target_dtype
    return train_Y.to(device=train_X.device, dtype=dtype).contiguous()


def canonicalize_shared_inducing_points(
    train_X: Tensor,
    *,
    num_inducing_points: int,
    inducing_points: Optional[Tensor] = None,
) -> Tensor:
    """Return shared inducing locations for all latent functions.

    This helper returns the raw shared locations ``[p, d]``. The latent model
    repeats them over the latent-function dimension so every latent process is
    initialized at the same locations.
    """
    if inducing_points is not None:
        inducing_points = torch.as_tensor(
            inducing_points,
            device=train_X.device,
            dtype=train_X.dtype,
        )
        if inducing_points.ndim != 2:
            raise ValueError(
                "inducing_points must have shape [p, d], "
                f"got {tuple(inducing_points.shape)}."
            )
        if inducing_points.shape[-1] != train_X.shape[-1]:
            raise ValueError(
                "inducing_points feature dimension must match train_X: "
                f"expected {train_X.shape[-1]}, got {inducing_points.shape[-1]}."
            )
        return inducing_points.contiguous()

    p = min(int(num_inducing_points), train_X.shape[-2])
    if p < 1:
        raise ValueError("num_inducing_points must be >= 1.")

    # Deterministic coverage of the stored training order avoids constructor-level
    # randomness while retaining points from the full input range.
    indices = torch.linspace(
        0,
        train_X.shape[-2] - 1,
        p,
        device=train_X.device,
    ).round().long()
    return train_X.index_select(-2, indices).detach().clone().contiguous()


class BlockDesignVariationalELBO(VariationalELBO):
    """Variational ELBO that sums all block-design task observations.

    GPyTorch's generic :class:`VariationalELBO` sums only the last dimension of
    ``expected_log_prob``. For a multitask event shaped ``[n, m]``, that can
    leave an ``[n]`` tensor and repeat the KL term when callers subsequently
    reduce the loss. This specialization reduces the full likelihood term to a
    scalar before the standard minibatch and KL scaling are applied.
    """

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        return self.likelihood.expected_log_prob(
            target,
            variational_dist_f,
            **kwargs,
        ).sum()


class LatentKroneckerMultiTaskGP(ApproximateGP):
    r"""Variational latent GP with an ICM/Kronecker multi-task prior.

    The model uses ``rank`` independent latent GPs that share one data kernel
    ``K_X``. ``LMCVariationalStrategy`` mixes them into ``num_tasks`` outputs:

    .. math::

        f_t(x) = \sum_{q=1}^{r} a_{qt} g_q(x).

    Since every ``g_q`` shares ``K_X``, the prior covariance is

    .. math::

        K((x, t), (x', t')) = K_X(x, x') B_{tt'},
        \qquad B = A^\top A,

    i.e. ``K_X ⊗ B``. Non-Gaussian binary or ordinal likelihoods are attached by
    the public wrappers in their respective model families.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        rank: Optional[int] = None,
        num_inducing_points: int = 128,
        inducing_points: Optional[Tensor] = None,
        learn_inducing_locations: bool = True,
        mean_module: Optional[Mean] = None,
        data_covar_module: Optional[Kernel] = None,
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError(f"train_X must have shape [n, d], got {tuple(train_X.shape)}.")
        if train_Y.ndim != 2:
            raise ValueError(f"train_Y must have shape [n, m], got {tuple(train_Y.shape)}.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must have the same first dimension.")

        num_tasks = int(train_Y.shape[-1])
        rank = num_tasks if rank is None else int(rank)
        if rank < 1 or rank > num_tasks:
            raise ValueError(
                f"rank must satisfy 1 <= rank <= num_tasks ({num_tasks}), got {rank}."
            )

        shared_inducing_points = canonicalize_shared_inducing_points(
            train_X,
            num_inducing_points=num_inducing_points,
            inducing_points=inducing_points,
        )
        latent_inducing_points = shared_inducing_points.unsqueeze(0).expand(
            rank,
            *shared_inducing_points.shape,
        ).clone()

        latent_batch_shape = torch.Size([rank])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=shared_inducing_points.shape[-2],
            batch_shape=latent_batch_shape,
        )
        base_variational_strategy = VariationalStrategy(
            model=self,
            inducing_points=latent_inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=rank,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module or ConstantMean()
        self.data_covar_module = data_covar_module or ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
        )
        self.covar_module = self.data_covar_module

        self.num_tasks = num_tasks
        self.rank = rank
        self.num_inducing_points = int(shared_inducing_points.shape[-2])
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self.shared_inducing_points = shared_inducing_points

        self.to(device=train_X.device, dtype=train_X.dtype)

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(X)
        covar_x = self.data_covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    @property
    def lmc_coefficients(self) -> Tensor:
        """Return the latent-to-task mixing matrix with shape ``[rank, m]``."""
        return self.variational_strategy.lmc_coefficients

    @property
    def task_covar_matrix(self) -> Tensor:
        """Return the positive-semidefinite ICM task covariance ``A.T @ A``."""
        coefficients = self.lmc_coefficients
        return coefficients.transpose(-1, -2) @ coefficients

    def get_shared_inducing_points(self) -> Tensor:
        """Return one ``[p, d]`` copy of the current inducing locations."""
        points = self.variational_strategy.base_variational_strategy.inducing_points
        return points[0]
