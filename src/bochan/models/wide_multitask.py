"""Wide-format adapters for task-feature multi-task models.

The public data contract is ``train_X=[n, d]`` and ``train_Y=[n, m]`` with
optional NaNs for unobserved task values. Internally the adapters convert to the
long task-feature representation used by the existing multi-task models.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import GetSampler, get_sampler
from torch import Tensor

from bochan.models.classification.binary.base.multitask import (
    MultiTaskBinaryClassificationGPModel,
)
from bochan.models.classification.multiclass.base.multitask import (
    MultiTaskMulticlassClassificationGPModel,
)
from bochan.models.ordinal.base.multitask import MultiTaskOrdinalGPModel


def wide_to_long(
    train_X: Tensor,
    train_Y: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Convert wide targets to long task-feature observations.

    NaN target cells are interpreted as unobserved task values and omitted.
    Every task must contain at least one finite observation.
    """

    train_X = torch.as_tensor(train_X)
    train_Y = torch.as_tensor(train_Y, device=train_X.device)
    if train_X.ndim != 2:
        raise ValueError(f"train_X must have shape [n, d], got {tuple(train_X.shape)}.")
    if train_Y.ndim != 2:
        raise ValueError(f"train_Y must have shape [n, m], got {tuple(train_Y.shape)}.")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
    if train_Y.shape[1] < 2:
        raise ValueError("Wide multi-task models require at least two task columns.")
    if torch.isinf(train_Y).any():
        raise ValueError("train_Y may contain NaN for missing observations, but not inf.")

    observed = ~torch.isnan(train_Y)
    missing_tasks = torch.where(~observed.any(dim=0))[0]
    if missing_tasks.numel():
        raise ValueError(
            "Each task must contain at least one observed value. Missing task ids: "
            f"{missing_tasks.detach().cpu().tolist()}."
        )

    row_idx, task_idx = observed.nonzero(as_tuple=True)
    X_obs = train_X[row_idx]
    task_column = task_idx.to(dtype=train_X.dtype).unsqueeze(-1)
    X_long = torch.cat([X_obs, task_column], dim=-1)
    Y_long = train_Y[row_idx, task_idx].to(dtype=train_X.dtype).unsqueeze(-1)
    return X_long, Y_long, int(train_Y.shape[1])


def _expand_tasks(X: Tensor, num_tasks: int) -> Tensor:
    """Append every task id and flatten the point-task grid."""

    X = torch.as_tensor(X)
    q = X.shape[-2]
    expanded = X.unsqueeze(-2).expand(*X.shape[:-2], q, num_tasks, X.shape[-1])
    task_shape = (*X.shape[:-2], q, num_tasks, 1)
    task_ids = torch.arange(num_tasks, device=X.device, dtype=X.dtype)
    task_ids = task_ids.view(*([1] * (X.ndim - 2)), 1, num_tasks, 1).expand(task_shape)
    return torch.cat([expanded, task_ids], dim=-1).reshape(
        *X.shape[:-2], q * num_tasks, X.shape[-1] + 1
    )


def _reshape_wide(value: Tensor, q: int, num_tasks: int) -> Tensor:
    """Reshape flattened point-task posterior values to wide output form."""

    if value.shape[-1] == 1:
        value = value.squeeze(-1)
        return value.reshape(*value.shape[:-1], q, num_tasks)
    return value.reshape(*value.shape[:-2], q, num_tasks, value.shape[-1])


class _WidePosterior(Posterior):
    """Posterior view that preserves base-sample support.

    The wrapped posterior keeps its flattened ``q * m`` base-sample shape. Only
    user-visible moments and samples are reshaped to ``q x m``. Class-valued
    posteriors additionally preserve their final class axis.
    """

    def __init__(
        self,
        posterior: Posterior,
        *,
        q: int,
        num_tasks: int,
        output_indices: list[int],
        input_ndim: int,
    ) -> None:
        self.posterior = posterior
        self.q = int(q)
        self.num_tasks = int(num_tasks)
        self.output_indices = list(output_indices)
        self.input_ndim = int(input_ndim)
        self.scalar_task_values = bool(posterior.mean.shape[-1] == 1)

    def _transform(self, value: Tensor) -> Tensor:
        wide = _reshape_wide(value, q=self.q, num_tasks=self.num_tasks)
        task_dim = -1 if self.scalar_task_values else -2
        index = torch.tensor(self.output_indices, device=wide.device)
        return wide.index_select(task_dim, index)

    @property
    def mean(self) -> Tensor:
        return self._transform(self.posterior.mean)

    @property
    def variance(self) -> Tensor:
        return self._transform(self.posterior.variance)

    @property
    def device(self) -> torch.device:
        return self.posterior.device

    @property
    def dtype(self) -> torch.dtype:
        return self.posterior.dtype

    @property
    def base_sample_shape(self) -> torch.Size:
        return self.posterior.base_sample_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        return self.posterior.batch_range

    @property
    def batch_shape(self) -> torch.Size:
        mean = self.mean
        trailing = 2 if self.scalar_task_values else 3
        if mean.ndim <= trailing:
            return torch.Size()
        return torch.Size(mean.shape[:-trailing])

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        return self._transform(self.posterior.rsample(sample_shape=sample_shape))

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        samples = self.posterior.rsample_from_base_samples(
            sample_shape=sample_shape,
            base_samples=base_samples,
        )
        return self._transform(samples)

    def _extended_shape(
        self,
        sample_shape: torch.Size | None = None,
    ) -> torch.Size:
        mean_shape = self.mean.shape
        if not self.scalar_task_values:
            # Multiclass probability objectives reduce the final class axis and
            # expose ``[..., q, m]`` to BoTorch's multi-objective machinery.
            mean_shape = mean_shape[:-1]
        sample_shape = (
            torch.Size() if sample_shape is None else torch.Size(sample_shape)
        )
        return sample_shape + torch.Size(mean_shape)


@GetSampler.register(_WidePosterior)
def _get_wide_posterior_sampler(
    posterior: _WidePosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> MCSampler:
    """Reuse the wrapped posterior's registered sampler for wide samples."""

    return get_sampler(
        posterior=posterior.posterior,
        sample_shape=sample_shape,
        seed=seed,
    )


class _WidePosteriorMixin:
    """Expose a task-feature model as a normal wide multi-output model."""

    num_tasks: int

    def _selected_outputs(self, output_indices: list[int] | None) -> list[int]:
        selected = (
            list(range(self.num_tasks))
            if output_indices is None
            else [int(index) for index in output_indices]
        )
        if not selected:
            raise ValueError("output_indices must contain at least one task index.")
        if min(selected) < 0 or max(selected) >= self.num_tasks:
            raise ValueError(f"output_indices must be in [0, {self.num_tasks - 1}].")
        return selected

    def _wrap_wide_posterior(
        self,
        base: Posterior,
        *,
        X: Tensor,
        selected: list[int],
        posterior_transform: Any = None,
    ) -> Posterior:
        posterior = _WidePosterior(
            base,
            q=int(X.shape[-2]),
            num_tasks=self.num_tasks,
            output_indices=selected,
            input_ndim=X.ndim,
        )
        return posterior_transform(posterior) if posterior_transform is not None else posterior

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior:
        X = torch.as_tensor(X)
        selected = self._selected_outputs(output_indices)
        X_all = _expand_tasks(X, self.num_tasks)
        base = super().posterior(
            X_all,
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        return self._wrap_wide_posterior(
            base,
            X=X,
            selected=selected,
            posterior_transform=posterior_transform,
        )

    def _wide_latent_posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior:
        """Expand raw candidates before calling a long-format latent accessor."""

        X = torch.as_tensor(X)
        selected = self._selected_outputs(output_indices)
        X_all = _expand_tasks(X, self.num_tasks)
        accessor = super().latent_posterior

        try:
            parameters = inspect.signature(accessor).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        call_kwargs = dict(kwargs)
        if accepts_kwargs or "posterior_transform" in parameters:
            call_kwargs["posterior_transform"] = None
        call_kwargs = {
            key: value
            for key, value in call_kwargs.items()
            if accepts_kwargs or key in parameters
        }

        base = accessor(X_all, **call_kwargs)
        return self._wrap_wide_posterior(
            base,
            X=X,
            selected=selected,
            posterior_transform=posterior_transform,
        )


class WideMultiTaskGP(_WidePosteriorMixin, MultiTaskGP):
    """Exact regression MultiTaskGP accepting wide targets with optional NaNs."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            task_feature=-1,
            all_tasks=list(range(num_tasks)),
            **kwargs,
        )
        self.num_tasks = num_tasks
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = torch.as_tensor(train_Y)


class WideMultiTaskBinaryClassificationGPModel(
    _WidePosteriorMixin,
    MultiTaskBinaryClassificationGPModel,
):
    """Binary multi-task GP accepting wide 0/1 targets with optional NaNs."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_Y_tensor = torch.as_tensor(train_Y)
        observed = train_Y_tensor[~torch.isnan(train_Y_tensor)]
        if not torch.all((observed == 0) | (observed == 1)):
            raise ValueError("Observed binary targets must be 0 or 1.")
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            num_tasks=num_tasks,
            task_feature=-1,
            **kwargs,
        )
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = train_Y_tensor

    def latent_posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior:
        """Return the latent binary posterior in public ``[..., q, m]`` form."""

        return self._wide_latent_posterior(
            X,
            output_indices=output_indices,
            posterior_transform=posterior_transform,
            **kwargs,
        )


class WideMultiTaskOrdinalGPModel(_WidePosteriorMixin, MultiTaskOrdinalGPModel):
    """Ordinal multi-task GP accepting wide class ids with optional NaNs."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_Y_tensor = torch.as_tensor(train_Y)
        observed = train_Y_tensor[~torch.isnan(train_Y_tensor)]
        if not torch.allclose(observed, observed.round()):
            raise ValueError("Observed ordinal targets must be integer-coded.")
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            num_tasks=num_tasks,
            task_feature=-1,
            **kwargs,
        )
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = train_Y_tensor


class WideMultiTaskMulticlassClassificationGPModel(
    _WidePosteriorMixin,
    MultiTaskMulticlassClassificationGPModel,
):
    """Multiclass multi-task GP accepting wide class ids with optional NaNs."""

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs: Any) -> None:
        train_Y_tensor = torch.as_tensor(train_Y)
        observed = train_Y_tensor[~torch.isnan(train_Y_tensor)]
        if not torch.allclose(observed, observed.round()):
            raise ValueError("Observed multiclass targets must be integer-coded.")
        X_long, Y_long, num_tasks = wide_to_long(train_X, train_Y)
        kwargs.pop("num_tasks", None)
        kwargs.pop("task_feature", None)
        super().__init__(
            train_X=X_long,
            train_Y=Y_long,
            num_tasks=num_tasks,
            task_feature=-1,
            **kwargs,
        )
        self.train_X_wide = torch.as_tensor(train_X)
        self.train_Y_wide = train_Y_tensor


__all__ = [
    "WideMultiTaskGP",
    "WideMultiTaskBinaryClassificationGPModel",
    "WideMultiTaskOrdinalGPModel",
    "WideMultiTaskMulticlassClassificationGPModel",
    "wide_to_long",
]
