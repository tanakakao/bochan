"""Fixed known observation noise for correlated multitask GPs."""

from __future__ import annotations

import warnings

import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.utils.warnings import GPInputWarning
from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor


class MultitaskFixedNoiseGaussianLikelihood(FixedNoiseGaussianLikelihood):
    """Fixed per-observation, per-task variance for wide multitask targets.

    The public noise contract is ``[..., n, m]``. The likelihood converts this
    wide tensor to the covariance event order at the likelihood boundary, using
    ``MultitaskMultivariateNormal._interleaved`` rather than assuming one layout.

    ``allow_missing=True`` is reserved for partially observed correlated
    multitask training. In that mode, ``NaN`` marks an unobserved target cell.
    The underlying fixed-noise covariance stores a finite placeholder at those
    positions; the corresponding target entries are removed by the exact-GP
    missing-observation mask, so the placeholder is never used as an observed
    variance and is not an imputation.
    """

    def __init__(
        self,
        noise: Tensor,
        *,
        num_tasks: int | None = None,
        allow_missing: bool = False,
    ) -> None:
        resolved_num_tasks = self._validate_wide_noise(
            noise,
            num_tasks=num_tasks,
            argument_name="noise",
            allow_missing=allow_missing,
        )
        missing_mask = torch.isnan(noise) if allow_missing else torch.zeros_like(
            noise, dtype=torch.bool
        )
        stored_noise = torch.where(missing_mask, torch.ones_like(noise), noise)
        super().__init__(
            noise=self._flatten_event(stored_noise, interleaved=True),
            learn_additional_noise=False,
        )
        self.num_tasks = resolved_num_tasks
        self.allow_missing = bool(allow_missing)
        self.register_buffer("_missing_noise_mask", missing_mask.detach().clone())

    @staticmethod
    def _validate_wide_noise(
        noise: Tensor,
        *,
        num_tasks: int | None,
        argument_name: str,
        allow_missing: bool = False,
    ) -> int:
        if not torch.is_tensor(noise):
            raise TypeError(f"{argument_name} must be a Tensor.")
        if noise.ndim < 2:
            raise ValueError(
                f"{argument_name} must have shape [..., n, m] for multitask noise."
            )
        inferred_num_tasks = int(noise.shape[-1])
        if inferred_num_tasks < 2:
            raise ValueError(
                f"{argument_name} must contain at least two task columns."
            )
        if num_tasks is not None and inferred_num_tasks != int(num_tasks):
            raise ValueError(
                f"{argument_name} task dimension does not match num_tasks: "
                f"{inferred_num_tasks} != {int(num_tasks)}."
            )

        if allow_missing:
            if torch.isinf(noise).any():
                raise ValueError(
                    f"{argument_name} must contain finite variances or NaN missing markers."
                )
            observed = ~torch.isnan(noise)
        else:
            if not torch.isfinite(noise).all():
                raise ValueError(f"{argument_name} must contain only finite variances.")
            observed = torch.ones_like(noise, dtype=torch.bool)

        if observed.any() and (noise[observed] <= 0).any():
            raise ValueError(
                f"{argument_name} must contain strictly positive variances."
            )
        return inferred_num_tasks

    @staticmethod
    def _flatten_event(noise: Tensor, *, interleaved: bool) -> Tensor:
        ordered = noise if interleaved else noise.transpose(-1, -2)
        return ordered.reshape(*ordered.shape[:-2], -1)

    @property
    def _stored_task_noise(self) -> Tensor:
        flat_noise = self.noise_covar.noise
        if flat_noise.shape[-1] % self.num_tasks != 0:
            raise RuntimeError(
                "Stored fixed noise is incompatible with the configured task count."
            )
        num_data = flat_noise.shape[-1] // self.num_tasks
        return flat_noise.reshape(
            *flat_noise.shape[:-1],
            num_data,
            self.num_tasks,
        )

    @property
    def missing_noise_mask(self) -> Tensor:
        """Return the canonical wide mask of unobserved variance cells."""

        return self._missing_noise_mask

    @property
    def task_noise(self) -> Tensor:
        """Return fixed training variance in natural ``[..., n, m]`` shape."""

        stored = self._stored_task_noise
        if not bool(self._missing_noise_mask.any()):
            return stored
        return stored.masked_fill(self._missing_noise_mask, float("nan"))

    def _shaped_noise_covar(
        self,
        base_shape: torch.Size,
        *params,
        **kwargs,
    ) -> Tensor | LinearOperator:
        del params
        if len(base_shape) < 2 or int(base_shape[-1]) != self.num_tasks:
            raise ValueError(
                "Multitask fixed noise expects an event shape [..., n, m] with "
                f"m={self.num_tasks}; got {tuple(base_shape)}."
            )

        interleaved = bool(kwargs.pop("_interleaved", True))
        explicit_noise = kwargs.pop("noise", None)
        flat_shape = torch.Size(
            (*base_shape[:-2], int(base_shape[-2]) * self.num_tasks)
        )
        if explicit_noise is not None:
            # Test-time observation noise represents actual requested observations,
            # so it remains strict even when the stored training table is partial.
            self._validate_wide_noise(
                explicit_noise,
                num_tasks=self.num_tasks,
                argument_name="noise",
                allow_missing=False,
            )
            if tuple(explicit_noise.shape[-2:]) != tuple(base_shape[-2:]):
                raise ValueError(
                    "Explicit multitask observation noise must match the requested "
                    f"[n, m] event shape: {tuple(explicit_noise.shape[-2:])} != "
                    f"{tuple(base_shape[-2:])}."
                )
            return self.noise_covar(
                shape=flat_shape,
                noise=self._flatten_event(
                    explicit_noise,
                    interleaved=interleaved,
                ),
                **kwargs,
            )

        stored_noise = self._stored_task_noise
        if int(stored_noise.shape[-2]) == int(base_shape[-2]):
            return self.noise_covar(
                shape=flat_shape,
                noise=self._flatten_event(
                    stored_noise,
                    interleaved=interleaved,
                ),
                **kwargs,
            )

        result = self.noise_covar(shape=flat_shape, **kwargs)
        if isinstance(result, ZeroLinearOperator):
            warnings.warn(
                "The requested multitask event size does not match the stored fixed "
                "training noise and no explicit observation noise was supplied. "
                "This is treated as a no-op.",
                GPInputWarning,
                stacklevel=2,
            )
        return result

    def marginal(
        self,
        function_dist: MultitaskMultivariateNormal,
        *params,
        **kwargs,
    ) -> MultitaskMultivariateNormal:
        """Add fixed noise while preserving the multitask event ordering."""

        if not isinstance(function_dist, MultitaskMultivariateNormal):
            raise TypeError(
                "MultitaskFixedNoiseGaussianLikelihood requires a "
                "MultitaskMultivariateNormal."
            )
        mean = function_dist.mean
        covariance = function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(
            mean.shape,
            *params,
            _interleaved=bool(function_dist._interleaved),
            **kwargs,
        )
        return MultitaskMultivariateNormal(
            mean,
            covariance + noise_covar,
            interleaved=bool(function_dist._interleaved),
        )

    def get_fantasy_likelihood(self, **kwargs):
        """Append wide fixed noise for fantasy observations along the data axis."""

        if "noise" not in kwargs:
            raise RuntimeError(
                "MultitaskFixedNoiseGaussianLikelihood.fantasize requires a "
                "wide `noise` kwarg with shape [..., q, m]."
            )
        new_noise = kwargs["noise"]
        self._validate_wide_noise(
            new_noise,
            num_tasks=self.num_tasks,
            argument_name="noise",
            allow_missing=False,
        )
        old_noise = self.task_noise
        batch_shape = torch.broadcast_shapes(
            old_noise.shape[:-2],
            new_noise.shape[:-2],
        )
        old_noise = old_noise.expand(
            *batch_shape,
            old_noise.shape[-2],
            self.num_tasks,
        )
        new_noise = new_noise.expand(
            *batch_shape,
            new_noise.shape[-2],
            self.num_tasks,
        )
        fantasy = type(self)(
            torch.cat([old_noise, new_noise], dim=-2),
            num_tasks=self.num_tasks,
            allow_missing=self.allow_missing,
        )
        fantasy.train(self.training)
        return fantasy


__all__ = ["MultitaskFixedNoiseGaussianLikelihood"]
