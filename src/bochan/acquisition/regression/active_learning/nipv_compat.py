from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from . import multi_output as _multi_output


ReductionType = Literal["mean", "sum", "max", "min"]
OutputReductionType = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_sum",
    "weighted_mean",
]

_ORIGINAL_INIT_ATTR = "_bochan_original_init_before_nipv_compat"
_ORIGINAL_FORWARD_ATTR = "_bochan_original_forward_before_nipv_compat"


def _prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for size in shape:
        out *= int(size)
    return out


def _reduce(tensor: Tensor, *, dim: int | tuple[int, ...], mode: str) -> Tensor:
    if mode == "mean":
        return tensor.mean(dim=dim)
    if mode == "sum":
        return tensor.sum(dim=dim)
    if mode == "max":
        if isinstance(dim, tuple):
            for current_dim in sorted(dim, reverse=True):
                tensor = tensor.max(dim=current_dim).values
            return tensor
        return tensor.max(dim=dim).values
    if mode == "min":
        if isinstance(dim, tuple):
            for current_dim in sorted(dim, reverse=True):
                tensor = tensor.min(dim=current_dim).values
            return tensor
        return tensor.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode!r}.")


def _find_target_block(shape: tuple[int, ...], target_shape: tuple[int, ...]) -> int | None:
    if len(target_shape) == 0:
        return 0
    if len(shape) < len(target_shape):
        return None
    for start in range(len(shape) - len(target_shape) + 1):
        if shape[start : start + len(target_shape)] == target_shape:
            return start
    return None


def _output_weights_like(self, value: Tensor) -> Tensor | None:
    weights = getattr(self, "_nipv_output_weights", None)
    if weights is None:
        return None
    if int(value.shape[-1]) != int(weights.numel()):
        raise ValueError(
            "qMultiOutputRegressionNegIntegratedPosteriorVariance: "
            f"output dimension={value.shape[-1]} does not match "
            f"output_weights length={weights.numel()}."
        )
    weights = weights.to(device=value.device, dtype=value.dtype)
    if bool(getattr(self, "normalize_output_weights", True)):
        weights = weights / weights.sum().clamp_min(torch.finfo(value.dtype).eps)
    return weights


def _reduce_output_dim(self, value: Tensor) -> Tensor:
    if value.ndim == 0:
        return value
    if int(value.shape[-1]) == 1:
        return value.squeeze(-1)

    mode = str(getattr(self, "output_reduction", "mean"))
    if mode == "weighted_sum":
        weights = _output_weights_like(self, value)
        if weights is None:
            raise ValueError("output_reduction='weighted_sum' requires output_weights.")
        return (value * weights).sum(dim=-1)
    if mode == "weighted_mean":
        weights = _output_weights_like(self, value)
        if weights is None:
            return value.mean(dim=-1)
        return (value * weights).sum(dim=-1)
    return _reduce(value, dim=-1, mode=mode)


def _finalize_nipv_output(self, value: Tensor, X: Tensor) -> Tensor:
    """Reduce integration and output dimensions while preserving t-batch shape.

    BoTorch's qNegIntegratedPosteriorVariance normally returns ``batch_shape``.
    Some multi-output / hybrid models instead return values such as
    ``[n_mc, *batch_shape, m]``. For example:

        value.shape = [60, 32, 2]
        X.shape     = [32, 10, 5]

    Here ``60`` is the integration-point dimension, ``32`` is the t-batch
    dimension, and ``2`` is the output dimension. This function reduces the
    first and last dimensions and returns ``[32]``.
    """
    value = torch.as_tensor(value)
    target_shape = tuple(X.shape[:-2])

    if tuple(value.shape) == target_shape:
        return value
    if value.ndim == 0:
        return value.expand(*target_shape) if len(target_shape) > 0 else value
    if len(target_shape) == 0:
        return value.mean()

    start = _find_target_block(tuple(value.shape), target_shape)
    if start is not None:
        # Dimensions before the preserved t-batch block are integration / fantasy
        # dimensions and are reduced together.
        if start > 0:
            leading_dims = tuple(range(start))
            value = _reduce(
                value,
                dim=leading_dims,
                mode=str(getattr(self, "integration_reduction", "mean")),
            )

        # The preserved batch block is now at the beginning. Any dimensions after
        # it are extra integration dimensions and the final multi-output dimension.
        trailing_count = value.ndim - len(target_shape)
        if trailing_count > 1:
            extra_dims = tuple(range(len(target_shape), value.ndim - 1))
            value = _reduce(
                value,
                dim=extra_dims,
                mode=str(getattr(self, "integration_reduction", "mean")),
            )
        if value.ndim == len(target_shape) + 1:
            value = _reduce_output_dim(self, value)

        if tuple(value.shape) == target_shape:
            return value

    # Equivalent-size fallback after reduction.
    if value.numel() == _prod(target_shape):
        return value.reshape(target_shape)

    # If the t-batch is one-dimensional, preserve the matching dimension and
    # reduce everything else. This covers uncommon axis orders such as [m, B, n_mc].
    if len(target_shape) == 1:
        batch_size = int(target_shape[0])
        matching_dims = [dim for dim, size in enumerate(value.shape) if int(size) == batch_size]
        if len(matching_dims) == 1:
            batch_dim = matching_dims[0]
            value = value.movedim(batch_dim, 0)
            while value.ndim > 2:
                value = _reduce(
                    value,
                    dim=1,
                    mode=str(getattr(self, "integration_reduction", "mean")),
                )
            if value.ndim == 2:
                value = _reduce_output_dim(self, value)
            if tuple(value.shape) == target_shape:
                return value

    raise RuntimeError(
        "qMultiOutputRegressionNegIntegratedPosteriorVariance: could not align "
        f"inner acquisition output shape={tuple(value.shape)} to "
        f"t-batch shape={target_shape} for X.shape={tuple(X.shape)}."
    )


def apply_multioutput_nipv_compat() -> None:
    """Patch multi-output NIPV initialization and output-shape handling."""
    cls = _multi_output.qMultiOutputRegressionNegIntegratedPosteriorVariance

    if not hasattr(cls, _ORIGINAL_INIT_ATTR):
        original_init = cls.__init__
        setattr(cls, _ORIGINAL_INIT_ATTR, original_init)

        def _init(
            self,
            model,
            mc_points: Tensor,
            *,
            sampler=None,
            objective=None,
            posterior_transform=None,
            X_pending: Tensor | None = None,
            integration_reduction: ReductionType = "mean",
            output_reduction: OutputReductionType = "mean",
            output_weights: Tensor | Sequence[float] | None = None,
            normalize_output_weights: bool = True,
            **kwargs,
        ) -> None:
            if integration_reduction not in {"mean", "sum", "max", "min"}:
                raise ValueError("integration_reduction must be mean, sum, max, or min.")
            if output_reduction not in {
                "mean",
                "sum",
                "max",
                "min",
                "weighted_sum",
                "weighted_mean",
            }:
                raise ValueError(
                    "output_reduction must be mean, sum, max, min, "
                    "weighted_sum, or weighted_mean."
                )

            original_init(
                self,
                model=model,
                mc_points=mc_points,
                sampler=sampler,
                objective=objective,
                posterior_transform=posterior_transform,
                X_pending=X_pending,
                **kwargs,
            )
            self.integration_reduction = integration_reduction
            self.output_reduction = output_reduction
            self.normalize_output_weights = bool(normalize_output_weights)
            if output_weights is None:
                self._nipv_output_weights = None
            else:
                weights = torch.as_tensor(output_weights).reshape(-1)
                self.register_buffer("_nipv_output_weights", weights.detach().clone())

        cls.__init__ = _init

    if not hasattr(cls, _ORIGINAL_FORWARD_ATTR):
        setattr(cls, _ORIGINAL_FORWARD_ATTR, cls.forward)

        @t_batch_mode_transform()
        def _forward(self, X: Tensor) -> Tensor:
            value = self.acqf(X)
            return _finalize_nipv_output(self, value, X)

        cls.forward = _forward


apply_multioutput_nipv_compat()


__all__ = ["apply_multioutput_nipv_compat"]
