"""Shared multitask contracts for material-aware Gaussian models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gpytorch.kernels import MultitaskKernel
from torch import Tensor, nn

MaterialTaskMode = Literal["correlated", "independent"]


@dataclass(frozen=True)
class MaterialMultiTaskSpec:
    """Describe the output semantics of a material-aware surrogate.

    Args:
        mode: ``"correlated"`` for one shared multitask GP or ``"independent"``
            for one surrogate per output.
        num_tasks: Number of target columns represented by the surrogate.
    """

    mode: MaterialTaskMode
    num_tasks: int

    def __post_init__(self) -> None:
        if self.mode not in {"correlated", "independent"}:
            raise ValueError("mode must be 'correlated' or 'independent'.")
        if isinstance(self.num_tasks, bool) or not isinstance(self.num_tasks, int):
            raise TypeError("num_tasks must be an integer.")
        if self.num_tasks < 2:
            raise ValueError("num_tasks must be at least two for multitask models.")


def validate_wide_material_targets(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None = None,
    *,
    model_name: str = "material multitask model",
) -> int:
    """Validate the shared wide-target shape contract.

    This helper intentionally validates shape only. Observation-aware handling
    of missing/partial targets and known observation noise remains delegated to
    the established Gaussian model construction path so NaN masks and
    ``train_Yvar`` semantics are not reinterpreted here.

    Returns:
        Number of target columns.
    """

    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError(
            f"{model_name} requires wide train_Y with shape [n, m] and at least "
            "two target columns."
        )
    if train_Y.shape[0] != train_X.shape[0]:
        raise ValueError(
            "train_X and train_Y must contain the same number of observations: "
            f"{train_X.shape[0]} != {train_Y.shape[0]}."
        )
    if train_Yvar is not None:
        if train_Yvar.shape != train_Y.shape:
            raise ValueError(
                "train_Yvar must match wide train_Y shape exactly: "
                f"{tuple(train_Yvar.shape)} != {tuple(train_Y.shape)}."
            )
    return int(train_Y.shape[-1])


def validate_correlated_task_kernel(module: object, *, model_name: str) -> MultitaskKernel:
    """Return a correlated multitask kernel or raise a clear contract error."""

    if not isinstance(module, MultitaskKernel):
        raise RuntimeError(f"{model_name} requires a correlated MultitaskKernel.")
    return module


def task_covar_module(module: object, *, model_name: str) -> nn.Module:
    """Return the learned task covariance module from a correlated kernel."""

    return validate_correlated_task_kernel(module, model_name=model_name).task_covar_module


__all__ = [
    "MaterialMultiTaskSpec",
    "MaterialTaskMode",
    "task_covar_module",
    "validate_correlated_task_kernel",
    "validate_wide_material_targets",
]
