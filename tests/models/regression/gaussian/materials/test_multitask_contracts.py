"""Contract tests for material-aware multitask helpers."""

import pytest
import torch
from gpytorch.kernels import MultitaskKernel, RBFKernel

from bochan.models.regression.gaussian.materials import (
    MaterialMultiTaskSpec,
    task_covar_module,
    validate_correlated_task_kernel,
    validate_wide_material_targets,
)


def test_material_multitask_spec_accepts_correlated_and_independent() -> None:
    assert MaterialMultiTaskSpec("correlated", 3).num_tasks == 3
    assert MaterialMultiTaskSpec("independent", 2).mode == "independent"


@pytest.mark.parametrize("mode", ["unknown", "shared"])
def test_material_multitask_spec_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError):
        MaterialMultiTaskSpec(mode, 2)  # type: ignore[arg-type]


def test_validate_wide_material_targets_preserves_observation_values() -> None:
    train_X = torch.randn(4, 3, dtype=torch.double)
    train_Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 3.0], [float("nan"), 4.0], [5.0, 6.0]],
        dtype=torch.double,
    )
    train_Yvar = torch.tensor(
        [[0.1, float("nan")], [0.2, 0.3], [float("nan"), 0.4], [0.5, 0.6]],
        dtype=torch.double,
    )

    assert validate_wide_material_targets(train_X, train_Y, train_Yvar) == 2
    assert torch.isnan(train_Y[0, 1])
    assert torch.isnan(train_Yvar[2, 0])


def test_validate_wide_material_targets_requires_matching_yvar_shape() -> None:
    train_X = torch.randn(4, 3)
    train_Y = torch.randn(4, 2)
    with pytest.raises(ValueError, match="train_Yvar must match"):
        validate_wide_material_targets(train_X, train_Y, torch.randn(4, 1))


def test_validate_wide_material_targets_requires_two_outputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validate_wide_material_targets(torch.randn(4, 3), torch.randn(4, 1))


def test_correlated_kernel_helpers_expose_task_covariance() -> None:
    kernel = MultitaskKernel(RBFKernel(), num_tasks=3, rank=1)
    assert validate_correlated_task_kernel(kernel, model_name="Demo") is kernel
    assert task_covar_module(kernel, model_name="Demo") is kernel.task_covar_module


def test_correlated_kernel_helpers_reject_non_multitask_kernel() -> None:
    with pytest.raises(RuntimeError, match="requires a correlated MultitaskKernel"):
        validate_correlated_task_kernel(RBFKernel(), model_name="Demo")
