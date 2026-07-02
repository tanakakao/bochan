from __future__ import annotations

import importlib

import pytest
import torch

from bochan.acquisition._nparego_shape import (
    reduce_nparego_sample_and_q_to_tbatch,
)


MODULE_NAMES = [
    "bochan.acquisition.binary.bayesian_optimization.multi_output",
    "bochan.acquisition.ordinal.bayesian_optimization.multi_output",
    "bochan.acquisition.multiclass.bayesian_optimization.multi_output",
]


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_multioutput_nparego_modules_use_shared_reducer(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert (
        module._reduce_sample_and_q_to_tbatch
        is reduce_nparego_sample_and_q_to_tbatch
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_q1_squeezed_axis_preserves_optimizer_batch(module_name: str) -> None:
    module = importlib.import_module(module_name)
    X = torch.zeros(32, 1, 5, dtype=torch.double)
    value = torch.arange(128 * 32, dtype=torch.double).reshape(128, 32)

    result = module._reduce_sample_and_q_to_tbatch(value, X)

    assert result.shape == torch.Size([32])
    torch.testing.assert_close(result, value.mean(dim=0))


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_q1_explicit_axis_preserves_optimizer_batch(module_name: str) -> None:
    module = importlib.import_module(module_name)
    X = torch.zeros(32, 1, 5, dtype=torch.double)
    value = torch.arange(128 * 32, dtype=torch.double).reshape(128, 32, 1)

    result = module._reduce_sample_and_q_to_tbatch(value, X)

    assert result.shape == torch.Size([32])
    torch.testing.assert_close(result, value.squeeze(-1).mean(dim=0))


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_q_batch_is_reduced_before_mc_samples(module_name: str) -> None:
    module = importlib.import_module(module_name)
    X = torch.zeros(32, 3, 5, dtype=torch.double)
    value = torch.arange(128 * 32 * 3, dtype=torch.double).reshape(128, 32, 3)

    result = module._reduce_sample_and_q_to_tbatch(value, X)

    assert result.shape == torch.Size([32])
    torch.testing.assert_close(result, value.max(dim=-1).values.mean(dim=0))


def test_q1_without_tbatch_reduces_to_scalar() -> None:
    X = torch.zeros(1, 5, dtype=torch.double)
    value = torch.arange(128, dtype=torch.double)

    result = reduce_nparego_sample_and_q_to_tbatch(value, X)

    assert result.shape == torch.Size([])
    torch.testing.assert_close(result, value.mean())
