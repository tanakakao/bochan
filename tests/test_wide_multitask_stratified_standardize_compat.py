from __future__ import annotations

import pytest
import torch

from bochan.models import wide_multitask_compat


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.rand(4, 3, dtype=torch.double)
    train_Y = torch.rand(4, 2, dtype=torch.double)
    return train_X, train_Y


def test_build_stratified_standardize_supports_task_values(monkeypatch) -> None:
    class TaskValuesTransform:
        def __init__(
            self,
            task_values,
            stratification_idx,
            batch_shape=torch.Size(),
            min_stdv=1e-8,
        ) -> None:
            self.task_values = task_values
            self.stratification_idx = stratification_idx
            self.batch_shape = batch_shape
            self.min_stdv = min_stdv

    monkeypatch.setattr(
        wide_multitask_compat,
        "StratifiedStandardize",
        TaskValuesTransform,
    )
    train_X, train_Y = _data()

    transform = wide_multitask_compat._build_stratified_standardize(
        train_X,
        train_Y,
    )

    torch.testing.assert_close(
        transform.task_values,
        torch.tensor([0.0, 1.0], dtype=torch.double),
    )
    assert transform.stratification_idx == -1


def test_build_stratified_standardize_supports_all_task_values(monkeypatch) -> None:
    class AllTaskValuesTransform:
        def __init__(
            self,
            stratification_idx,
            all_task_values,
            batch_shape=torch.Size(),
            min_stdv=1e-8,
            dtype=torch.double,
        ) -> None:
            self.stratification_idx = stratification_idx
            self.all_task_values = all_task_values
            self.batch_shape = batch_shape
            self.min_stdv = min_stdv
            self.dtype = dtype

    monkeypatch.setattr(
        wide_multitask_compat,
        "StratifiedStandardize",
        AllTaskValuesTransform,
    )
    train_X, train_Y = _data()

    transform = wide_multitask_compat._build_stratified_standardize(
        train_X,
        train_Y,
    )

    torch.testing.assert_close(
        transform.all_task_values,
        torch.tensor([0.0, 1.0], dtype=torch.double),
    )
    assert transform.stratification_idx == -1
    assert transform.dtype == torch.double


def test_build_stratified_standardize_rejects_unknown_signature(monkeypatch) -> None:
    class UnsupportedTransform:
        def __init__(self, values) -> None:
            self.values = values

    monkeypatch.setattr(
        wide_multitask_compat,
        "StratifiedStandardize",
        UnsupportedTransform,
    )
    train_X, train_Y = _data()

    with pytest.raises(RuntimeError, match="Unsupported StratifiedStandardize"):
        wide_multitask_compat._build_stratified_standardize(train_X, train_Y)
