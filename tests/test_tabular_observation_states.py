from __future__ import annotations

import pandas as pd
import pytest
import torch

from bochan.api import ExperimentFailureConfig, FitConfig
from bochan.tabular import (
    ObservationTabularDataset,
    TabularBayesianOptimizer,
    TabularDataConfig,
    dataframe_to_observation_tensors,
)


def test_dataframe_observation_conversion_keeps_partial_targets_and_masks_states() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "strength": [10.0, 99.0, 30.0, None],
            "conductivity": [None, 88.0, 40.0, 50.0],
            "status": ["success", "failed", "pending", "success"],
        }
    )
    dataset = dataframe_to_observation_tensors(
        data,
        TabularDataConfig(
            input_cols=["x"],
            target_cols=["strength", "conductivity"],
            experiment_status_col="status",
            target_missing_strategy="keep",
        ),
    )

    assert dataset.feature_names == ["x"]
    assert dataset.target_names == ["strength", "conductivity"]
    assert dataset.failed_mask.tolist() == [False, True, False, False]
    assert dataset.pending_mask.tolist() == [False, False, True, False]
    assert dataset.observed_mask.tolist() == [
        [True, False],
        [False, False],
        [False, False],
        [False, True],
    ]
    assert torch.isnan(dataset.Y[1]).all()
    assert torch.isnan(dataset.Y[2]).all()
    assert dataset.source_index.tolist() == [0, 1, 2, 3]


def test_target_drop_keeps_failed_and_pending_rows_for_state_learning() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "strength": [10.0, None, None, None],
            "status": ["success", "failed", "pending", "success"],
        }
    )
    dataset = dataframe_to_observation_tensors(
        data,
        TabularDataConfig(
            input_cols=["x"],
            target_cols=["strength"],
            experiment_status_col="status",
            target_missing_strategy="drop",
        ),
    )

    assert dataset.source_index.tolist() == [0, 1, 2]
    assert dataset.failed_mask.tolist() == [False, True, False]
    assert dataset.pending_mask.tolist() == [False, False, True]


def test_tabular_multitask_fit_preserves_unobserved_cells() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "strength": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "conductivity": [2.0, None, 3.0, 5.0, 6.0, 7.0],
        }
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="multitask",
        input_cols=["x"],
        target_cols=["strength", "conductivity"],
        target_missing_strategy="keep",
        skip_fit=True,
    )
    optimizer.fit(data)

    assert isinstance(optimizer.dataset, ObservationTabularDataset)
    assert type(optimizer.bo.model).__name__ == "WideMultiTaskGP"
    assert torch.isnan(optimizer.dataset.Y).sum().item() == 2
    assert torch.isnan(optimizer.bo.model.train_Y_wide).sum().item() == 2
    assert optimizer.bo.observations is not None
    assert optimizer.bo.observations.report()["observed_per_output"] == [5, 5]


def test_tabular_status_fits_success_model_from_all_completed_rows() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "strength": [1.0, None, 2.0, None, 3.0],
            "status": ["success", "failed", "success", "pending", "success"],
        }
    )
    failure = ExperimentFailureConfig(fit_config=FitConfig(skip_fit=True))
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["strength"],
        experiment_status_col="status",
        target_missing_strategy="keep",
        failure_config=failure,
        skip_fit=True,
    )
    optimizer.fit(data)

    observations = optimizer.bo.observations
    assert observations is not None
    assert observations.failed_mask.tolist() == [False, True, False, False, False]
    assert observations.pending_mask.tolist() == [False, False, False, True, False]
    assert optimizer.bo.failure_bundle is not None
    torch.testing.assert_close(
        optimizer.bo.failure_bundle.train_X.squeeze(-1),
        torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=optimizer.dataset.X.dtype),
    )
    torch.testing.assert_close(
        optimizer.bo.failure_bundle.train_Y.squeeze(-1),
        torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=optimizer.dataset.Y.dtype),
    )
    context = optimizer.bo._resolve_data_context(None)
    torch.testing.assert_close(context.X_pending, optimizer.dataset.X[3:4])


def test_tabular_observation_mode_rejects_generic_cross_validation() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "strength": [1.0, None, 3.0],
        }
    )
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x"],
        target_cols=["strength"],
        target_missing_strategy="keep",
        skip_fit=True,
    )

    with pytest.raises(ValueError, match="observation-aware validation"):
        optimizer.fit(data, cross_validation=True)


def test_unknown_experiment_status_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "strength": [1.0, None],
            "status": ["success", "aborted"],
        }
    )

    with pytest.raises(ValueError, match="success.*failed.*pending"):
        dataframe_to_observation_tensors(
            data,
            TabularDataConfig(
                input_cols=["x"],
                target_cols=["strength"],
                experiment_status_col="status",
                target_missing_strategy="keep",
            ),
        )
