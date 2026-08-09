from __future__ import annotations

import pytest
import torch

from bochan.api.observation import ObservationData


def test_observation_data_separates_missing_failed_and_pending() -> None:
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.double)
    Y = torch.tensor(
        [
            [1.0, float("nan")],
            [float("nan"), float("nan")],
            [2.0, 4.0],
            [float("nan"), 5.0],
        ],
        dtype=torch.double,
    )
    observations = ObservationData.from_status(
        X,
        Y,
        status=["success", "failed", "pending", "success"],
    )

    assert observations.observed_mask.tolist() == [
        [True, False],
        [False, False],
        [False, False],
        [False, True],
    ]
    assert observations.failed_mask.tolist() == [False, True, False, False]
    assert observations.pending_mask.tolist() == [False, False, True, False]
    torch.testing.assert_close(observations.pending_X, X[2:3])

    success_X, success_y = observations.success_training_data()
    torch.testing.assert_close(success_X, X[[0, 1, 3]])
    torch.testing.assert_close(
        success_y,
        torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.double),
    )


def test_output_training_data_uses_only_successful_observed_rows() -> None:
    X = torch.arange(5, dtype=torch.double).unsqueeze(-1)
    Y = torch.tensor(
        [[1.0, 10.0], [2.0, float("nan")], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        dtype=torch.double,
    )
    observations = ObservationData(
        X=X,
        Y=Y,
        failed_mask=[False, False, True, False, False],
        pending_mask=[False, False, False, True, False],
    )

    X0, Y0 = observations.output_training_data(0)
    X1, Y1 = observations.output_training_data(1)
    torch.testing.assert_close(X0.squeeze(-1), torch.tensor([0.0, 1.0, 4.0], dtype=torch.double))
    torch.testing.assert_close(Y0.squeeze(-1), torch.tensor([1.0, 2.0, 5.0], dtype=torch.double))
    torch.testing.assert_close(X1.squeeze(-1), torch.tensor([0.0, 4.0], dtype=torch.double))
    torch.testing.assert_close(Y1.squeeze(-1), torch.tensor([10.0, 50.0], dtype=torch.double))


def test_observation_status_validation_is_explicit() -> None:
    X = torch.zeros(2, 1, dtype=torch.double)
    Y = torch.ones(2, 1, dtype=torch.double)

    with pytest.raises(ValueError, match="success.*failed.*pending"):
        ObservationData.from_status(X, Y, status=["success", "unknown"])

    with pytest.raises(ValueError, match="both failed and pending"):
        ObservationData(
            X=X,
            Y=Y,
            failed_mask=[True, False],
            pending_mask=[True, False],
        )


def test_observation_report_keeps_cellwise_counts() -> None:
    X = torch.zeros(3, 2, dtype=torch.double)
    Y = torch.tensor(
        [[1.0, float("nan")], [2.0, 3.0], [float("nan"), float("nan")]],
        dtype=torch.double,
    )
    observations = ObservationData.from_status(
        X,
        Y,
        status=["success", "success", "failed"],
    )

    assert observations.report() == {
        "n_rows": 3,
        "n_completed": 3,
        "n_success": 2,
        "n_failed": 1,
        "n_pending": 0,
        "observed_per_output": [2, 1],
    }
