from __future__ import annotations

import pytest
import torch

from bochan.serving.webapp import target_results, target_settings
from bochan.serving.webapp import workflows as web_workflows
from bochan.serving.webapp.prediction_shapes import normalize_prediction_rows


def test_normalizer_keeps_regular_prediction_rows() -> None:
    values = torch.tensor([[1.0], [2.0]], dtype=torch.double)

    actual = normalize_prediction_rows(values, n_rows=2)

    torch.testing.assert_close(actual, values)


def test_normalizer_averages_consecutive_input_perturbation_rows() -> None:
    values = torch.tensor(
        [[1.0], [3.0], [10.0], [14.0]],
        dtype=torch.double,
    )

    actual = normalize_prediction_rows(values, n_rows=2)

    expected = torch.tensor([[2.0], [12.0]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)


def test_normalizer_applies_var_to_adverse_tail() -> None:
    values = torch.tensor(
        [[1.0], [2.0], [8.0], [9.0], [10.0], [20.0], [30.0], [40.0]],
        dtype=torch.double,
    )

    actual = normalize_prediction_rows(
        values,
        n_rows=2,
        risk_type="var",
        alpha=0.5,
    )

    expected = torch.tensor([[2.0], [20.0]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)


def test_normalizer_applies_cvar_to_adverse_tail() -> None:
    values = torch.tensor(
        [[1.0], [2.0], [8.0], [9.0], [10.0], [20.0], [30.0], [40.0]],
        dtype=torch.double,
    )

    actual = normalize_prediction_rows(
        values,
        n_rows=2,
        risk_type="cvar",
        alpha=0.5,
    )

    expected = torch.tensor([[1.5], [15.0]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)


def test_normalizer_handles_reported_80_by_1_shape() -> None:
    values = torch.arange(80, dtype=torch.double).reshape(80, 1)

    actual = normalize_prediction_rows(values, n_rows=5)

    expected = torch.tensor(
        [[7.5], [23.5], [39.5], [55.5], [71.5]],
        dtype=torch.double,
    )
    torch.testing.assert_close(actual, expected)


def test_normalizer_handles_leading_singleton_batch_dimension() -> None:
    values = torch.tensor(
        [[[1.0], [3.0], [10.0], [14.0]]],
        dtype=torch.double,
    )

    actual = normalize_prediction_rows(values, n_rows=2)

    expected = torch.tensor([[2.0], [12.0]], dtype=torch.double)
    torch.testing.assert_close(actual, expected)


def test_normalizer_rejects_non_divisible_expansion() -> None:
    with pytest.raises(RuntimeError, match="Could not normalize prediction"):
        normalize_prediction_rows(torch.ones(5, 1), n_rows=2)


def test_web_workflow_keeps_display_mean_and_installs_risk_baseline_adapter() -> None:
    assert target_settings._as_2d is normalize_prediction_rows
    assert target_results._as_2d is normalize_prediction_rows
    assert web_workflows._workflows_tabular._as_2d is not normalize_prediction_rows
