from __future__ import annotations

from types import SimpleNamespace

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


def test_web_workflow_uses_source_level_prediction_row_normalization() -> None:
    values = torch.tensor(
        [[1.0], [3.0], [10.0], [14.0]],
        dtype=torch.double,
    )
    expected = torch.tensor([[2.0], [12.0]], dtype=torch.double)

    torch.testing.assert_close(target_settings._as_2d(values, n_rows=2), expected)
    torch.testing.assert_close(target_results._as_2d(values, n_rows=2), expected)
    assert target_results._as_2d is target_settings._as_2d
    assert not hasattr(web_workflows._workflows_tabular, "_as_2d")


class _Posterior:
    def __init__(self, mean: torch.Tensor, variance: torch.Tensor) -> None:
        self.mean = mean
        self.variance = variance


class _ExpandedHybridModel:
    """Mimic n_w=16 class-probability expansion from InputPerturbation."""

    def __init__(self, nominal_probabilities: torch.Tensor) -> None:
        self.nominal_probabilities = nominal_probabilities

    def posterior(
        self,
        X: torch.Tensor,
        *,
        output_mode: str = "mean",
        output_indices: list[int] | None = None,
    ) -> _Posterior:
        del output_indices
        n_rows = int(X.shape[0])
        if output_mode == "probability":
            probability = self.nominal_probabilities[:, :1]
            mean = probability.repeat_interleave(16, dim=0)
        else:
            mean = torch.zeros(n_rows * 16, 1, dtype=X.dtype, device=X.device)
        variance = torch.full_like(mean, 0.04)
        return _Posterior(mean, variance)

    def class_probs_list(
        self,
        X: torch.Tensor,
        *,
        output_indices: list[str] | None = None,
    ) -> list[torch.Tensor]:
        del X, output_indices
        return [self.nominal_probabilities.repeat_interleave(16, dim=0)]


def test_display_predictions_aggregates_multiclass_probability_rows() -> None:
    """Web YY data must keep one probability row per nominal observation."""

    X = torch.arange(8, dtype=torch.double).reshape(4, 2)
    nominal = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
        ],
        dtype=torch.double,
    )
    optimizer = SimpleNamespace(model=_ExpandedHybridModel(nominal))

    _, class_probabilities = target_results._display_predictions(
        optimizer,
        X,
        target_columns=["class"],
        target_metadata={"class": {"internal_task": "multiclass"}},
        hybrid_model=True,
    )

    assert class_probabilities["class"].shape == (4, 3)
    torch.testing.assert_close(class_probabilities["class"], nominal)


def test_display_predictions_aggregates_ordinal_probability_rows() -> None:
    """Expected-rank display must also use nominal rows after perturbation."""

    X = torch.arange(8, dtype=torch.double).reshape(4, 2)
    nominal = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
        ],
        dtype=torch.double,
    )
    optimizer = SimpleNamespace(model=_ExpandedHybridModel(nominal))

    display, class_probabilities = target_results._display_predictions(
        optimizer,
        X,
        target_columns=["class"],
        target_metadata={
            "class": {
                "internal_task": "ordinal",
                "num_classes": 3,
            }
        },
        hybrid_model=True,
    )

    ranks = torch.arange(3, dtype=torch.double)
    expected_mean = (nominal * ranks).sum(dim=-1)
    assert display["class"]["mean"].shape == (4,)
    torch.testing.assert_close(display["class"]["mean"], expected_mean)
    torch.testing.assert_close(class_probabilities["class"], nominal)
