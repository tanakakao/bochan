"""Tabular cross-validation integration tests."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from bochan.api import CrossValidationConfig
from bochan.tabular import TabularBayesianOptimizer


def test_tabular_fit_runs_cv_then_full_fit_and_clears_stale_result() -> None:
    """CV uses converted tensors while the regular full-data fit still runs."""
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        input_cols=["x"],
        target_cols="y",
        cross_validation=True,
        cv_config={"n_splits": 2},
    )
    calls: list[tuple[str, int]] = []
    marker = object()
    optimizer.bo.cross_validate = lambda x, y, **kwargs: (calls.append(("cv", len(x))) or marker)
    optimizer.bo.fit = lambda x, y, **kwargs: calls.append(("fit", len(x)))
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 4.0, 9.0]})

    assert optimizer.fit(frame) is optimizer
    assert calls == [("cv", 4), ("fit", 4)]
    assert optimizer.cross_validation_result_ is marker

    calls.clear()
    optimizer.fit(frame, cross_validation=False)
    assert calls == [("fit", 4)]
    assert optimizer.cross_validation_result_ is None


def test_tabular_accepts_core_cv_config() -> None:
    """The Python API preserves custom/core configuration support."""
    config = CrossValidationConfig(n_splits=3)
    optimizer = TabularBayesianOptimizer(cv_config=config)
    assert optimizer.cv_config is config
