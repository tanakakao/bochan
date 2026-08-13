from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.tabular import TabularBayesianOptimizer


def _make_fitted_stub() -> TabularBayesianOptimizer:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols=["property", "property2"],
    )
    optimizer.dataset = SimpleNamespace(
        target_names=["property", "property2"],
        target_category_maps={},
    )
    return optimizer


def _prepare_acquisition(
    optimizer: TabularBayesianOptimizer,
    acq_config,
    **values,
):
    resolved, _ = optimizer.candidates._prepare_configs(
        optimizer,
        acq_config,
        None,
        dict(values),
    )
    return resolved


def test_candidate_resolves_direct_named_outcome_constraint_outputs() -> None:
    optimizer = _make_fitted_stub()
    acq_config = _prepare_acquisition(
        optimizer,
        {"name": "ehvi"},
        outcome_constraint_config={
            "output_indices": ["property", "property2"],
            "operators": ["ge", "le"],
            "thresholds": [0.5, 1.2],
        },
    )
    assert acq_config.outcome_constraint_config is not None
    assert acq_config.outcome_constraint_config["output_indices"] == [0, 1]


def test_candidate_resolves_nested_named_outcome_constraint_outputs() -> None:
    optimizer = _make_fitted_stub()
    acq_config = _prepare_acquisition(
        optimizer,
        {
            "name": "ehvi",
            "outcome_constraint_config": {
                "output_indices": ["property", "property2"],
                "operators": ["ge", "le"],
                "thresholds": [0.5, 1.2],
            },
        },
    )
    assert acq_config.outcome_constraint_config is not None
    assert acq_config.outcome_constraint_config["output_indices"] == [0, 1]


def test_candidate_rejects_unknown_named_outcome_constraint_output() -> None:
    optimizer = _make_fitted_stub()
    with pytest.raises(KeyError, match="Unknown column 'missing'"):
        _prepare_acquisition(
            optimizer,
            {"name": "ehvi"},
            outcome_constraint_config={
                "output_indices": ["missing"],
                "operators": ["ge"],
                "thresholds": [0.5],
            },
        )
