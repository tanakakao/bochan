from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.tabular import TabularBayesianOptimizer


def _make_fitted_stub() -> TabularBayesianOptimizer:
    optimizer = TabularBayesianOptimizer(
        task_type="ordinal",
        model_type="base",
        input_cols=["x1"],
        target_cols=["quality_rank"],
    )
    optimizer.dataset = SimpleNamespace(
        target_names=["quality_rank"],
        target_category_maps={
            "quality_rank": {"low": 0, "medium": 1, "high": 2},
        },
    )
    return optimizer


def _prepare(optimizer: TabularBayesianOptimizer, acq_config=None, **values):
    resolved, _ = optimizer.candidates._prepare_configs(
        optimizer,
        acq_config,
        None,
        dict(values),
    )
    return resolved


def test_candidate_resolves_direct_string_ordinal_rank() -> None:
    optimizer = _make_fitted_stub()
    acq_config = _prepare(
        optimizer,
        acq_name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": "medium",
                    "sense": "ge",
                    "probability_threshold": 0.8,
                }
            ]
        },
    )
    assert acq_config.outcome_constraint_config is not None
    spec = acq_config.outcome_constraint_config.constraints[0]
    assert spec.output == "quality_rank"
    assert spec.rank == 1
    assert spec.sense == "ge"
    assert spec.probability_threshold == pytest.approx(0.8)


def test_candidate_resolves_nested_string_ordinal_rank() -> None:
    optimizer = _make_fitted_stub()
    acq_config = _prepare(
        optimizer,
        {
            "name": "ei",
            "outcome_constraint_config": {
                "constraints": {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": "high",
                    "sense": "eq",
                    "probability_threshold": 0.6,
                }
            },
        },
    )
    assert acq_config.outcome_constraint_config is not None
    assert acq_config.outcome_constraint_config.constraints[0].rank == 2


def test_candidate_preserves_integer_ordinal_rank() -> None:
    optimizer = _make_fitted_stub()
    acq_config = _prepare(
        optimizer,
        acq_name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": 1,
                    "sense": "ge",
                    "probability_threshold": 0.8,
                }
            ]
        },
    )
    assert acq_config.outcome_constraint_config.constraints[0].rank == 1


def test_candidate_rejects_unknown_string_ordinal_rank() -> None:
    optimizer = _make_fitted_stub()
    with pytest.raises(KeyError, match="Unknown target class label 'missing'"):
        _prepare(
            optimizer,
            acq_name="ei",
            outcome_constraint_config={
                "constraints": [
                    {
                        "kind": "ordinal_rank",
                        "output": "quality_rank",
                        "rank": "missing",
                        "sense": "ge",
                        "probability_threshold": 0.8,
                    }
                ]
            },
        )
