from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.tabular import TabularBayesianOptimizer, optimizer_api


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
            "quality_rank": {
                "low": 0,
                "medium": 1,
                "high": 2,
            },
        },
    )
    return optimizer


def test_candidate_resolves_direct_string_ordinal_rank(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)
    optimizer = _make_fitted_stub()

    result = optimizer.candidate(
        acq_name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": "medium",
                    "sense": "ge",
                    "probability_threshold": 0.8,
                },
            ],
        },
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    assert acq_config.outcome_constraint_config is not None
    spec = acq_config.outcome_constraint_config.constraints[0]
    assert spec.output == "quality_rank"
    assert spec.rank == 1
    assert spec.sense == "ge"
    assert spec.probability_threshold == pytest.approx(0.8)


def test_candidate_resolves_nested_string_ordinal_rank(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)
    optimizer = _make_fitted_stub()

    result = optimizer.candidate(
        acq_config={
            "name": "ei",
            "outcome_constraint_config": {
                "constraints": {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": "high",
                    "sense": "eq",
                    "probability_threshold": 0.6,
                },
            },
        },
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    constraint = acq_config["outcome_constraint_config"]["constraints"][0]
    assert constraint["rank"] == 2


def test_candidate_preserves_integer_ordinal_rank(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)
    optimizer = _make_fitted_stub()

    result = optimizer.candidate(
        acq_name="ei",
        outcome_constraint_config={
            "constraints": [
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": 1,
                    "sense": "ge",
                    "probability_threshold": 0.8,
                },
            ],
        },
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    spec = acq_config.outcome_constraint_config.constraints[0]
    assert spec.rank == 1


def test_candidate_rejects_unknown_string_ordinal_rank(monkeypatch) -> None:
    monkeypatch.setattr(
        optimizer_api._BaseTabularBayesianOptimizer,
        "candidate",
        lambda self, acq_config=None, opt_config=None, **kwargs: "ok",
    )
    optimizer = _make_fitted_stub()

    with pytest.raises(KeyError, match="Unknown target class label 'missing'"):
        optimizer.candidate(
            acq_name="ei",
            outcome_constraint_config={
                "constraints": [
                    {
                        "kind": "ordinal_rank",
                        "output": "quality_rank",
                        "rank": "missing",
                        "sense": "ge",
                        "probability_threshold": 0.8,
                    },
                ],
            },
        )
