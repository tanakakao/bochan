from __future__ import annotations

from types import SimpleNamespace

import pytest

from bochan.tabular import TabularBayesianOptimizer, optimizer_api


def _make_fitted_stub() -> TabularBayesianOptimizer:
    optimizer = TabularBayesianOptimizer(
        task_type="regression",
        model_type="base",
        input_cols=["x1"],
        target_cols=["property", "property2"],
    )
    optimizer.dataset = SimpleNamespace(target_names=["property", "property2"])
    return optimizer


def test_candidate_resolves_direct_named_outcome_constraint_outputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)
    optimizer = _make_fitted_stub()

    result = optimizer.candidate(
        acq_config={"name": "ehvi"},
        outcome_constraint_config={
            "output_indices": ["property", "property2"],
            "operators": ["ge", "le"],
            "thresholds": [0.5, 1.2],
        },
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    assert acq_config.outcome_constraint_config is not None
    assert acq_config.outcome_constraint_config.output_indices == [0, 1]


def test_candidate_resolves_nested_named_outcome_constraint_outputs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_candidate(self, acq_config=None, opt_config=None, **kwargs):
        captured["acq_config"] = acq_config
        return "ok"

    monkeypatch.setattr(optimizer_api._BaseTabularBayesianOptimizer, "candidate", fake_candidate)
    optimizer = _make_fitted_stub()

    result = optimizer.candidate(
        acq_config={
            "name": "ehvi",
            "outcome_constraint_config": {
                "output_indices": ["property", "property2"],
                "operators": ["ge", "le"],
                "thresholds": [0.5, 1.2],
            },
        }
    )

    assert result == "ok"
    acq_config = captured["acq_config"]
    assert acq_config["outcome_constraint_config"]["output_indices"] == [0, 1]


def test_candidate_rejects_unknown_named_outcome_constraint_output(monkeypatch) -> None:
    monkeypatch.setattr(
        optimizer_api._BaseTabularBayesianOptimizer,
        "candidate",
        lambda self, acq_config=None, opt_config=None, **kwargs: "ok",
    )
    optimizer = _make_fitted_stub()

    with pytest.raises(KeyError, match="Unknown column 'missing'"):
        optimizer.candidate(
            acq_config={"name": "ehvi"},
            outcome_constraint_config={
                "output_indices": ["missing"],
                "operators": ["ge"],
                "thresholds": [0.5],
            },
        )
