from __future__ import annotations

import math

import pytest

from bochan.models.multifidelity.diagnostics import (
    _mfkg_configs,
    compare_mfkg_cost_awareness,
    run_mfkg_diagnostic,
)
from bochan.models.multifidelity.experiment import SyntheticBenchmarkConfig
from bochan.models.multifidelity.synthetic import augmented_branin_problem


def _smoke_config() -> SyntheticBenchmarkConfig:
    return SyntheticBenchmarkConfig(
        n_initial=3,
        budget=3.0,
        max_steps=1,
        num_restarts=1,
        raw_samples=8,
        fit_maxiter=1,
        num_fantasies=2,
        skip_fit=True,
    )


def test_mfkg_diagnostic_configs_differ_only_by_cost_contract():
    problem = augmented_branin_problem()
    config = _smoke_config()

    aware_acq, aware_opt = _mfkg_configs(problem, config, cost_aware=True)
    plain_acq, plain_opt = _mfkg_configs(problem, config, cost_aware=False)

    assert aware_acq.name == plain_acq.name == "mfkg"
    assert aware_acq.acqf_kwargs["cost_config"] == problem.cost_config
    assert "cost_config" not in plain_acq.acqf_kwargs
    assert aware_acq.acqf_kwargs["num_fantasies"] == plain_acq.acqf_kwargs["num_fantasies"]
    assert aware_opt.fidelity_values == plain_opt.fidelity_values == problem.fidelity_values
    assert aware_opt.num_restarts == plain_opt.num_restarts
    assert aware_opt.raw_samples == plain_opt.raw_samples


def test_cost_aware_diagnostic_requires_cost_config():
    problem = augmented_branin_problem()
    object.__setattr__(problem, "cost_config", None)

    with pytest.raises(ValueError, match="cost_config"):
        _mfkg_configs(problem, _smoke_config(), cost_aware=True)


def test_diagnostic_smoke_preserves_initial_rows_and_marks_acquisition_unknown():
    problem = augmented_branin_problem()
    result = run_mfkg_diagnostic(problem, seed=5, config=_smoke_config(), cost_aware=False)

    assert result.run.strategy == "mfkg_unweighted"
    assert result.cost_aware is False
    assert result.acquisition_values.shape == (3,)
    assert all(math.isnan(float(value)) for value in result.acquisition_values.tolist())

    rows = result.rows()
    assert len(rows) == 3
    assert rows[0]["cost_aware"] is False
    assert math.isnan(float(rows[0]["acquisition_value"]))
    assert result.fidelity_counts() == {
        level: 0 for level in sorted(set(problem.fidelity_values))
    }


def test_cost_comparison_uses_identical_initial_design():
    problem = augmented_branin_problem()
    config = _smoke_config()

    aware, unweighted = compare_mfkg_cost_awareness(problem, seed=11, config=config)

    assert aware.cost_aware is True
    assert unweighted.cost_aware is False
    assert aware.run.strategy == "mfkg"
    assert unweighted.run.strategy == "mfkg_unweighted"
    assert aware.run.X.equal(unweighted.run.X)
    assert aware.run.Y.equal(unweighted.run.Y)
    assert aware.run.costs.equal(unweighted.run.costs)
