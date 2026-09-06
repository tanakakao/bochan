from __future__ import annotations

import math

from bochan.models.multifidelity import (
    augmented_branin_problem,
    run_fidelity_transfer_diagnostic,
)
from bochan.models.multifidelity.experiment import SyntheticBenchmarkConfig


def test_phase72_fidelity_transfer_smoke() -> None:
    problem = augmented_branin_problem()
    config = SyntheticBenchmarkConfig(
        n_initial=6,
        fit_maxiter=2,
        skip_fit=True,
    )

    diagnostics = run_fidelity_transfer_diagnostic(
        problem,
        seed=0,
        n_probe=8,
        config=config,
    )

    assert [item.source_fidelity for item in diagnostics] == [0.5, 0.75]
    for item in diagnostics:
        assert item.target_fidelity == 1.0
        assert item.n_probe == 8
        assert -1.0 <= item.mean_posterior_correlation <= 1.0
        assert -1.0 <= item.median_posterior_correlation <= 1.0
        assert -1.0 <= item.min_posterior_correlation <= 1.0
        assert -1.0 <= item.max_posterior_correlation <= 1.0
        assert 0.0 <= item.mean_squared_correlation <= 1.0
        assert 0.0 <= item.mean_target_variance_reduction_fraction <= 1.0
        assert item.mean_target_variance > 0.0
        assert math.isfinite(item.true_output_correlation)
        assert -1.0 <= item.true_output_correlation <= 1.0
