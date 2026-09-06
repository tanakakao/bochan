from bochan.models.multifidelity.experiment import SyntheticBenchmarkConfig
from bochan.models.multifidelity.fidelity_score_diagnostics import (
    run_mfkg_fidelity_score_diagnostic,
)
from bochan.models.multifidelity.synthetic import augmented_branin_problem


def test_phase71_zero_step_smoke():
    problem = augmented_branin_problem(fixed_cost=0.25)
    result = run_mfkg_fidelity_score_diagnostic(
        problem,
        seed=0,
        config=SyntheticBenchmarkConfig(n_initial=4, max_steps=0, fit_maxiter=5),
    )
    assert result.rows() == []
    assert result.run.initial_count == 4
    assert result.run.X.shape[0] == 4
