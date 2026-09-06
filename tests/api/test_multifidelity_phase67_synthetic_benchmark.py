from __future__ import annotations

import pytest
import torch

from bochan.models.multifidelity.experiment import (
    SyntheticBenchmarkConfig,
    _strategy_configs,
    _target_metric_trace,
    generate_initial_data,
    run_synthetic_strategy,
)
from bochan.models.multifidelity.synthetic import (
    SyntheticMultiFidelityProblem,
    augmented_branin_problem,
    augmented_hartmann_problem,
    momf_branin_currin_problem,
)


def test_augmented_branin_problem_contract_and_affine_cost():
    problem = augmented_branin_problem()
    X = torch.tensor(
        [[0.0, 7.5, 0.5], [1.0, 8.0, 1.0]],
        dtype=torch.double,
    )

    Y = problem.evaluate(X)
    cost = problem.cost(X)

    assert problem.dim == 3
    assert problem.num_objectives == 1
    assert problem.target_fidelity == 1.0
    assert Y.shape == (2, 1)
    assert torch.allclose(cost, torch.tensor([0.75, 1.25], dtype=torch.double))
    assert problem.cost_config == {
        "kind": "affine",
        "fixed_cost": 0.25,
        "fidelity_weights": {2: 1.0},
    }


def test_augmented_hartmann_and_momf_problem_shapes():
    hartmann = augmented_hartmann_problem()
    X_h = torch.full((2, 7), 0.5, dtype=torch.double)
    X_h[0, -1] = 0.75
    X_h[1, -1] = 1.0
    assert hartmann.evaluate(X_h).shape == (2, 1)

    momf = momf_branin_currin_problem()
    X_m = momf.bounds.mean(dim=0, keepdim=True)
    X_m[:, momf.fidelity_feature] = 1.0
    assert momf.evaluate(X_m).shape == (1, 2)
    assert momf.cost(X_m).item() > 1.0
    assert momf.ref_point is not None
    assert momf.ref_point.shape == (2,)


def test_generate_initial_data_is_reproducible_and_seeds_target_fidelity_first():
    problem = augmented_branin_problem()
    first = generate_initial_data(problem, n=5, seed=17)
    second = generate_initial_data(problem, n=5, seed=17)

    assert torch.equal(first, second)
    assert first[0, problem.fidelity_feature].item() == pytest.approx(1.0)
    assert set(first[:, problem.fidelity_feature].tolist()).issubset(
        set(problem.fidelity_values)
    )

    high_only = generate_initial_data(
        problem,
        n=4,
        seed=17,
        high_fidelity_only=True,
    )
    assert torch.all(high_only[:, problem.fidelity_feature] == 1.0)


def test_target_metric_ignores_low_fidelity_objective_values():
    problem = SyntheticMultiFidelityProblem(
        name="toy",
        bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        fidelity_feature=1,
        fidelity_values=(0.5, 1.0),
        target_fidelity=1.0,
        num_objectives=1,
        evaluate_fn=lambda X: X[..., :1],
        cost_fn=lambda X: 0.25 + X[..., 1],
    )
    X = torch.tensor(
        [[0.1, 1.0], [0.9, 0.5], [0.4, 1.0]],
        dtype=torch.double,
    )
    Y = torch.tensor([[1.0], [100.0], [2.0]], dtype=torch.double)

    metric, name = _target_metric_trace(problem, X, Y)

    assert name == "best_target_objective"
    assert torch.equal(metric, torch.tensor([1.0, 1.0, 2.0], dtype=torch.double))


def test_strategy_configs_fix_or_enumerate_fidelity_as_expected():
    problem = augmented_branin_problem()
    config = SyntheticBenchmarkConfig(
        n_initial=3,
        budget=4.0,
        max_steps=1,
        num_restarts=1,
        raw_samples=8,
        fit_maxiter=1,
        num_fantasies=2,
        candidate_set_size=16,
        num_mv_samples=2,
    )

    _, high_opt = _strategy_configs(problem, "high_fidelity", config)
    assert high_opt.fixed_features == {problem.fidelity_feature: 1.0}
    assert high_opt.fidelity_values is None

    mfkg_acq, mfkg_opt = _strategy_configs(problem, "mfkg", config)
    assert mfkg_acq.name == "mfkg"
    assert mfkg_opt.fidelity_values == list(problem.fidelity_values)
    assert mfkg_acq.acqf_kwargs["cost_config"]["kind"] == "affine"

    mfmes_acq, mfmes_opt = _strategy_configs(problem, "mfmes", config)
    assert mfmes_acq.name == "mfmes"
    assert mfmes_opt.fidelity_values == list(problem.fidelity_values)


def test_strategy_family_validation_is_explicit():
    single = augmented_branin_problem()
    multi = momf_branin_currin_problem()

    with pytest.raises(ValueError, match="not valid"):
        _strategy_configs(single, "momf", SyntheticBenchmarkConfig())

    # Public runner validates strategy/problem family before fitting.
    with pytest.raises(ValueError, match="not valid"):
        run_synthetic_strategy(
            multi,
            "mfkg",
            config=SyntheticBenchmarkConfig(
                n_initial=1,
                budget=1.0,
                max_steps=1,
                skip_fit=True,
            ),
        )


def test_runner_smoke_returns_cost_aligned_target_trace_without_bo_step():
    problem = augmented_branin_problem()
    # Three mixed initial observations cost exactly 1.25 + 0.75 + 1.0 = 3.0,
    # so this exercises real bochan model construction while intentionally
    # skipping acquisition optimization in the CI smoke test.
    config = SyntheticBenchmarkConfig(
        n_initial=3,
        budget=3.0,
        max_steps=1,
        num_restarts=1,
        raw_samples=8,
        fit_maxiter=1,
        num_fantasies=2,
        candidate_set_size=16,
        num_mv_samples=2,
        skip_fit=True,
    )

    result = run_synthetic_strategy(problem, "mfkg", seed=3, config=config)

    assert result.X.shape == (3, 3)
    assert result.Y.shape == (3, 1)
    assert result.costs.shape == result.cumulative_cost.shape == result.metric.shape == (3,)
    assert result.target_mask[0]
    assert result.metric_name == "best_target_objective"
    assert result.cumulative_cost[-1].item() == pytest.approx(3.0)
    rows = result.rows()
    assert len(rows) == 3
    assert rows[0]["problem"] == "augmented_branin"
    assert rows[0]["strategy"] == "mfkg"
