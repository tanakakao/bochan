"""Per-fidelity score diagnostics for multi-fidelity knowledge gradient.

The diagnostics compare per-fidelity MFKG maxima against the production mixed
fidelity choice while reusing the exact same acquisition-function object. This
avoids rebuilding qMFKG (and its fantasy sampler) independently for each
fidelity, which would make the resulting acquisition values non-comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from bochan.api import optimize_candidates

from .benchmark import cumulative_cost
from .diagnostics import _mfkg_configs
from .experiment import (
    SyntheticBenchmarkConfig,
    SyntheticBenchmarkRun,
    _target_metric_trace,
    generate_initial_data,
)
from .synthetic import SyntheticMultiFidelityProblem


@dataclass(frozen=True)
class MFKGFidelityScoreRun:
    """One cost-aware MFKG run with per-fidelity acquisition score records."""

    run: SyntheticBenchmarkRun
    score_rows: tuple[dict[str, float | int | bool], ...]

    def rows(self) -> list[dict[str, float | int | bool]]:
        """Return a mutable copy of the per-fidelity score records."""

        return [dict(row) for row in self.score_rows]


def _fixed_fidelity_opt_config(
    problem: SyntheticMultiFidelityProblem,
    config: SyntheticBenchmarkConfig,
    fidelity: float,
) -> Any:
    """Build an optimization config restricted to one allowed fidelity value."""

    from bochan.api.configs import OptimizeConfig

    return OptimizeConfig(
        q=1,
        num_restarts=config.num_restarts,
        raw_samples=config.raw_samples,
        ensure_unique_candidates=False,
        fidelity_values=[float(fidelity)],
    )


def _scalar_acquisition_value(value: Any, *, like: Tensor) -> float:
    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(-1)
    return float(tensor.max().item())


def _as_candidate(value: Any, *, like: Tensor) -> Tensor:
    candidate = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if candidate.ndim == 1:
        candidate = candidate.unsqueeze(0)
    return candidate


def run_mfkg_fidelity_score_diagnostic(
    problem: SyntheticMultiFidelityProblem,
    *,
    seed: int = 0,
    config: SyntheticBenchmarkConfig | None = None,
) -> MFKGFidelityScoreRun:
    """Run cost-aware MFKG and score every allowed fidelity before each step.

    For each BO iteration, unweighted and cost-aware qMFKG are each built once.
    Their per-fidelity maxima are then obtained by repeatedly optimizing the same
    acquisition object with one fidelity fixed at a time. The normal cost-aware
    mixed-fidelity optimization reuses that same cost-aware acquisition object.

    The RNG state is replayed for the raw fixed-fidelity sweep, the cost-aware
    fixed-fidelity sweep, and the production mixed-fidelity optimization. This
    keeps optimizer initialization aligned while avoiding acquisition rebuilds.
    """

    if problem.num_objectives != 1:
        raise ValueError(
            "MFKG fidelity-score diagnostics currently require a single-objective problem."
        )
    if problem.cost_config is None:
        raise ValueError("MFKG fidelity-score diagnostics require problem.cost_config.")

    from bochan.api import BayesianOptimizer
    from bochan.api.configs import DataContext, FitConfig, ModelConfig

    config = config or SyntheticBenchmarkConfig()
    torch.manual_seed(int(seed))

    X = generate_initial_data(problem, n=config.n_initial, seed=int(seed))
    Y = problem.evaluate(X)
    costs = problem.cost(X)

    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression",
            model_type="multifidelity_gp",
            model_kwargs={
                "fidelity_features": [problem.fidelity_feature],
                "target_fidelities": {
                    problem.fidelity_feature: problem.target_fidelity,
                },
            },
        ),
        FitConfig(maxiter=config.fit_maxiter, skip_fit=config.skip_fit),
        bounds=problem.bounds,
        data_context=DataContext(bounds=problem.bounds),
    )
    optimizer.fit(X, Y)

    aware_acq_config, aware_opt_config = _mfkg_configs(
        problem,
        config,
        cost_aware=True,
    )
    raw_acq_config, _ = _mfkg_configs(problem, config, cost_aware=False)

    records: list[dict[str, float | int | bool]] = []
    total_cost = float(costs.sum().item())

    for iteration in range(config.max_steps):
        if total_cost >= float(config.budget):
            break

        # Build each qMFKG object exactly once for the current fitted surrogate.
        # All fidelity-specific scores and the final mixed choice therefore share
        # the same fantasies / current-value state within each acquisition type.
        raw_acqf = optimizer.acquisition(raw_acq_config)
        aware_acqf = optimizer.acquisition(aware_acq_config)
        step_rng_state = torch.random.get_rng_state()

        raw_results: dict[float, tuple[Tensor, float]] = {}
        torch.random.set_rng_state(step_rng_state)
        for fidelity in problem.fidelity_values:
            opt_config = _fixed_fidelity_opt_config(problem, config, float(fidelity))
            candidate, value = optimize_candidates(
                acqf=raw_acqf,
                bounds=problem.bounds,
                config=opt_config,
            )
            candidate = _as_candidate(candidate, like=X)
            raw_results[float(fidelity)] = (
                candidate,
                _scalar_acquisition_value(value, like=X),
            )

        aware_results: dict[float, tuple[Tensor, float]] = {}
        torch.random.set_rng_state(step_rng_state)
        for fidelity in problem.fidelity_values:
            opt_config = _fixed_fidelity_opt_config(problem, config, float(fidelity))
            candidate, value = optimize_candidates(
                acqf=aware_acqf,
                bounds=problem.bounds,
                config=opt_config,
            )
            candidate = _as_candidate(candidate, like=X)
            aware_results[float(fidelity)] = (
                candidate,
                _scalar_acquisition_value(value, like=X),
            )

        # Replay the same optimizer RNG sequence used by the aware per-fidelity
        # sweep. optimize_acqf_mixed evaluates the same fidelity assignments and
        # should therefore select the best corresponding cost-aware optimum.
        torch.random.set_rng_state(step_rng_state)
        candidate, selected_value = optimize_candidates(
            acqf=aware_acqf,
            bounds=problem.bounds,
            config=aware_opt_config,
        )
        candidate = _as_candidate(candidate, like=X)

        new_cost = problem.cost(candidate)
        candidate_cost = float(new_cost.sum().item())
        if total_cost + candidate_cost > float(config.budget):
            break

        selected_fidelity = float(candidate[0, problem.fidelity_feature].item())
        selected_acquisition_value = _scalar_acquisition_value(selected_value, like=X)

        for fidelity in problem.fidelity_values:
            level = float(fidelity)
            raw_candidate, raw_value = raw_results[level]
            aware_candidate, aware_value = aware_results[level]
            records.append(
                {
                    "seed": int(seed),
                    "iteration": int(iteration),
                    "fidelity": level,
                    "raw_kg": raw_value,
                    "cost_aware_kg": aware_value,
                    "raw_candidate_cost": float(problem.cost(raw_candidate).sum().item()),
                    "aware_candidate_cost": float(problem.cost(aware_candidate).sum().item()),
                    "selected": bool(abs(level - selected_fidelity) <= 1e-8),
                    "selected_fidelity": selected_fidelity,
                    "selected_acquisition_value": selected_acquisition_value,
                }
            )

        new_Y = problem.evaluate(candidate)
        optimizer.tell(candidate, new_Y, refit=True)
        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, new_Y], dim=0)
        costs = torch.cat([costs, new_cost], dim=0)
        total_cost += candidate_cost

    metric, metric_name = _target_metric_trace(problem, X, Y)
    run = SyntheticBenchmarkRun(
        problem=problem.name,
        strategy="mfkg_fidelity_score_diagnostic",
        seed=int(seed),
        fidelity_feature=problem.fidelity_feature,
        X=X,
        Y=Y,
        costs=costs,
        target_mask=problem.is_target_fidelity(X),
        cumulative_cost=cumulative_cost(costs),
        metric=metric,
        metric_name=metric_name,
        initial_count=config.n_initial,
    )
    return MFKGFidelityScoreRun(run=run, score_rows=tuple(records))


__all__ = [
    "MFKGFidelityScoreRun",
    "run_mfkg_fidelity_score_diagnostic",
]
