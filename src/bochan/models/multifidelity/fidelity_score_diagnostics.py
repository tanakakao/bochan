"""Per-fidelity score diagnostics for multi-fidelity knowledge gradient.

This module measures the best MFKG value obtainable at each allowed fidelity
before the production optimizer chooses its next point. Both unweighted KG and
cost-aware KG are recorded against the same fitted surrogate state, making it
possible to distinguish a surrogate / information-gain preference from a
cost-normalization preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

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


def run_mfkg_fidelity_score_diagnostic(
    problem: SyntheticMultiFidelityProblem,
    *,
    seed: int = 0,
    config: SyntheticBenchmarkConfig | None = None,
) -> MFKGFidelityScoreRun:
    """Run cost-aware MFKG and score every allowed fidelity before each step.

    For every BO iteration, the function optimizes both unweighted MFKG and
    cost-aware MFKG while fixing the fidelity to each value in
    ``problem.fidelity_values``. The normal cost-aware MFKG optimization over all
    allowed fidelities is then executed and that candidate is evaluated.
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

    aware_acq, aware_opt = _mfkg_configs(problem, config, cost_aware=True)
    raw_acq, _ = _mfkg_configs(problem, config, cost_aware=False)

    records: list[dict[str, float | int | bool]] = []
    total_cost = float(costs.sum().item())

    for iteration in range(config.max_steps):
        if total_cost >= float(config.budget):
            break

        step_records: list[dict[str, float | int | bool]] = []
        for fidelity in problem.fidelity_values:
            opt_config = _fixed_fidelity_opt_config(problem, config, float(fidelity))

            # Reuse the same RNG state for the raw and cost-aware optimization so
            # their KG fantasies / raw samples are paired as closely as possible.
            rng_state = torch.random.get_rng_state()
            raw_candidate, raw_value = optimizer.ask(
                acq_config=raw_acq,
                opt_config=opt_config,
            )
            torch.random.set_rng_state(rng_state)
            aware_candidate, aware_value = optimizer.ask(
                acq_config=aware_acq,
                opt_config=opt_config,
            )

            raw_candidate = torch.as_tensor(
                raw_candidate,
                dtype=X.dtype,
                device=X.device,
            )
            aware_candidate = torch.as_tensor(
                aware_candidate,
                dtype=X.dtype,
                device=X.device,
            )
            if raw_candidate.ndim == 1:
                raw_candidate = raw_candidate.unsqueeze(0)
            if aware_candidate.ndim == 1:
                aware_candidate = aware_candidate.unsqueeze(0)

            step_records.append(
                {
                    "seed": int(seed),
                    "iteration": int(iteration),
                    "fidelity": float(fidelity),
                    "raw_kg": _scalar_acquisition_value(raw_value, like=X),
                    "cost_aware_kg": _scalar_acquisition_value(aware_value, like=X),
                    "raw_candidate_cost": float(problem.cost(raw_candidate).sum().item()),
                    "aware_candidate_cost": float(problem.cost(aware_candidate).sum().item()),
                    "selected": False,
                }
            )

        candidate, selected_value = optimizer.ask(
            acq_config=aware_acq,
            opt_config=aware_opt,
        )
        candidate = torch.as_tensor(candidate, dtype=X.dtype, device=X.device)
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)

        new_cost = problem.cost(candidate)
        candidate_cost = float(new_cost.sum().item())
        if total_cost + candidate_cost > float(config.budget):
            break

        selected_fidelity = float(candidate[0, problem.fidelity_feature].item())
        for row in step_records:
            row["selected"] = bool(
                abs(float(row["fidelity"]) - selected_fidelity) <= 1e-8
            )
            row["selected_fidelity"] = selected_fidelity
            row["selected_acquisition_value"] = _scalar_acquisition_value(
                selected_value,
                like=X,
            )
            records.append(row)

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
