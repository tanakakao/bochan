"""Diagnostics for multi-fidelity knowledge-gradient benchmark behavior.

Phase 68 keeps the production MFKG implementation unchanged and adds a controlled
benchmark path that compares the same qMultiFidelityKnowledgeGradient acquisition
with and without cost-aware utility.  This makes it possible to distinguish a
surrogate / target-projection problem from a cost-normalization effect when cheap
fidelities are selected disproportionately often.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .benchmark import cumulative_cost
from .experiment import (
    SyntheticBenchmarkConfig,
    SyntheticBenchmarkRun,
    _target_metric_trace,
    generate_initial_data,
)
from .synthetic import SyntheticMultiFidelityProblem


@dataclass(frozen=True)
class MFKGDiagnosticRun:
    """One MFKG trajectory plus the acquisition value chosen at each BO step."""

    run: SyntheticBenchmarkRun
    acquisition_values: Tensor
    cost_aware: bool

    def __post_init__(self) -> None:
        values = torch.as_tensor(
            self.acquisition_values,
            dtype=self.run.X.dtype,
            device=self.run.X.device,
        ).reshape(-1)
        if values.shape != (self.run.X.shape[0],):
            raise ValueError("acquisition_values must contain one value per observation.")
        object.__setattr__(self, "acquisition_values", values)

    def rows(self) -> list[dict[str, float | int | str | bool]]:
        """Return CSV-friendly rows with acquisition diagnostics attached."""

        rows = self.run.rows()
        for index, row in enumerate(rows):
            value = self.acquisition_values[index]
            row["cost_aware"] = bool(self.cost_aware)
            row["acquisition_value"] = (
                float(value.item()) if bool(torch.isfinite(value)) else float("nan")
            )
        return rows

    def fidelity_counts(self, *, include_initial: bool = False) -> dict[float, int]:
        """Count selected fidelity levels, optionally including initialization."""

        start = 0 if include_initial else self.run.initial_count
        values = self.run.X[start:, self.run.fidelity_feature]
        return {
            float(level): int((values == values.new_tensor(level)).sum().item())
            for level in sorted(set(float(v) for v in self.run.X[:, self.run.fidelity_feature].tolist()))
        }


def _mfkg_configs(
    problem: SyntheticMultiFidelityProblem,
    config: SyntheticBenchmarkConfig,
    *,
    cost_aware: bool,
) -> tuple[Any, Any]:
    """Build identical MFKG configs except for the cost-aware utility."""

    from bochan.api.configs import AcquisitionConfig, OptimizeConfig

    kwargs: dict[str, Any] = {
        "num_fantasies": config.num_fantasies,
        "current_value_num_restarts": config.num_restarts,
        "current_value_raw_samples": config.raw_samples,
    }
    if cost_aware:
        if problem.cost_config is None:
            raise ValueError("cost_aware=True requires problem.cost_config.")
        kwargs["cost_config"] = dict(problem.cost_config)

    return (
        AcquisitionConfig(name="mfkg", acqf_kwargs=kwargs),
        OptimizeConfig(
            q=1,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
            ensure_unique_candidates=False,
            fidelity_values=list(problem.fidelity_values),
        ),
    )


def run_mfkg_diagnostic(
    problem: SyntheticMultiFidelityProblem,
    *,
    seed: int = 0,
    config: SyntheticBenchmarkConfig | None = None,
    cost_aware: bool = True,
) -> MFKGDiagnosticRun:
    """Run MFKG while recording selected acquisition values and fidelities."""

    if problem.num_objectives != 1:
        raise ValueError("MFKG diagnostics currently require a single-objective problem.")

    from bochan.api import BayesianOptimizer
    from bochan.api.configs import DataContext, FitConfig, ModelConfig

    config = config or SyntheticBenchmarkConfig()
    torch.manual_seed(int(seed))

    X = generate_initial_data(problem, n=config.n_initial, seed=int(seed))
    Y = problem.evaluate(X)
    costs = problem.cost(X)
    acquisition_values = [float("nan")] * X.shape[0]

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

    acq_config, opt_config = _mfkg_configs(problem, config, cost_aware=cost_aware)
    total_cost = float(costs.sum().item())
    for _ in range(config.max_steps):
        if total_cost >= float(config.budget):
            break
        candidate, acq_value = optimizer.ask(acq_config=acq_config, opt_config=opt_config)
        candidate = torch.as_tensor(candidate, dtype=X.dtype, device=X.device)
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)

        new_cost = problem.cost(candidate)
        candidate_cost = float(new_cost.sum().item())
        if total_cost + candidate_cost > float(config.budget):
            break

        value_tensor = torch.as_tensor(acq_value, dtype=X.dtype, device=X.device).reshape(-1)
        acquisition_values.append(float(value_tensor.max().item()))
        new_Y = problem.evaluate(candidate)
        optimizer.tell(candidate, new_Y, refit=True)
        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, new_Y], dim=0)
        costs = torch.cat([costs, new_cost], dim=0)
        total_cost += candidate_cost

    metric, metric_name = _target_metric_trace(problem, X, Y)
    run = SyntheticBenchmarkRun(
        problem=problem.name,
        strategy="mfkg" if cost_aware else "mfkg_unweighted",
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
    return MFKGDiagnosticRun(
        run=run,
        acquisition_values=torch.tensor(
            acquisition_values,
            dtype=X.dtype,
            device=X.device,
        ),
        cost_aware=bool(cost_aware),
    )


def compare_mfkg_cost_awareness(
    problem: SyntheticMultiFidelityProblem,
    *,
    seed: int = 0,
    config: SyntheticBenchmarkConfig | None = None,
) -> tuple[MFKGDiagnosticRun, MFKGDiagnosticRun]:
    """Run cost-aware and unweighted MFKG with identical seed/configuration."""

    config = config or SyntheticBenchmarkConfig()
    aware = run_mfkg_diagnostic(problem, seed=seed, config=config, cost_aware=True)
    unweighted = run_mfkg_diagnostic(problem, seed=seed, config=config, cost_aware=False)
    return aware, unweighted


__all__ = [
    "MFKGDiagnosticRun",
    "compare_mfkg_cost_awareness",
    "run_mfkg_diagnostic",
]
