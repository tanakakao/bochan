"""Executable synthetic experiments for the multi-fidelity v2 stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from .benchmark import cumulative_cost
from .synthetic import SyntheticMultiFidelityProblem

BenchmarkStrategy = Literal[
    "high_fidelity",
    "mfkg",
    "mfmes",
    "mfhvkg",
    "momf",
]


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Controls one repeated synthetic benchmark run."""

    n_initial: int = 6
    budget: float = 12.0
    max_steps: int = 12
    num_restarts: int = 4
    raw_samples: int = 64
    fit_maxiter: int = 50
    num_fantasies: int = 8
    num_pareto: int = 8
    candidate_set_size: int = 128
    num_mv_samples: int = 8
    skip_fit: bool = False
    correlated_outputs: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "n_initial",
            "max_steps",
            "num_restarts",
            "raw_samples",
            "fit_maxiter",
            "num_fantasies",
            "num_pareto",
            "candidate_set_size",
            "num_mv_samples",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if float(self.budget) <= 0:
            raise ValueError("budget must be positive.")


@dataclass(frozen=True)
class SyntheticBenchmarkRun:
    """One strategy/seed trajectory on a synthetic MF problem."""

    problem: str
    strategy: str
    seed: int
    X: Tensor
    Y: Tensor
    costs: Tensor
    target_mask: Tensor
    cumulative_cost: Tensor
    metric: Tensor
    metric_name: str
    initial_count: int

    def rows(self) -> list[dict[str, float | int | str | bool]]:
        """Return a JSON/CSV-friendly long-form representation."""

        rows: list[dict[str, float | int | str | bool]] = []
        for index in range(self.X.shape[0]):
            rows.append(
                {
                    "problem": self.problem,
                    "strategy": self.strategy,
                    "seed": int(self.seed),
                    "step": int(index),
                    "is_initial": bool(index < self.initial_count),
                    "is_target_fidelity": bool(self.target_mask[index]),
                    "evaluation_cost": float(self.costs[index]),
                    "cumulative_cost": float(self.cumulative_cost[index]),
                    "metric_name": self.metric_name,
                    "metric": float(self.metric[index]),
                    "fidelity": float(self.X[index, -1]),
                }
            )
        return rows


def _normalize_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower().replace("-", "_")
    aliases = {
        "high": "high_fidelity",
        "highfidelity": "high_fidelity",
        "hf": "high_fidelity",
        "qmfkg": "mfkg",
        "qmfmes": "mfmes",
        "qmfhvkg": "mfhvkg",
        "qmomf": "momf",
    }
    return aliases.get(normalized, normalized)


def _validate_strategy(problem: SyntheticMultiFidelityProblem, strategy: str) -> str:
    strategy = _normalize_strategy(strategy)
    single = {"high_fidelity", "mfkg", "mfmes"}
    multi = {"high_fidelity", "mfhvkg", "momf"}
    allowed = single if problem.num_objectives == 1 else multi
    if strategy not in allowed:
        raise ValueError(
            f"Strategy {strategy!r} is not valid for a {problem.num_objectives}-objective problem; "
            f"expected one of {sorted(allowed)}."
        )
    return strategy


def generate_initial_data(
    problem: SyntheticMultiFidelityProblem,
    *,
    n: int,
    seed: int,
    high_fidelity_only: bool = False,
) -> Tensor:
    """Generate deterministic Sobol initialization with explicit fidelity levels."""

    n = int(n)
    if n < 1:
        raise ValueError("n must be positive.")
    X = draw_sobol_samples(
        bounds=problem.bounds,
        n=n,
        q=1,
        seed=int(seed),
    ).squeeze(-2)
    feature = problem.fidelity_feature
    if high_fidelity_only:
        X[:, feature] = problem.target_fidelity
        return X

    # Always seed at least one target-fidelity observation so target-only quality
    # traces remain defined from the first experiment step onward.
    values = (problem.target_fidelity,) + tuple(
        value for value in problem.fidelity_values if value != problem.target_fidelity
    )
    assigned = [values[index % len(values)] for index in range(n)]
    X[:, feature] = X.new_tensor(assigned)
    return X


def _target_metric_trace(
    problem: SyntheticMultiFidelityProblem,
    X: Tensor,
    Y: Tensor,
) -> tuple[Tensor, str]:
    """Evaluate optimization quality using target-fidelity observations only."""

    target_mask = problem.is_target_fidelity(X)
    if not bool(target_mask.any()):
        raise ValueError("Benchmark trajectories require at least one target-fidelity observation.")

    if problem.num_objectives == 1:
        best: Tensor | None = None
        values: list[Tensor] = []
        for index in range(X.shape[0]):
            if bool(target_mask[index]):
                value = Y[index, 0]
                best = value if best is None else torch.maximum(best, value)
            if best is None:
                raise RuntimeError("The first benchmark observation must include target fidelity.")
            values.append(best)
        return torch.stack(values), "best_target_objective"

    if problem.ref_point is None:
        raise ValueError("Multi-objective synthetic benchmarks require a ref_point.")
    hv = Hypervolume(ref_point=problem.ref_point.to(dtype=Y.dtype, device=Y.device))
    values = []
    for index in range(X.shape[0]):
        target_Y = Y[: index + 1][target_mask[: index + 1]]
        raw = hv.compute(target_Y)
        values.append(Y.new_tensor(float(raw)))
    return torch.stack(values), "target_hypervolume"


def _strategy_configs(
    problem: SyntheticMultiFidelityProblem,
    strategy: str,
    config: SyntheticBenchmarkConfig,
) -> tuple[Any, Any]:
    from bochan.api.configs import AcquisitionConfig, OptimizeConfig

    common_opt = {
        "q": 1,
        "num_restarts": config.num_restarts,
        "raw_samples": config.raw_samples,
        "ensure_unique_candidates": False,
    }
    feature = problem.fidelity_feature
    target = problem.target_fidelity

    if strategy == "high_fidelity":
        name = "qlogei" if problem.num_objectives == 1 else "qlognehvi"
        return (
            AcquisitionConfig(name=name),
            OptimizeConfig(
                **common_opt,
                fixed_features={feature: target},
            ),
        )

    cost_config = dict(problem.cost_config or {})
    if strategy == "mfkg":
        return (
            AcquisitionConfig(
                name="mfkg",
                acqf_kwargs={
                    "cost_config": cost_config,
                    "num_fantasies": config.num_fantasies,
                    "current_value_num_restarts": config.num_restarts,
                    "current_value_raw_samples": config.raw_samples,
                },
            ),
            OptimizeConfig(
                **common_opt,
                fidelity_values=list(problem.fidelity_values),
            ),
        )
    if strategy == "mfmes":
        return (
            AcquisitionConfig(
                name="mfmes",
                acqf_kwargs={
                    "cost_config": cost_config,
                    "candidate_set_size": config.candidate_set_size,
                    "num_mv_samples": config.num_mv_samples,
                },
            ),
            OptimizeConfig(
                **common_opt,
                fidelity_values=list(problem.fidelity_values),
            ),
        )
    if strategy == "mfhvkg":
        return (
            AcquisitionConfig(
                name="mfhvkg",
                acqf_kwargs={
                    "cost_config": cost_config,
                    "num_fantasies": config.num_fantasies,
                    "num_pareto": config.num_pareto,
                    "current_value_num_restarts": config.num_restarts,
                    "current_value_raw_samples": config.raw_samples,
                },
            ),
            OptimizeConfig(
                **common_opt,
                fidelity_values=list(problem.fidelity_values),
            ),
        )
    if strategy == "momf":
        return (
            AcquisitionConfig(
                name="momf",
                acqf_kwargs={"cost_config": cost_config},
            ),
            OptimizeConfig(
                **common_opt,
                fidelity_values=list(problem.fidelity_values),
            ),
        )
    raise ValueError(f"Unsupported benchmark strategy: {strategy!r}.")


def run_synthetic_strategy(
    problem: SyntheticMultiFidelityProblem,
    strategy: BenchmarkStrategy | str,
    *,
    seed: int = 0,
    config: SyntheticBenchmarkConfig | None = None,
) -> SyntheticBenchmarkRun:
    """Run one real bochan BO trajectory against a synthetic MF problem."""

    from bochan.api import BayesianOptimizer
    from bochan.api.configs import DataContext, FitConfig, ModelConfig

    config = config or SyntheticBenchmarkConfig()
    strategy = _validate_strategy(problem, str(strategy))
    torch.manual_seed(int(seed))

    X = generate_initial_data(
        problem,
        n=config.n_initial,
        seed=int(seed),
        high_fidelity_only=strategy == "high_fidelity",
    )
    Y = problem.evaluate(X)
    costs = problem.cost(X)

    model_kwargs: dict[str, Any] = {
        "fidelity_features": [problem.fidelity_feature],
        "target_fidelities": {problem.fidelity_feature: problem.target_fidelity},
    }
    if problem.num_objectives > 1 and config.correlated_outputs:
        model_kwargs["correlated_outputs"] = True

    optimizer = BayesianOptimizer(
        ModelConfig(
            task_type="regression" if problem.num_objectives == 1 else "multi_objective",
            model_type="multifidelity_gp",
            model_kwargs=model_kwargs,
        ),
        FitConfig(
            maxiter=config.fit_maxiter,
            skip_fit=config.skip_fit,
        ),
        bounds=problem.bounds,
        data_context=DataContext(
            bounds=problem.bounds,
            ref_point=problem.ref_point,
        ),
    )
    optimizer.fit(X, Y)

    acq_config, opt_config = _strategy_configs(problem, strategy, config)
    total_cost = float(costs.sum())
    for _ in range(config.max_steps):
        if total_cost >= float(config.budget):
            break
        candidate, _ = optimizer.ask(
            acq_config=acq_config,
            opt_config=opt_config,
        )
        candidate = torch.as_tensor(candidate, dtype=X.dtype, device=X.device)
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)
        new_cost = problem.cost(candidate)
        if total_cost + float(new_cost.sum()) > float(config.budget):
            break
        new_Y = problem.evaluate(candidate)
        optimizer.tell(candidate, new_Y, refit=True)
        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, new_Y], dim=0)
        costs = torch.cat([costs, new_cost], dim=0)
        total_cost += float(new_cost.sum())

    metric, metric_name = _target_metric_trace(problem, X, Y)
    return SyntheticBenchmarkRun(
        problem=problem.name,
        strategy=strategy,
        seed=int(seed),
        X=X,
        Y=Y,
        costs=costs,
        target_mask=problem.is_target_fidelity(X),
        cumulative_cost=cumulative_cost(costs),
        metric=metric,
        metric_name=metric_name,
        initial_count=config.n_initial,
    )


def run_synthetic_benchmark(
    problem: SyntheticMultiFidelityProblem,
    strategies: tuple[str, ...] | list[str],
    *,
    seeds: tuple[int, ...] | list[int] = (0,),
    config: SyntheticBenchmarkConfig | None = None,
) -> list[SyntheticBenchmarkRun]:
    """Run the requested strategies across seeds with identical settings."""

    if not strategies:
        raise ValueError("strategies must not be empty.")
    if not seeds:
        raise ValueError("seeds must not be empty.")
    results: list[SyntheticBenchmarkRun] = []
    for seed in seeds:
        for strategy in strategies:
            results.append(
                run_synthetic_strategy(
                    problem,
                    strategy,
                    seed=int(seed),
                    config=config,
                )
            )
    return results


__all__ = [
    "BenchmarkStrategy",
    "SyntheticBenchmarkConfig",
    "SyntheticBenchmarkRun",
    "generate_initial_data",
    "run_synthetic_benchmark",
    "run_synthetic_strategy",
]
