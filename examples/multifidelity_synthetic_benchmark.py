"""Run cost-normalized synthetic multi-fidelity benchmarks.

Examples
--------
Single objective::

    python examples/multifidelity_synthetic_benchmark.py --problem branin \
        --strategies high_fidelity mfkg mfmes --seeds 0 1 2 --budget 15

Multi objective::

    python examples/multifidelity_synthetic_benchmark.py --problem momf_branin_currin \
        --strategies high_fidelity mfhvkg momf --seeds 0 1 2 --budget 500
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bochan.models.multifidelity.experiment import (
    SyntheticBenchmarkConfig,
    run_synthetic_benchmark,
)
from bochan.models.multifidelity.synthetic import (
    augmented_branin_problem,
    augmented_hartmann_problem,
    momf_branin_currin_problem,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        choices=("branin", "hartmann", "momf_branin_currin"),
        default="branin",
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--n-initial", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=64)
    parser.add_argument("--fit-maxiter", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("multifidelity_benchmark.csv"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.problem == "branin":
        problem = augmented_branin_problem()
        default_strategies = ["high_fidelity", "mfkg", "mfmes"]
        default_budget = 15.0
    elif args.problem == "hartmann":
        problem = augmented_hartmann_problem()
        default_strategies = ["high_fidelity", "mfkg", "mfmes"]
        default_budget = 15.0
    else:
        problem = momf_branin_currin_problem()
        default_strategies = ["high_fidelity", "mfhvkg", "momf"]
        default_budget = 500.0

    config = SyntheticBenchmarkConfig(
        n_initial=args.n_initial,
        budget=default_budget if args.budget is None else args.budget,
        max_steps=args.max_steps,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        fit_maxiter=args.fit_maxiter,
    )
    runs = run_synthetic_benchmark(
        problem,
        args.strategies or default_strategies,
        seeds=args.seeds,
        config=config,
    )
    rows = [row for run in runs for row in run.rows()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}")
    for run in runs:
        print(
            run.problem,
            run.strategy,
            f"seed={run.seed}",
            f"cost={run.cumulative_cost[-1].item():.3f}",
            f"{run.metric_name}={run.metric[-1].item():.6g}",
        )


if __name__ == "__main__":
    main()
