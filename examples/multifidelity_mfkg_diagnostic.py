"""Compare cost-aware and unweighted MFKG on synthetic MF problems."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bochan.models.multifidelity.diagnostics import run_mfkg_diagnostic
from bochan.models.multifidelity.experiment import SyntheticBenchmarkConfig
from bochan.models.multifidelity.synthetic import (
    augmented_branin_problem,
    augmented_hartmann_problem,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("branin", "hartmann"), default="branin")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=float, default=15.0)
    parser.add_argument("--n-initial", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--num-restarts", type=int, default=4)
    parser.add_argument("--raw-samples", type=int, default=64)
    parser.add_argument("--fit-maxiter", type=int, default=50)
    parser.add_argument("--num-fantasies", type=int, default=8)
    parser.add_argument(
        "--fixed-cost",
        type=float,
        default=0.25,
        help="Affine fidelity-independent cost intercept for Branin/Hartmann.",
    )
    parser.add_argument("--output", type=Path, default=Path("mfkg_diagnostic.csv"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.fixed_cost <= 0:
        raise ValueError("--fixed-cost must be positive.")
    problem = (
        augmented_branin_problem(fixed_cost=args.fixed_cost)
        if args.problem == "branin"
        else augmented_hartmann_problem(fixed_cost=args.fixed_cost)
    )
    config = SyntheticBenchmarkConfig(
        n_initial=args.n_initial,
        budget=args.budget,
        max_steps=args.max_steps,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        fit_maxiter=args.fit_maxiter,
        num_fantasies=args.num_fantasies,
    )

    rows = []
    for seed in args.seeds:
        for cost_aware in (True, False):
            result = run_mfkg_diagnostic(
                problem,
                seed=seed,
                config=config,
                cost_aware=cost_aware,
            )
            for row in result.rows():
                row["fixed_cost"] = float(args.fixed_cost)
                rows.append(row)
            print(
                result.run.strategy,
                f"seed={seed}",
                f"fixed_cost={args.fixed_cost:g}",
                f"cost={result.run.cumulative_cost[-1].item():.3f}",
                f"metric={result.run.metric[-1].item():.6g}",
                f"fidelities={result.fidelity_counts()}",
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
