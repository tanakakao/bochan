"""Run Phase 72 source-to-target posterior transfer diagnostics.

Examples:
    python examples/multifidelity_fidelity_transfer_diagnostic.py \
        --problem branin --seeds 0 1 2 --output phase72-branin.csv

    python examples/multifidelity_fidelity_transfer_diagnostic.py \
        --problem hartmann --seeds 0 1 2 --output phase72-hartmann.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bochan.models.multifidelity import (
    augmented_branin_problem,
    augmented_hartmann_problem,
    run_fidelity_transfer_diagnostic,
)
from bochan.models.multifidelity.experiment import SyntheticBenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["branin", "hartmann"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-initial", type=int, default=6)
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--fit-maxiter", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = (
        augmented_branin_problem()
        if args.problem == "branin"
        else augmented_hartmann_problem()
    )
    config = SyntheticBenchmarkConfig(
        n_initial=args.n_initial,
        fit_maxiter=args.fit_maxiter,
    )

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        diagnostics = run_fidelity_transfer_diagnostic(
            problem,
            seed=seed,
            n_probe=args.n_probe,
            config=config,
        )
        rows.extend(item.row() for item in diagnostics)
        print(f"{problem.name} seed={seed} diagnostics={len(diagnostics)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No diagnostic rows were produced.")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
