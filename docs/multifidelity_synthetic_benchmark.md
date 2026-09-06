# Multi-Fidelity Synthetic Benchmark Suite

Phase 67 adds executable synthetic experiments on top of the Multi-Fidelity v2 stack completed in Phases 43–66.

## Purpose

The benchmark compares strategies on **cumulative evaluation cost**, not iteration count. A cheap low-fidelity query therefore contributes less to the x-axis than a target-fidelity query.

Optimization quality is evaluated only at the target fidelity:

- single objective: best observed target-fidelity objective vs cumulative cost
- multi objective: observed target-fidelity hypervolume vs cumulative cost

Low-fidelity outcomes are used by the surrogate and acquisition function, but they are not incorrectly counted as target-fidelity objective quality.

## Included synthetic problems

### Augmented Branin

`augmented_branin_problem()` wraps BoTorch's `AugmentedBranin(negate=True)`.

- design variables: 2
- fidelity variable: last column
- fidelity levels: `0.5, 0.75, 1.0`
- target fidelity: `1.0`
- default cost: `0.25 + fidelity`

Recommended comparison:

- `high_fidelity`
- `mfkg`
- `mfmes`

### Augmented Hartmann

`augmented_hartmann_problem()` wraps BoTorch's `AugmentedHartmann(negate=True)`.

- design variables: 6
- fidelity variable: last column
- fidelity levels: `0.5, 0.75, 1.0`
- target fidelity: `1.0`
- default cost: `0.25 + fidelity`

Recommended comparison:

- `high_fidelity`
- `mfkg`
- `mfmes`

### Multi-fidelity Branin-Currin

`momf_branin_currin_problem()` wraps BoTorch's `MOMFBraninCurrin(negate=True)`.

- objectives: 2
- fidelity variable: last column
- fidelity levels: `0.5, 0.75, 1.0`
- target fidelity: `1.0`
- cost: `exp(4.8 * fidelity)`
- reference point: `(0, 0)`

Recommended comparison:

- `high_fidelity`
- `mfhvkg`
- `momf`

## Python API

```python
from bochan.models.multifidelity.experiment import (
    SyntheticBenchmarkConfig,
    run_synthetic_benchmark,
)
from bochan.models.multifidelity.synthetic import augmented_branin_problem

problem = augmented_branin_problem()
config = SyntheticBenchmarkConfig(
    n_initial=6,
    budget=15.0,
    max_steps=12,
    num_restarts=4,
    raw_samples=64,
    fit_maxiter=50,
)

runs = run_synthetic_benchmark(
    problem,
    ["high_fidelity", "mfkg", "mfmes"],
    seeds=list(range(10)),
    config=config,
)

rows = [row for run in runs for row in run.rows()]
```

Each row contains:

- problem
- strategy
- seed
- step
- initialization flag
- target-fidelity flag
- evaluation cost
- cumulative cost
- metric name
- metric value
- selected fidelity

This long-form representation can be written directly to CSV and then aggregated by seed.

## Command-line example

```bash
python examples/multifidelity_synthetic_benchmark.py \
  --problem branin \
  --strategies high_fidelity mfkg mfmes \
  --seeds 0 1 2 3 4 \
  --budget 15 \
  --output results/branin_mf.csv
```

For the multi-objective experiment:

```bash
python examples/multifidelity_synthetic_benchmark.py \
  --problem momf_branin_currin \
  --strategies high_fidelity mfhvkg momf \
  --seeds 0 1 2 \
  --budget 500 \
  --output results/branin_currin_mf.csv
```

MF-HVKG is substantially more expensive to optimize than MOMF, so begin with a small number of seeds and increase after the configuration is confirmed.

## Recommended experiment protocol

For a reportable comparison, use at least 10 independent seeds initially and preferably 20 or more for the final result. Keep the following identical across strategies within a problem:

1. random/Sobol seed
2. cost model
3. target fidelity
4. fidelity choices
5. cumulative cost budget
6. optimizer restart/raw-sample settings where applicable

The high-fidelity baseline receives only target-fidelity observations. Multi-fidelity strategies receive an initialization spanning the configured fidelity levels, with the first observation forced to target fidelity so the target-only performance trace is defined from the beginning.

## Interpretation

The main comparison should answer questions such as:

- At the same cumulative cost, does MFKG or MF-MES reach a better target-fidelity incumbent than high-fidelity-only BO?
- Does MF-HVKG achieve higher target-fidelity hypervolume per unit cost than MOMF?
- Which fidelity levels are selected as the optimization progresses?
- How stable are those conclusions across seeds?

The Phase 66 benchmark helpers remain useful for post-processing arbitrary experiment logs. The Phase 67 suite adds the actual synthetic objectives, BO loop, strategy wiring, and reproducible experiment output.
