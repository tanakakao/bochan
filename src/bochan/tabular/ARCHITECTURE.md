# Tabular architecture

`bochan.tabular` is the DataFrame / numpy boundary around the tensor-oriented
`bochan.api` package. The public entry point is
`bochan.tabular.TabularBayesianOptimizer`.

## Package responsibilities

- `config/`: tabular configuration dataclasses and builders that normalize
  public keyword/mapping input into canonical API configs.
- `data/`: `TabularDataset`, DataFrame/numpy/tensor conversion, missing-value
  preparation, category encoding, column-index resolution, and bounds/config
  column resolution.
- `targets/`: categorical target metadata, class-label resolution, and ordinal
  rank semantics.
- `composition/`: composition-domain parsing, transforms, search spaces,
  bounds, variable totals, and composition constraints.
- `observation/`: partially observed / failed / pending experiment data and the
  adapter that attaches observation state to the core optimizer.
- `optimizer/`: the public facade and orchestration services for construction,
  fitting, candidate generation, prediction formatting, and diagnostics.

## Dependency direction

Lower-level domain packages must not import the public optimizer facade.
The intended dependency direction is:

```text
config ──────┐
data ────────┼──> optimizer ───> bochan.api
targets ─────┤
composition ─┤
observation ─┘
```

`composition` and `targets` may depend on lower-level data/config utilities
when necessary, but `data`, `targets`, `composition`, and `observation` must not
depend on `optimizer.core`.

## Extension model

Features are integrated by explicit component delegation rather than optimizer
subclass stacks or runtime method replacement. The facade owns components such
as `CompositionAdapter`, `ObservationAdapter`, `CandidateService`, and
`DiagnosticsService` and invokes them explicitly from `fit`, `candidate`,
`predict`, and diagnostic workflows.

Do not add compatibility optimizer subclasses, forwarding modules, monkey
patches, or import-time method replacement. When a module is relocated, migrate
its callers to the canonical path and remove the old module.
