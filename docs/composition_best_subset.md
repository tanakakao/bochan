# Composition best-subset search

`composition_sites` can opt a fixed-total composition into acquisition-aware element support search.

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature"],
    target_cols="property",
    composition_sites={
        "alloy": {
            "column": "formula",
            "elements": ["Al", "Ti", "V", "Cr", "Nb"],
            "representation": "ilr",
            "bounds": {
                "Al": [0.05, 0.80],
                "Ti": [0.00, 0.80],
                "V": [0.00, 0.80],
                "Cr": [0.00, 0.80],
                "Nb": [0.00, 0.80],
            },
            "required_components": ["Al"],
            "forbidden_components": ["Cr"],
            "min_components": 3,
            "max_components": 3,
            "support_selection": "best_subset",
            "best_subset_strategy": "auto",
        }
    },
)
```

## Raw support space and model space

Element support is always defined in raw composition space. A zero CLR, ALR, or ILR coordinate does **not** mean that an element is absent, so transformed coordinates are never sparsified directly.

For `representation="fractions"`, raw element fractions and model decision features coincide. For `"clr"`, `"alr"`, and `"ilr"`, bochan uses a two-space bridge:

1. support selection and composition constraints are optimized in synthetic raw-fraction decision variables;
2. every candidate is differentiably transformed into the fitted CLR / ALR / ILR coordinates before the surrogate and acquisition function are evaluated;
3. the exact raw fractions are retained for final formula/fraction output, including structural zeros.

The fitted surrogate is reused as-is. Best Subset does not refit one model per support.

## Support semantics

The composition resolver maps raw element rules into the generic k-sparse best-subset optimizer:

- required components stay free-valued and are present in every candidate;
- positive component lower bounds also imply required support;
- forbidden components and zero upper bounds are fixed to zero;
- only optional element fraction features are placed in the combinatorial `comp_idx` group;
- `k` is derived from `max_components - required_component_count`;
- fraction decisions receive a fixed-total equality constraint and final repair;
- active dimensions receive support-conditional positive floors, so an exact-k support cannot silently collapse to fewer active elements;
- one support is shared by the whole joint q-batch.

Search strategy can be configured on the composition site or through `OptimizeConfig.optimizer_kwargs`. Explicit `OptimizeConfig` values take precedence.

```python
from bochan.api import OptimizeConfig

opt_config = OptimizeConfig(
    optimizer_kwargs={
        "best_subset_strategy": "auto",  # exact | beam | auto
        "best_subset_max_combinations": 2000,
        "best_subset_beam_width": 8,
        "best_subset_beam_steps": 4,
        "best_subset_max_evaluations": 200,
    }
)
```

## Component step grids

Best Subset can also return experiment-ready stepped compositions, for example 5 at.% increments:

```python
composition_sites={
    "alloy": {
        # ...
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
        "min_components": 3,
        "max_components": 3,
        "steps": {
            "Al": 0.05,
            "Ti": 0.05,
            "V": 0.05,
            "Nb": 0.05,
        },
    }
}
```

The inner acquisition optimization remains continuous. For each exact support, the final composition is then projected with a small mixed-integer program onto the nearest feasible grid point while preserving:

- the selected exact-k support;
- component lower/upper bounds;
- every configured component step;
- the fixed composition total.

The acquisition function is re-evaluated on the projected candidate, so support ranking is based on the actual experiment-space composition rather than on a continuous candidate that is rounded only after selection.

All enumerated supports are checked for grid feasibility before candidate optimization. A support that cannot satisfy the configured total on its step grid causes an explicit configuration error.

### Step-grid strategy scope

Step-grid Best Subset currently uses exhaustive support search:

- `best_subset_strategy="exact"` is supported;
- `"auto"` is supported only when the support count is within `best_subset_max_combinations`, so Auto resolves to Exact;
- `"beam"` with component steps is rejected explicitly.

The current step-grid projector changes only the composition block. To avoid silently invalidating a constraint after projection, step-grid Best Subset also rejects for now:

- additional linear constraints involving composition fractions / elements;
- non-zero fixed composition fraction values;
- optimizer backends that bypass `final_candidate_postprocess`.

Process-only linear constraints and process-variable rounding/fixed values remain compatible because they do not alter the projected composition block.

## Web / API

The React/FastAPI workbench transports component steps unchanged for Best Subset. Selecting Best Subset no longer clears configured steps.

The UI warns when a stepped search would resolve to Beam. The canonical backend resolver remains the source of truth for Exact/Auto thresholds, grid feasibility, and unsupported constraint combinations.

For CLR / ALR / ILR, the Web response restores formula and fraction columns from the exact raw decision candidate rather than inferring support from finite pseudocount log-ratio coordinates.

## Current scope

Supported:

- one best-subset composition site per candidate optimization;
- fixed composition total;
- Fraction / CLR / ALR / ILR surrogate representations;
- exact cardinality (`min_components == max_components`);
- element bounds;
- required and forbidden elements;
- continuous compositions with exact, beam, or auto support search;
- component step grids with Exact support search (or Auto resolving to Exact);
- shared support for joint q-batches;
- Web/API transport for raw-space and stepped Best Subset settings.

Still explicit future extensions:

- variable-total composition sites;
- multiple simultaneous composition best-subset groups;
- variable active-component ranges (`min_components != max_components`);
- Beam search over stepped compositions;
- joint step-grid MILP enforcement of additional composition linear constraints;
- one-shot acquisition functions that introduce augmented optimization variables.

These cases are intentionally rejected instead of falling back to transformed-coordinate sparsity or post-hoc rounding that would change the optimization problem.
