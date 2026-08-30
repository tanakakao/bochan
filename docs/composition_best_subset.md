# Composition best-subset search

`composition_sites` can opt a composition into acquisition-aware element support search.

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

For fixed-total compositions, raw decisions are element fractions. For `representation="fractions"`, those decisions coincide with the model composition features. For `"clr"`, `"alr"`, and `"ilr"`, bochan uses a two-space bridge:

1. support selection and composition constraints are optimized in synthetic raw-fraction variables;
2. every candidate is differentiably transformed into the fitted CLR / ALR / ILR coordinates before the surrogate and acquisition function are evaluated;
3. exact raw fractions are retained as the support-defining values, including structural zeros.

The fitted surrogate is reused as-is. Best Subset does not refit one model per support.

## Support semantics

The composition resolver maps raw element rules into the generic k-sparse Best Subset optimizer:

- required components are present in every candidate;
- positive component lower bounds also imply required support;
- forbidden components and zero upper bounds are fixed to zero;
- only optional element variables are placed in the combinatorial `comp_idx` group;
- `k` is derived from `max_components - required_component_count`;
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

## Variable-total compositions

Variable-total Best Subset is optimized in **raw absolute component amounts**, not in `fraction + total` coordinates.

For example, an element-column site may be configured as:

```python
composition_sites={
    "alloy": {
        "element_columns": {
            "Al": "Al",
            "Ti": "Ti",
            "V": "V",
            "Nb": "Nb",
        },
        "representation": "ilr",
        "total_bounds": [40.0, 90.0],
        "bounds": {
            "Al": [5.0, 70.0],
            "Ti": [0.0, 70.0],
            "V": [0.0, 70.0],
            "Nb": [0.0, 70.0],
        },
        "required_components": ["Al"],
        "min_components": 3,
        "max_components": 3,
        "support_selection": "best_subset",
        "best_subset_strategy": "auto",
    }
}
```

If the raw amount vector is `a`, bochan derives the model inputs as follows:

- total = `sum(a)`;
- normalized composition = `a / sum(a)`;
- the normalized composition is transformed to Fraction / CLR / ALR / ILR coordinates;
- the derived total is inserted into the fitted model's total feature;
- process variables keep their normal decision coordinates.

This choice keeps support, absolute component bounds, `total_bounds`, and amount-basis element constraints in one decision space. It also avoids introducing bilinear constraints of the form `fraction * total`.

`total_bounds` are enforced as linear inequalities on the sum of raw amounts. A linear constraint involving the fitted total feature is expanded to the same sum of amount variables. Element linear constraints for a variable-total Best Subset site are resolved directly to synthetic `__amount__<Element>` variables.

Exact, Beam, and Auto support search reuse the same generic Best Subset implementation used by fixed-total compositions. The fitted surrogate and acquisition function are still evaluated in their original model space.

Variable-total component step grids are not yet supported; the continuous amount formulation is used in this phase.

## Component step grids

Fixed-total Best Subset can return experiment-ready stepped compositions, for example 5 at.% increments:

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

The inner acquisition optimization remains continuous. For each exact support, the final composition is projected with a small mixed-integer program onto the nearest feasible grid point while preserving:

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

The current fixed-total step-grid projector also rejects for now:

- additional linear constraints involving composition fractions / elements;
- non-zero fixed composition fraction values;
- optimizer backends that bypass `final_candidate_postprocess`.

Process-only linear constraints and process-variable rounding/fixed values remain compatible because they do not alter the projected composition block.

## Web / API

The current React/FastAPI composition workbench remains **fixed-total**. It transports fixed-total Best Subset settings, log-ratio raw-space search, and component steps, but it does not yet expose `total_bounds` controls.

The Python/tabular API supports the variable-total Best Subset path described above. A future Web phase can add `total_bounds` transport and UI without changing the raw absolute-amount optimization semantics.

For fixed-total CLR / ALR / ILR searches, the Web response restores formula and fraction columns from the exact raw decision candidate rather than inferring support from finite pseudocount log-ratio coordinates.

## Current scope

Supported:

- one Best Subset composition site per candidate optimization;
- fixed-total and variable-total compositions in the Python/tabular API;
- Fraction / CLR / ALR / ILR surrogate representations;
- exact cardinality (`min_components == max_components`);
- required and forbidden elements;
- fixed-total fractional bounds and variable-total absolute amount bounds;
- continuous compositions with Exact, Beam, or Auto support search;
- fixed-total component step grids with Exact support search (or Auto resolving to Exact);
- variable-total `total_bounds` and raw amount linear constraints;
- shared support for joint q-batches;
- fixed-total Web/API transport for raw-space and stepped Best Subset settings.

Still explicit future extensions:

- Web/React `total_bounds` controls for variable-total Best Subset;
- variable-total component step/grid projection;
- multiple simultaneous composition Best Subset groups;
- variable active-component ranges (`min_components != max_components`);
- Beam search over stepped fixed-total compositions;
- joint fixed-total step-grid MILP enforcement of additional composition linear constraints;
- one-shot acquisition functions that introduce augmented optimization variables.

These cases are intentionally rejected instead of falling back to transformed-coordinate sparsity or post-hoc rounding that would change the optimization problem.
