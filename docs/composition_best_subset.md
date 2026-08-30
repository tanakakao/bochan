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

## Component step grids

Best Subset can return experiment-ready stepped compositions. The inner acquisition optimization stays continuous; for each exact support, the final composition is projected with a small mixed-integer program onto the nearest feasible experiment-space grid point. The acquisition function is then re-evaluated on that projected candidate, so support ranking uses the composition that can actually be executed.

### Fixed-total example

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

The fixed-total projection preserves:

- the selected exact-k support;
- component lower/upper bounds;
- every configured component step;
- the fixed composition total.

### Variable-total example

For a variable-total composition, `steps` are absolute-amount increments:

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
        "steps": {
            "Al": 5.0,
            "Ti": 5.0,
            "V": 5.0,
            "Nb": 5.0,
        },
        "required_components": ["Al"],
        "min_components": 3,
        "max_components": 3,
        "support_selection": "best_subset",
        "best_subset_strategy": "exact",
    }
}
```

The variable-total MILP operates directly in raw absolute amounts and preserves:

- the selected exact-k element support;
- absolute component lower/upper bounds;
- every configured absolute-amount step;
- `total_lower <= sum(amounts) <= total_upper`.

Unlike fixed-total projection, the projected total is allowed to move inside `total_bounds`. This is deliberate: the experiment-space candidate closest to the continuous acquisition optimum is chosen while the total remains a genuine decision variable.

All enumerated supports are checked for grid feasibility before candidate optimization. A support that cannot satisfy the fixed total or variable `total_bounds` on its configured grid causes an explicit configuration error rather than being rounded post hoc.

### Step-grid strategy scope

Step-grid Best Subset currently uses exhaustive support search for both fixed and variable totals:

- `best_subset_strategy="exact"` is supported;
- `"auto"` is supported only when the support count is within `best_subset_max_combinations`, so Auto resolves to Exact;
- `"beam"` with component steps is rejected explicitly.

The current step-grid projectors also reject for now:

- additional linear constraints involving composition fractions / raw amounts / elements;
- non-zero fixed composition values;
- optimizer backends that bypass `final_candidate_postprocess`.

Process-only linear constraints and process-variable rounding/fixed values remain compatible because they do not alter the projected composition block.

## Web / API

The React/FastAPI composition workbench supports the same fixed-total and variable-total contracts as the tabular API:

- choose **fixed total** to send `total`;
- choose **variable total** to send `total_bounds`;
- the two fields are mutually exclusive;
- variable-total formula data derives a separate fitted total feature from the original formula coefficients;
- Fraction / CLR / ALR / ILR model coordinates remain normalized composition features;
- variable-total Best Subset uses raw absolute element amounts internally and exposes the optimized total in the candidate result;
- variable-total `steps` are entered as absolute element-amount increments and use the same Exact-only step-grid contract as the Python API.

The Web search-space controls therefore distinguish between fixed-total element amounts and variable-total absolute amounts. For a variable-total Best Subset result, response metadata reports `support_space="raw_amount"`, `total_bounds`, and the generated total-feature name. Structural zeros and the optimized total are restored from the exact raw candidate rather than inferred from pseudocount-smoothed log-ratio coordinates.

When a step grid is active, the Web result is restored from the projected raw amount candidate. The displayed total is therefore the sum of the actual stepped amounts, not the pre-projection continuous total.

For fixed-total CLR / ALR / ILR searches, the Web response continues to restore formula and fraction columns from the exact raw-fraction decision candidate.

## Current scope

Supported:

- one Best Subset composition site per candidate optimization;
- fixed-total and variable-total compositions in Python/tabular and React/FastAPI Web workflows;
- Fraction / CLR / ALR / ILR surrogate representations;
- exact cardinality (`min_components == max_components`);
- required and forbidden elements;
- fixed-total fractional/amount bounds and variable-total absolute amount bounds;
- continuous compositions with Exact, Beam, or Auto support search;
- fixed-total and variable-total component step grids with Exact support search (or Auto resolving to Exact);
- variable-total `total_bounds` and raw amount linear constraints when no step grid is active;
- shared support for joint q-batches;
- Web result restoration from exact raw fraction or raw amount decisions.

Still explicit future extensions:

- multiple simultaneous composition Best Subset groups;
- variable active-component ranges (`min_components != max_components`);
- Beam search over stepped compositions;
- joint step-grid MILP enforcement of additional composition linear constraints;
- one-shot acquisition functions that introduce augmented optimization variables.

These cases are intentionally rejected instead of falling back to transformed-coordinate sparsity or post-hoc rounding that would change the optimization problem.
