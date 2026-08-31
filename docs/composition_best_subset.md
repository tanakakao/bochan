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
            "min_components": 2,
            "max_components": 4,
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

## Support and cardinality semantics

The composition resolver maps raw element rules into the generic sparse Best Subset optimizer:

- required components are present in every candidate;
- positive component lower bounds also imply required support;
- forbidden components and zero upper bounds are fixed to zero;
- only optional element variables are placed in the combinatorial `comp_idx` group;
- `min_components` / `max_components` count **all** active elements;
- required elements are subtracted before the residual optional-cardinality range is passed to generic Best Subset;
- active dimensions receive support-conditional positive floors, so a selected support cannot silently collapse to fewer active elements;
- one support and one cardinality are shared by the whole joint q-batch.

For example, with one required element and:

```python
"min_components": 2,
"max_components": 4,
```

the generic optional group searches `k = 1, 2, 3`. The selected total element count is still 2, 3, or 4 because the required element is included in every support.

Conceptually, continuous variable-cardinality Best Subset selects both the support and its size by the same acquisition value:

```text
(k*, S*) = arg max_k max_{|S|=k} max_{x: supp(x)=S} acquisition(x)
```

No cardinality penalty is added implicitly. If a user wants to prefer simpler supports, that preference should be expressed deliberately in the objective/acquisition rather than hidden inside support enumeration.

### Exact, Beam, and Auto across cardinalities

Search strategy can be configured on the composition site or through `OptimizeConfig.optimizer_kwargs`. Explicit `OptimizeConfig` values still take precedence for the normal search-strategy controls.

For a variable-cardinality search:

- **Exact** enumerates every feasible support at every allowed cardinality;
- **Auto** compares the **sum** of support counts over the whole cardinality range against `best_subset_max_combinations`;
- **Beam** seeds every allowed cardinality and explores `swap`, `add`, and `drop` neighborhoods, allowing the search to move both within one k and between adjacent k values.

The exact support-count guard therefore uses:

```text
sum(comb(n_optional, k) for k in allowed_optional_cardinalities)
```

The composition site owns the cardinality range. Directly supplying conflicting generic `best_subset_min_k` / `best_subset_max_k` values is rejected rather than creating two independent sources of truth.

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
        "min_components": 2,
        "max_components": 4,
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

This choice keeps support, cardinality, absolute component bounds, `total_bounds`, and amount-basis element constraints in one decision space. It also avoids introducing bilinear constraints of the form `fraction * total`.

`total_bounds` are enforced as linear inequalities on the sum of raw amounts. A linear constraint involving the fitted total feature is expanded to the same sum of amount variables. Element linear constraints for a variable-total Best Subset site are resolved directly to synthetic `__amount__<Element>` variables.

Exact, Beam, and Auto support/cardinality search reuse the same generic Best Subset implementation used by fixed-total compositions. The fitted surrogate and acquisition function are still evaluated in their original model space.

## Component step grids

Best Subset can return experiment-ready stepped compositions. The inner acquisition optimization stays continuous; for each exact support, the final composition is projected with a small mixed-integer program onto the nearest feasible experiment-space grid point. The acquisition function is then re-evaluated on that projected candidate, so support ranking uses the composition that can actually be executed.

The MILP step projectors preserve the support selected by Best Subset and accept every resolved cardinality between `min_components` and `max_components`. The discrete support/cardinality search remains owned by Exact / Beam / Auto; the projector optimizes only the active component values on the experiment grid.

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

- the selected support and its cardinality;
- component lower/upper bounds;
- every configured component step;
- the fixed composition total;
- supported composition-only linear equalities and inequalities.

For fixed-total sites, the continuous optimizer expresses composition constraints on fractions while the grid MILP works internally in total-scaled component amounts. bochan therefore maps `c @ fraction >= rhs` to the equivalent `c @ amount >= rhs * total` before MILP projection. The same conversion is applied to equalities and to repair constraints.

A composition-only constraint is enforced twice deliberately: once during continuous acquisition optimization and again during final grid projection. The final acquisition value is then recomputed on the projected candidate. This prevents support ranking from using a continuous candidate whose nearest step-grid point violates the intended composition constraint.

Process-only constraints remain compatible because the composition projector does not modify process dimensions. A single stepped constraint that mixes composition and process terms is still rejected explicitly: projecting only the composition block could otherwise invalidate the coupled relation after process postprocessing.

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

- the selected element support and its cardinality;
- absolute component lower/upper bounds;
- every configured absolute-amount step;
- `total_lower <= sum(amounts) <= total_upper`;
- composition-only linear equalities and inequalities on raw amounts.

A fitted total-feature constraint is first expanded to the equivalent sum of raw element amounts, so it is enforced consistently by both the continuous optimizer and the final step-grid MILP. Raw-amount element constraints are copied to the same projector without scale conversion.

Unlike fixed-total projection, the projected total is allowed to move inside `total_bounds`. This is deliberate: the experiment-space candidate closest to the continuous acquisition optimum is chosen while the total remains a genuine decision variable.

A composition constraint can make only some element supports feasible. Such support-specific grid infeasibility is represented explicitly and skipped by Best Subset rather than being treated as a global optimizer failure. If no support is feasible, candidate search fails explicitly. Unrelated optimizer and configuration exceptions are not swallowed.

### Step-grid strategy scope

Step-grid Best Subset supports both exhaustive and approximate support search for fixed and variable totals:

- `best_subset_strategy="exact"` enumerates every exact-cardinality support within `best_subset_max_combinations`;
- `"beam"` explores only the configured support-evaluation budget;
- `"auto"` uses Exact below `best_subset_max_combinations` and Beam above it;
- every evaluated support is ranked by the acquisition value of its final MILP-projected experiment-space candidate;
- grid/linear-constraint-infeasible supports are skipped explicitly during Beam search;
- variable-cardinality step grids preserve the cardinality of each evaluated support during MILP projection.

Exact mode prevalidates the complete support set because it will enumerate it anyway. Beam deliberately avoids that combinatorial pre-scan: feasibility is checked lazily when a support is evaluated. If the heuristic top-k seed is infeasible, Beam walks neighboring supports within `best_subset_max_evaluations` until it finds a feasible starting point or exhausts the budget.

For both fixed-total and variable-total stepped compositions, composition-only linear equalities and inequalities are enforced inside the MILP projector. The remaining step-grid restrictions are:

- a constraint that mixes composition and process variables is rejected;
- non-zero fixed composition values are rejected;
- optimizer backends that bypass `final_candidate_postprocess` are rejected.

Process-only linear constraints and process-variable rounding/fixed values remain compatible because they do not alter the projected composition block.

## Web / API

The React/FastAPI composition workbench supports the same fixed-total and variable-total contracts as the tabular API:

- choose **fixed total** to send `total`;
- choose **variable total** to send `total_bounds`;
- the two fields are mutually exclusive;
- Best Subset exposes separate minimum and maximum active-element counts;
- Auto computes its Exact/Beam decision from the support count over the complete allowed cardinality range;
- Beam requires enough evaluation budget to seed each allowed cardinality at least once;
- variable-total formula data derives a separate fitted total feature from the original formula coefficients;
- Fraction / CLR / ALR / ILR model coordinates remain normalized composition features;
- variable-total Best Subset uses raw absolute element amounts internally and exposes the optimized total in the candidate result;
- component steps use the same variable-cardinality, support-preserving MILP contract as the Python API.

The Web search-space controls therefore distinguish between fixed-total element amounts and variable-total absolute amounts. For a variable-total Best Subset result, response metadata reports `support_space="raw_amount"`, `total_bounds`, and the generated total-feature name. Structural zeros and the optimized total are restored from the exact raw candidate rather than inferred from pseudocount-smoothed log-ratio coordinates.

When a step grid is active, the Web result is restored from the projected raw amount candidate. The displayed total is therefore the sum of the actual stepped amounts, not the pre-projection continuous total.

For fixed-total CLR / ALR / ILR searches, the Web response continues to restore formula and fraction columns from the exact raw-fraction decision candidate.

## Current scope

Supported:

- one Best Subset composition site per candidate optimization;
- fixed-total and variable-total compositions in Python/tabular and React/FastAPI Web workflows;
- Fraction / CLR / ALR / ILR surrogate representations;
- continuous variable cardinality (`min_components <= max_components`);
- exact cardinality as the special case `min_components == max_components`;
- required and forbidden elements;
- fixed-total fractional/amount bounds and variable-total absolute amount bounds;
- continuous compositions with Exact, Beam, or Auto support/cardinality search;
- fixed-total and variable-total component step grids with exact or variable cardinality and Exact, Beam, or Auto support search;
- fixed-total raw-fraction and variable-total raw-amount composition-only linear constraints with step-grid Best Subset projection;
- variable-total `total_bounds` and raw amount linear constraints;
- shared support/cardinality for joint q-batches;
- Web result restoration from exact raw fraction or raw amount decisions.

Still explicit future extensions:

- multiple simultaneous composition Best Subset groups;
- stepped constraints that couple composition and process variables;
- one-shot acquisition functions that introduce augmented optimization variables.

These cases are intentionally rejected instead of falling back to transformed-coordinate sparsity or post-hoc rounding that would change the optimization problem.
