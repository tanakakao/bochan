# Composition best-subset search

`composition_sites` can opt a fixed-total fraction composition into acquisition-aware element support search.

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
            "representation": "fractions",
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
        }
    },
)
```

The composition resolver maps the raw element rules into the generic k-sparse best-subset optimizer:

- required components stay free-valued and are present in every candidate;
- positive component lower bounds also imply required support;
- forbidden components and zero upper bounds are fixed to zero;
- only optional element fraction features are placed in the combinatorial `comp_idx` group;
- `k` is derived from `max_components - required_component_count`;
- the fraction features receive a unit-sum optimizer constraint and final repair;
- active optional dimensions receive support-conditional positive floors during repair, so an exact-k selected support cannot silently collapse to fewer active elements.

Search strategy is still configured through the existing optimizer controls:

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

## Why log-ratio coordinates are not sparsified

Element support is defined in raw composition space. A zero CLR, ALR, or ILR coordinate does not mean that an element is absent. Therefore `support_selection="best_subset"` currently rejects `representation="clr"`, `"alr"`, and `"ilr"` rather than treating transformed coordinates as elements.

The first supported integration intentionally uses `representation="fractions"` (or `"none"`), where each optimization feature corresponds one-to-one with one element fraction.

## Current scope

The initial composition integration supports:

- one best-subset composition site per candidate optimization;
- fixed composition total;
- continuous fraction variables;
- exact cardinality (`min_components == max_components`);
- element bounds;
- required and forbidden elements;
- exact, beam, and auto support search from the shared core implementation;
- shared support for joint q-batches.

The following cases fail explicitly instead of falling back to semantically incorrect behavior:

- CLR / ALR / ILR support selection;
- variable-total composition sites;
- component step grids;
- multiple simultaneous composition best-subset groups;
- variable active-component ranges (`min_components != max_components`).

Those cases require either raw-space support variables alongside transformed model coordinates or multi-group support search, and should be added as explicit extensions rather than inferred from transformed-coordinate sparsity.
