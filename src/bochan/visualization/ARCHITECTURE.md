# Visualization architecture

`bochan.visualization` uses a small number of concrete owner modules rather than compatibility layers or forwarding packages.

## Package contract

The package root, `bochan.visualization`, is a public export surface only. It must not replace functions on imported modules, mutate module attributes, or install runtime compatibility aliases.

The stable high-level plotting entry points are owned by `bochan.visualization.plots`:

- `show_1dplot_from_optimizer`
- `show_scatter_with_acqf_from_optimizer`
- `show_triscatter_with_acqf_from_optimizer`
- `show_yyplot_from_optimizer`

Task-specific modules provide concrete implementations used by that dispatcher. They are not compatibility wrappers.

## Concrete owners

### Data builders

`data/` is a package because its responsibilities are independent and reusable:

- `data/frames.py`: training, prediction, candidate, and YY DataFrames
- `data/grids.py`: 1D and 2D evaluation grids
- `data/ternary.py`: simplex / ternary grids
- `data/study.py`: study-target DataFrames

### Task-aware probability views

These modules remain flat intentionally. They form one dispatch graph and moving them into nested packages would add cross-package relative imports without improving ownership.

- `multiclass.py`: multiclass probabilities, 1D views, and 2D heatmaps
- `multiclass_ternary.py`: multiclass ternary views
- `multiclass_yy.py`: multiclass YY probability views
- `ordinal.py`: ordinal probabilities and probability views
- `ordinal_display.py`: public ordinal display selection

### Prediction uncertainty

These modules also remain flat because their primitives are shared by data builders and task-aware plots.

- `input_perturbation.py`: perturbation-aware prediction moments and aggregation
- `heteroscedastic_1d.py`: heteroscedastic uncertainty decomposition
- `probability_1d.py`: bounded discrete-output 1D probability presentation

### Generic rendering and diagnostics

- `plots.py`: generic Plotly renderers and the canonical optimizer-plot dispatcher
- `feature_importance.py`: feature-importance and model-diagnostic figures
- `target_relation.py`: target-to-target figures
- `study.py`: study history and Pareto figures
- `categorical_axis.py`: categorical-axis rendering helpers
- `_heatmap_layout.py`: probability-heatmap layout helpers
- `utils.py`: shared visualization-only utilities

## Architecture rules

1. Every public behavior has one concrete owner.
2. `__init__.py` files may export symbols but must not monkey-patch imported modules.
3. Do not add compatibility aliases, forwarding modules, or forwarding packages when moving code. Update consumers to the canonical owner instead.
4. Do not recreate the removed flat `data.py` or `probability_input_perturbation.py` modules.
5. Add a subpackage only when it contains multiple independently meaningful responsibilities. Folder depth is not an architecture goal by itself.
6. Direct imports from `bochan.visualization.plots` must behave the same as package-root optimizer plotting imports.

The architecture test in `tests/test_visualization_architecture.py` locks these boundaries so future additions cannot silently return the package to an ambiguous flat collection or runtime-patched dispatch model.
