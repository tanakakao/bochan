# Linear constraints between composition elements

`TabularBayesianOptimizer` can constrain linear relationships between element
amounts within one composition site or across multiple sites.

## Exact ratio

The following condition keeps Sr at one half of La in atomic amount:

```python
bo = TabularBayesianOptimizer(
    input_cols=["A_La", "A_Sr", "A_Ba", "temperature"],
    target_cols="property",
    composition_sites={
        "A": {
            "element_columns": {
                "La": "A_La",
                "Sr": "A_Sr",
                "Ba": "A_Ba",
            },
            "representation": "ilr",
            "total": 1.0,
            "min_components": 2,
            "max_components": 3,
        }
    },
    composition_element_constraints=[
        {
            "terms": [
                {"site": "A", "element": "Sr", "coefficient": 1.0},
                {"site": "A", "element": "La", "coefficient": -0.5},
            ],
            "operator": "=",
            "rhs": 0.0,
            "basis": "atomic_amount",
        }
    ],
)
```

This represents

```text
A_Sr - 0.5 * A_La = 0
```

## Ratio range

Two inequalities express `0.4 * La <= Sr <= 0.6 * La`:

```python
composition_element_constraints=[
    {
        "terms": [
            {"site": "A", "element": "Sr", "coefficient": 1.0},
            {"site": "A", "element": "La", "coefficient": -0.4},
        ],
        "operator": ">=",
        "rhs": 0.0,
    },
    {
        "terms": [
            {"site": "A", "element": "Sr", "coefficient": 1.0},
            {"site": "A", "element": "La", "coefficient": -0.6},
        ],
        "operator": "<=",
        "rhs": 0.0,
    },
]
```

## Cross-site constraints

Terms can refer to different sites:

```python
composition_element_constraints=[
    {
        "terms": [
            {"site": "A", "element": "La", "coefficient": 1.0},
            {"site": "B", "element": "Fe", "coefficient": -0.5},
        ],
        "operator": "=",
        "basis": "atomic_amount",
    }
]
```

This represents `A_La = 0.5 * B_Fe` in atomic amount.

## Atomic and weight bases

- `basis="atomic_amount"` applies coefficients to atom or mole amounts.
- `basis="weight_amount"` applies coefficients to mass amounts.

Each site's `normalization` or `input_basis` may differ. Atomic weights are used
to convert the site's native amount before evaluating a constraint.

## Interaction with composition constraints

The element constraints are solved together with:

- fixed or candidate-dependent site totals;
- component lower and upper bounds;
- minimum and maximum active-element counts;
- required elements;
- element step sizes.

The implementation enumerates candidate active-element supports and solves a
mixed-integer linear nearest-candidate problem. An error is raised when no joint
solution exists.

For fixed-total Fraction representations, compatible element constraints are
also forwarded to the existing named linear-constraint optimizer. CLR, ALR,
ILR, variable-total, sparse, and stepped cases are enforced after inverse
transformation.

## Repair-aware reranking

By default, the optimizer oversamples raw candidates, repairs them, removes
duplicates, evaluates the acquisition function again with `q=1`, and returns
the best repaired candidates.

```python
bo = TabularBayesianOptimizer(
    ...,
    composition_constraint_rerank=True,
    composition_constraint_rerank_factor=4,
)
```

Set `composition_constraint_rerank=False` to retain the original candidate
pipeline while still applying exact element-constraint repair.
