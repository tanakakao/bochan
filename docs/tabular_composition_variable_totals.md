# Variable composition-site totals

A composition site can use a bounded total instead of a fixed total. The site
sum is then included as an ordinary numeric model feature, while the within-site
ratios continue to use Fraction, CLR, ALR, or ILR coordinates.

## One variable site total

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    input_cols=["A_La", "A_Sr", "A_Ba", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1100.0]},
    composition_sites={
        "A": {
            "element_columns": {
                "La": "A_La",
                "Sr": "A_Sr",
                "Ba": "A_Ba",
            },
            "representation": "ilr",
            "total_bounds": [30.0, 70.0],
            "min_components": 1,
            "max_components": 2,
            "required_components": ["La"],
        }
    },
)
```

The transformed model inputs contain the ILR coordinates and an additional
`A__total` feature. For element-column input, the training value of `A__total`
is the row-wise sum of `A_La`, `A_Sr`, and `A_Ba`.

Generated candidates satisfy

```text
30 <= A_La + A_Sr + A_Ba <= 70
```

and the returned `A__total` value equals that sum.

`total` and `total_bounds` are mutually exclusive. Component `bounds` and
`steps` remain expressed in the same absolute scale as the site total.

## Coupled A/B totals

Use `composition_total_constraints` when totals from multiple sites must satisfy
a shared linear constraint.

```python
bo = TabularBayesianOptimizer(
    input_cols=[
        "A_La", "A_Sr", "A_Ba",
        "B_Fe", "B_Co", "B_Mn",
        "temperature",
    ],
    target_cols="property",
    composition_sites={
        "A": {
            "element_columns": {
                "La": "A_La",
                "Sr": "A_Sr",
                "Ba": "A_Ba",
            },
            "total_bounds": [30.0, 70.0],
        },
        "B": {
            "element_columns": {
                "Fe": "B_Fe",
                "Co": "B_Co",
                "Mn": "B_Mn",
            },
            "total_bounds": [30.0, 70.0],
        },
    },
    composition_total_constraints=[
        {
            "sites": ["A", "B"],
            "operator": "=",
            "total": 100.0,
        }
    ],
)
```

This is translated to the named optimizer constraint

```text
A__total + B__total = 100
```

while preserving the individual site ranges. `<=` and `>=` operators and custom
coefficients are also supported.

```python
composition_total_constraints=[
    {
        "sites": ["A", "B"],
        "coefficients": [2.0, 1.0],
        "operator": "<=",
        "rhs": 120.0,
    }
]
```

Fixed and variable totals can be mixed. If B uses `total=50`, the constraint
`A + B = 100` is reduced internally to `A__total = 50`.

## Formula-column input

`total_bounds` is also accepted for formula-column sites. The total feature is
derived from the raw stoichiometric coefficients. For weight-basis input, it is
derived from the corresponding mass amounts.

The site total is a separate model feature because CLR, ALR, and ILR retain
ratios but deliberately discard absolute scale.
