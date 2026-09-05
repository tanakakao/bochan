# Phase 60 — Multi-dimensional Fidelity Optimization

Phase 60 extends query-fidelity optimization to models with multiple continuous fidelity features.

## Discrete Cartesian-product search

```python
OptimizeConfig(
    fidelity_values={
        -2: [0.25, 0.5, 1.0],
        -1: [0.5, 1.0],
    },
)
```

Each mapping key resolves against the public input dimension, so negative indices use the same convention as `fidelity_features`. Every configured fidelity feature must be present. The allowed values are expanded into the Cartesian product and combined with any categorical `fixed_features_list` assignments before mixed optimization.

## Explicit discrete assignments

```python
OptimizeConfig(
    fidelity_assignments=[
        {-2: 0.25, -1: 0.5},
        {-2: 0.5, -1: 1.0},
        {-2: 1.0, -1: 1.0},
    ],
)
```

This avoids evaluating invalid or meaningless Cartesian-product combinations. Every assignment must specify every configured fidelity feature.

## Continuous joint search

```python
OptimizeConfig(optimize_fidelity=True)
```

All configured fidelity dimensions remain free and are optimized jointly with the design variables. Fixing any fidelity dimension through `fixed_features` or `fixed_features_list` is rejected in this mode.

## Backward compatibility

The existing single-fidelity form remains valid:

```python
OptimizeConfig(fidelity_values=[0.25, 0.5, 1.0])
```

A plain sequence is intentionally accepted only when the model has exactly one fidelity feature.

`fidelity_values`, `fidelity_assignments`, and `optimize_fidelity=True` are mutually exclusive query-fidelity modes.
