# Material `train_Yvar` Phase 3

Phase 3 exposes the fixed known-observation-variance support from Phases 1 and 2 through bochan's high-level tabular and FastAPI workflows.

## Tabular column contract

`TabularDataConfig.target_variance_cols` contains one **variance** column for each `target_cols` entry, in the same order. Values are variances, not standard deviations. They must be numeric, finite, and strictly positive.

Variance columns are metadata: when `input_cols` is inferred they are excluded from `X`, and explicitly including a variance column in `input_cols` is rejected.

```python
optimizer = TabularBayesianOptimizer(
    model_config={"model_type": "mace_multitask", "task_type": "multi_objective"},
    input_cols=["structure_id", "temperature"],
    target_cols=["strength", "ductility"],
    target_variance_cols=["strength_var", "ductility_var"],
    structure_col="structure_id",
    structure_catalog=structures,
    bounds={"temperature": [800.0, 1200.0]},
)
optimizer.fit(frame)
```

The resulting `TabularDataset.Yvar` is forwarded to `BayesianOptimizer.fit`. Correlated multitask models receive the full `[n, m]` tensor. Independent multi-output models receive the corresponding `[n, 1]` slice for each output.

## Direct tensor API

The core optimizer accepts known observation variance directly:

```python
optimizer.fit(train_X, train_Y, train_Yvar)
```

`refit()` preserves the stored variance. `tell(new_X, new_Y, new_Yvar)` appends it. Once an optimizer was fitted with known variance, every later observed row must also provide `new_Yvar`; partial known-noise histories are rejected.

## FastAPI

The direct tensor endpoints accept `train_Yvar`, and `/models/{model_id}/tell` accepts `new_Yvar`. Tabular fit requests accept `target_variance_cols` and material-specific tabular schemas inherit the same field.

## Noise-policy rules

Known per-row variance and a global `alpha` / explicit tabular likelihood are different noise contracts. Phase 3 rejects `target_variance_cols` together with `alpha` or `model_kwargs.likelihood` instead of silently choosing one.

When no variance is supplied, existing learned-noise behavior is unchanged.

## Current boundary

Observation-aware tabular conversion (`target_missing_strategy="keep"` or `experiment_status_col`) is intentionally rejected together with `target_variance_cols` in Phase 3. That path needs a separate contract for missing/pending rows whose objective variance is not yet observed.
