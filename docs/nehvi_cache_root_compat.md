# qNEHVI cache-root compatibility

BoTorch's cached-Cholesky qNEHVI path expects a multitask posterior covariance
that can be represented as independent output batches. Correlated Kronecker
classification and ordinal posteriors do not satisfy that assumption.

bochan models that cannot use this path expose:

```python
_supports_cache_root = False
```

The binary and ordinal multi-output qNEHVI wrappers now resolve `cache_root` as
follows:

1. An explicit `cache_root=True` or `cache_root=False` is preserved.
2. If omitted or set to `None`, `model._supports_cache_root` is used.
3. Models without the capability flag retain BoTorch's cached default.

Therefore, `KroneckerMultiTaskOrdinalGPModel` and
`KroneckerMultiTaskBinaryClassificationGPModel` require no user-side
`cache_root=False` workaround.
