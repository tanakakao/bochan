# ALIGNN Phase 6: scaling and structure-feature caching

Phase 6 reduces the cost of repeated acquisition optimization for structure-aware
ALIGNN models without changing the public `alignn_gp` / `alignn_dkl` model names
or the Phase-3 feasible-space contract.

## 1. Frozen structure representation cache

`ALIGNNGPModel` uses a frozen ALIGNN encoder. The crystal graph for a known
structure therefore has the same ALIGNN representation every time it appears in
training, prediction, or acquisition optimization.

Phase 6 lazily computes one representation for every graph in the fitted
structure bank and reuses those tensors by structure index:

```text
structure graph bank
        |
        | ALIGNN encoder, once per cache fill
        v
[structure_0_embedding, ..., structure_n_embedding]
        |
        | index_select(structure_index)
        v
process fusion -> projection -> GP kernel
```

This is especially useful during acquisition optimization, where the same
structures may otherwise be encoded repeatedly across raw samples, restarts,
and continuous optimization steps.

The cache is:

- enabled only while the ALIGNN encoder is frozen;
- detached and computed under `torch.no_grad()`;
- device/dtype aware and rebuilt when necessary;
- invalidated automatically when encoder parameter mutation counters change,
  including checkpoint/state-dict weight replacement;
- excluded from full pickle artifacts as well as `state_dict()`, so `.bochan.pt`
  files do not duplicate derived embeddings and rebuild them lazily after load;
- explicitly invalidated when the encoder training policy changes.

`ALIGNNDKLModel` partial/full fine-tuning does **not** use this cache because its
encoder representation changes during training.

Python users can inspect or reset the cache through:

```python
model.structure_feature_cache_enabled
model.clear_structure_feature_cache()
```

## 2. Why full mixed enumeration becomes expensive

The Phase-3 exact candidate contract enumerates:

```text
selected structures x currently observed joint process-category assignments
```

and optimizes the continuous variables for every fixed discrete configuration.
This is robust and remains the default for small spaces. However, the number of
continuous solves scales with the product of those two counts.

For example:

```text
40 structures x 8 observed process-category tuples = 320 fixed configurations
```

The process category tuples are deliberately **joint observed assignments**.
Bochan does not convert them into an independent Cartesian product because doing
so would introduce process combinations that have never been admitted by the
Phase-3 search-space contract.

## 3. Automatic alternating structure search

For the standard `optimize_acqf` backend with `q=1`, Phase 6 automatically
switches when more than 10 selected structures remain. The threshold follows the
same scale used by BoTorch for deciding when mixed alternating optimization is
preferable to direct enumeration.

The scalable path is:

```text
for each observed joint process-category tuple:
    fix the complete process-category tuple
    optimize structure as a categorical dimension
        with optimize_acqf_mixed_alternating
    optimize continuous process variables jointly

select the best acquisition value across process-category tuples
```

Thus a search with 40 structures and 8 observed process-category tuples invokes
8 alternating mixed optimizations rather than materializing 320 structure/category
fixed-feature configurations.

The structure selector remains discrete. It is never relaxed into an ordinary
continuous process variable. Bochan also defaults the alternating optimizer to
`initialization_strategy="random"`, so initialization evaluates admissible
categorical structure values rather than fractional structure indices. An
advanced caller may explicitly override this with an alternating-compatible
option when appropriate.

## 4. Conservative fallback rules

Exact Phase-3 enumeration is retained when any of the following is true:

- 10 or fewer selected structures are being searched;
- `q > 1`;
- `return_best_only=False`;
- a non-standard optimizer backend such as `torch`, evolutionary optimization,
  Thompson sampling, or a custom callable was requested;
- `optimizer_kwargs` contains settings that are valid for ordinary
  `optimize_acqf` but are not supported by BoTorch's alternating optimizer.

For example, a standard SciPy option such as:

```python
{"options": {"maxiter": 100}}
```

keeps exact enumeration rather than changing meaning or failing merely because
the structure catalog crossed the automatic threshold. Alternating-compatible
settings such as `batch_limit`, `init_batch_limit`, `maxiter_alternating`, or an
explicit `initialization_strategy` may still use the scalable backend.

The `q > 1` fallback is intentional. Exact enumeration preserves the existing
batch semantics where different q slots may choose different process-category
assignments. A future phase can add a native joint alternating batch strategy
once that behavior can be preserved explicitly.

## 5. FastAPI behavior

No new endpoint or request field is required. Existing calls continue to use:

```text
POST /api/v1/tabular/alignn/models/{model_id}/candidates
POST /api/v1/tabular/alignn/models/{model_id}/ask
```

For the default optimizer, the server chooses exact enumeration or scalable
alternating structure search from the fitted search-space size. `structure_ids`
still restricts the allowed structure subset before this decision is made.

If the restricted subset contains 10 or fewer structures, exact enumeration is
used even if the full structure catalog is much larger.

## 6. Scientific/search-space semantics retained

Phase 6 changes execution strategy, not model meaning:

- structure ID remains a graph selector;
- process categorical dimensions remain categorical kernel inputs;
- only observed joint process-category assignments are admissible;
- continuous process coordinates retain acquisition gradients;
- frozen ALIGNN-GP and trainable ALIGNN-DKL remain distinct training regimes;
- unknown/new crystal-structure generation remains out of scope.
