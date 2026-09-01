# MACE Phase 9: performance and production hardening

Phase 9 hardens the MACE structure representation path for larger structure
catalogs and reproducible deployment.

## Native structure batching

`MACEEncoder` now batches structures when bochan owns MACE graph construction.
The default maximum batch size is 16 structures per raw MACE forward. Atomwise
invariant `l=0` descriptors are pooled independently for each graph using the
native graph-membership tensor, so batching does not change the crystal
representation semantics.

A Python caller can construct `MACEEncoder(..., batch_size=N)` to choose another
chunk size. A custom `batch_builder` retains its existing one-structure callback
contract and therefore uses the sequential callback path.

## Frozen GP cache and trainable DKL deduplication

Frozen `mace_gp` and `mace_multitask` continue to cache the complete structure
representation bank. This remains the fastest path for repeated posterior and
acquisition evaluation.

DKL cannot use a persistent representation cache because encoder weights change.
Phase 9 instead deduplicates structure IDs within each feature request: if the
same structure occurs in many candidate rows, MACE encodes that structure once
and the differentiable representation is gathered back to all matching rows.
Gradient flow to trainable MACE layers is retained.

## Device and dtype contract

All tensor fields produced by both native graph construction and injected Python
`batch_builder` callbacks are moved to the raw MACE model device before forward.
Floating fields are cast to the native MACE floating dtype. The existing outer
GP boundary remains unchanged: the MACE backbone may stay float32 while GP/DKL
features are returned in the outer model dtype (commonly float64).

The Phase 9 tests include a CPU mixed-dtype boundary test and a CUDA test that
runs automatically on CUDA-capable runners.

## Reproducibility metadata

MACE fit/save/load metadata now records additional runtime fields:

- installed `mace-torch` version;
- MACE encoder device and native floating dtype;
- representation batch size;
- whether native MACE batching is enabled;
- the existing model/checkpoint, invariant representation, layer, pooling,
  head, cutoff, output-dependency, and encoder-training fields.

The save endpoint persists the complete `metadata.mace` block inside the common
artifact envelope in addition to returning it in the HTTP response. This makes
checkpoint/runtime provenance available when inspecting a saved artifact.

## Validation

`.github/workflows/mace-phase9-smoke.yml` validates:

- native batched representation parity against sequential encoding;
- reduced raw forward count from chunked structure batching;
- DKL structure-ID deduplication with retained gradients;
- custom batch device/dtype coercion;
- runtime reproducibility metadata;
- optional CUDA device propagation;
- real `medium-mpa-0` batched-vs-single-structure representation parity;
- Phase 7/8 optimization and FastAPI/artifact regressions;
- Ruff on the Phase 9 surface.
