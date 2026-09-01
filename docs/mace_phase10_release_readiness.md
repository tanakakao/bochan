# MACE Phase 10: release readiness

Phase 10 closes the MACE integration around operational use rather than adding another model family.
The supported public model types remain:

| model_type | encoder training | output dependency | process categories |
|---|---|---|---|
| `mace_gp` | frozen | independent for multiple targets | supported |
| `mace_dkl` | partial/full | independent for multiple targets | supported |
| `mace_multitask` | frozen | correlated | supported |
| `mace_multitask_dkl` | partial/full | correlated | supported |

The representation contract is unchanged: bochan uses the raw pretrained MACE torch model,
extracts invariant `l=0` channels from `node_feats`, pools atoms per crystal, and then fuses the
crystal representation with optional process variables before the GP kernel.

## Focused installation

MACE can now be installed without pulling every materials-model dependency:

```bash
pip install -e ".[mace,tabular]"
```

For the FastAPI surface:

```bash
pip install -e ".[mace,tabular,api]"
```

`bochan[materials]` remains the umbrella extra for MACE together with the other materials encoders.
The supported dependency range is `mace-torch>=0.3.16,<0.4`; CI pins `0.3.16` for reproducibility.

## Candidate generation is covered with the real pretrained model

Earlier phases validated the real `medium-mpa-0` encoder and posterior path, while acquisition
optimization used lightweight differentiable stand-ins to keep CI fast. Phase 10 additionally
runs a real pretrained `mace_gp` through structure-aware `logei` candidate optimization.

The validated path is therefore:

```text
structure catalog
  -> pretrained medium-mpa-0
  -> invariant crystal representation
  -> frozen representation bank cache
  -> GP posterior
  -> acquisition function
  -> discrete structure enumeration + continuous process optimization
  -> tabular candidate
```

This does not mean an arbitrary user-defined property can be optimized with zero target observations.
MACE supplies the pretrained structural representation; the GP for the user target still learns from
that target's observed data.

## DKL candidate path

Phase 10 also executes `mace_dkl` through actual acquisition optimization. DKL keeps the persistent
frozen structure cache disabled so gradients can reach trainable MACE interaction/product blocks.
Repeated structure IDs inside one feature evaluation are deduplicated before encoding, as established
in Phase 9.

## ask / pending / artifact lifecycle

The MACE FastAPI `ask` endpoint registers generated candidates as `pending` observations. Phase 10
makes the tabular facade dataset synchronize immediately after that registration instead of waiting
for a later save/load operation to synchronize it.

The resulting state contract is:

```text
POST /api/v1/tabular/mace/models/{id}/ask
  -> candidate is generated
  -> underlying BayesianOptimizer gets a pending observation
  -> TabularBayesianOptimizer.dataset is synchronized immediately
  -> save preserves the same pending mask and NaN target placeholder
  -> trusted load restores the same observation state
```

Artifacts remain pickle-backed `.bochan.pt` files and therefore require `trust_pickle=true` only for
files produced by a trusted bochan process.

## Acquisition recommendation

For multi-objective MACE optimization, prefer `qlognehvi` / `lognehvi`. These resolve to BoTorch's
`qLogNoisyExpectedHypervolumeImprovement`. Legacy `qnehvi` / `nehvi` remain supported for explicit
compatibility but are not the recommended numerical path.

For single-objective MACE optimization, `logei` is the preferred expected-improvement path used by the
Phase 10 real-model candidate probe.

## Runtime and reproducibility metadata

The FastAPI fit/save metadata continues to record the MACE runtime contract introduced in Phase 9,
including:

- `mace_torch_version`
- pretrained model name
- invariant representation width
- selected interaction layers
- pooling and head
- cutoff
- native graph batch size
- native batching enabled/disabled state
- encoder device and dtype
- structure IDs and process dimensions
- independent versus correlated output dependency

Phase 10 does not change those semantics; it verifies that the lifecycle built on top of them remains
usable through candidate generation and artifact persistence.

## Phase 10 CI contract

The dedicated workflow validates:

1. immediate synchronization of MACE `ask` pending state;
2. pending-state artifact serialize/deserialize round-trip;
3. actual `mace_dkl` candidate optimization;
4. actual pretrained `medium-mpa-0` candidate optimization;
5. the focused `mace` optional dependency extra;
6. Phase 6-9 FastAPI, optimization, qLogNEHVI, performance, and artifact regressions;
7. Ruff on the Phase 10 Python surface.

With this phase, the MACE work is considered integration-complete for the current scope. Future work
such as additional MACE foundation checkpoints, composition+structure joint inputs, noisy-target
`train_Yvar`, or broader pretrained-property priors should be treated as new feature work rather than
continuation of the initial integration series.
