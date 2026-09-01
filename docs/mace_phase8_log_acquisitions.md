# MACE Phase 8: numerically stable multi-objective acquisition

Phase 8 promotes BoTorch's log-domain hypervolume improvement path as the recommended MACE multi-objective acquisition contract.

## Recommendation

For noisy or observed-baseline multi-objective structure/process Bayesian optimization, prefer:

```text
qlognehvi
lognehvi
```

Both names resolve to BoTorch `qLogNoisyExpectedHypervolumeImprovement`.

Legacy `qnehvi` / `nehvi` remain supported and are not silently rewritten. This preserves explicit user intent and backward compatibility while avoiding a MACE-specific acquisition wrapper.

## Supported MACE output models

The log-domain path is validated for both MACE multi-output semantics:

- `mace_gp` / `mace_dkl` with multiple targets: independent output models through `ModelListGP`;
- `mace_multitask` / `mace_multitask_dkl`: one correlated wide-output model with a shared MACE representation and `MultitaskKernel`.

The MACE structure selector remains discrete. Continuous process variables remain differentiable through the GP/DKL acquisition path, and categorical process variables continue to use the common mixed fixed-feature / structure-aware optimizer contract.

## Python example

```python
candidates, value = optimizer.candidate(
    acq_name="qlognehvi",
    q=1,
    objective_mode="multi_output",
    objective_outputs=["strength", "conductivity"],
    objective_directions=["maximize", "maximize"],
    data_context=DataContext(
        X_baseline=optimizer.train_X,
        Y_baseline=optimizer.train_Y,
        ref_point=optimizer.train_Y.min(dim=0).values - 0.1,
    ),
    structure_ids=["alpha", "beta", "gamma"],
)
```

## FastAPI example

The existing MACE candidate endpoint accepts the same acquisition name without a separate route or compatibility layer:

```json
{
  "acquisition_config": {"name": "qlognehvi"},
  "optimize_config": {"q": 1},
  "objective_mode": "multi_output",
  "objective_outputs": ["strength", "conductivity"],
  "objective_directions": ["maximize", "maximize"],
  "structure_ids": ["alpha", "beta", "gamma"]
}
```

Use the same baseline/reference-point data-context contract required by the generic bochan multi-objective acquisition layer.

## Why the log path is preferred

BoTorch warns that the legacy `qNoisyExpectedHypervolumeImprovement` implementation has known numerical issues and recommends `qLogNoisyExpectedHypervolumeImprovement` instead. The log-domain implementation preserves the same optimization intent while improving numerical behavior during acquisition optimization.

Phase 8 therefore validates qLogNEHVI directly rather than changing the meaning of the existing `nehvi` alias.

## CI contract

`.github/workflows/mace-phase8-smoke.yml` validates:

- public `qlognehvi` / `lognehvi` resolution to the official BoTorch log acquisition;
- independent MACE multi-output qLogNEHVI optimization;
- correlated mixed MACE multitask qLogNEHVI optimization;
- FastAPI candidate schema/service propagation of `qlognehvi`;
- Phase 7 structure/acquisition regression coverage and Ruff.
