# Material models architecture — Phase 10

## Scope

Phase 10 introduces a backend-neutral residual-GP layer for pretrained material
models that can make direct property predictions.

The intended decomposition is:

```text
pretrained direct prediction
        +
Gaussian-process residual correction
        =
corrected BoTorch posterior
```

This phase does not attach concrete MACE / CHGNet / M3GNet family adapters yet.
It establishes the common tensor, capability, residual-target, posterior, and
conditioning contracts that those adapters can use later.

## Direct predictor contract

`DirectMaterialPredictor` is a torch module with a fixed `output_dim` and the
forward contract:

```text
X:           [..., d]
prediction:  [..., output_dim]
```

Predictions must preserve leading dimensions and match `X` device and dtype.
Non-finite pretrained predictions are rejected.

## Residual targets

Training targets are transformed as:

```text
Y_residual = Y_observed - Y_pretrained
```

`compute_material_residual_targets` deliberately does not introduce a second
missing-observation policy. NaNs in observed targets remain NaNs after residual
construction, so the existing observation-aware Gaussian path remains the only
owner of partial-target semantics. Known `train_Yvar` values are not modified by
this layer.

## Corrected posterior

`ResidualMaterialGPModel` wraps a fitted BoTorch residual model. Posterior
samples and means are shifted by the deterministic pretrained prediction:

```text
mean_final     = mean_residual + prediction_pretrained
variance_final = variance_residual
```

The pretrained baseline is deterministic in this contract, so it contributes no
additional posterior variance. A future uncertainty-aware pretrained predictor
would require a separate probabilistic composition contract rather than silently
adding uncertainty here.

Standard `posterior_transform` handling remains available after the baseline
shift, allowing existing acquisition functions to consume the corrected
property-scale posterior.

## Conditioning / fantasization boundary

`condition_on_observations` accepts observations in the original property scale,
converts them to residual observations, and delegates conditioning to the
wrapped BoTorch model. This keeps future fantasy-model / lookahead workflows on
the same residual definition.

## Capability gate

A `PretrainedMaterialSpec` supplied to `ResidualMaterialGPModel` must declare
`residual_gp=True`. Phase 9 already guarantees that this implies direct
prediction support.

Concrete family capability declarations remain deferred until adapter behavior
is verified. This prevents the architecture layer from over-claiming optional
backend functionality.

## Compatibility guarantees

Phase 10 does not change:

- existing CrabNet / Roost / ALIGNN / CHGNet / M3GNet / MACE model classes;
- public `model_type` strings;
- GP / DKL mathematics;
- pretrained loading behavior;
- frozen / partial / full fine-tuning semantics;
- mixed categorical kernels;
- correlated multitask covariance;
- partial-target and known-noise handling;
- structure feature caches;
- FastAPI / Web schemas;
- historical saved-model module paths.

## Next phase

Phase 11 adds the material family registry. Registry entries can combine domain,
canonical model classes, pretrained capability metadata, and supported surrogate
kinds without importing optional backends eagerly.
