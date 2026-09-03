# Material model architecture — Phase 8

## Goal

Normalize the Gaussian surrogate construction boundary used by material-aware GP and DKL models without changing existing public model classes, model types, observation handling, or serialization paths.

## Architectural decision

Bochan's material GP and DKL variants share the same exact-Gaussian backend. The distinction is not a different GP class:

- **GP**: the material encoder is frozen and only the projection / GP / likelihood are trained.
- **DKL**: selected or all material-encoder parameters are trainable together with the projection / GP / likelihood.

Therefore Phase 8 introduces a common surrogate specification and builder instead of creating parallel GP and DKL backend implementations.

## Canonical API

`bochan.models.regression.gaussian.materials.common.surrogate` now defines:

- `MaterialSurrogateKind = Literal["gp", "dkl"]`
- `MaterialSurrogateSpec`
- `resolve_material_latent_dim`
- `build_material_gaussian_surrogate`

The same API is exported from `gaussian.materials.common` and `gaussian.materials`.

## Construction flow

```text
material encoder
    ↓
encoder training policy (Phase 5)
    ↓
material/process feature extractor
    ↓
MaterialSurrogateSpec
    ↓
build_material_gaussian_surrogate
    ↓
DeepKernelGaussianGPModel
or
DeepKernelGaussianMixedGPModel
```

The builder uses the existing configurable deep-kernel wrappers. It does not duplicate:

- known `train_Yvar` handling,
- fixed-noise likelihood construction,
- partial/missing multitask observation handling,
- input transforms,
- outcome transforms,
- correlated wide-output construction,
- mixed categorical kernels.

## GP versus DKL

`MaterialSurrogateSpec.kind` records the semantic role of the material surrogate. It intentionally does not select a separate Gaussian implementation.

DKL behavior is established before surrogate construction by applying the shared encoder training policy to the feature extractor's material encoder. This preserves the current implementation model used by CrabNet, Roost, ALIGNN, CHGNet, M3GNet, and MACE.

## Mixed inputs

For `mixed=True`, the builder creates `DeepKernelGaussianMixedGPModel` and requires explicit `cat_dims`.

Categorical process variables remain outside the material feature extractor and continue to be modeled by the mixed categorical kernel. The Phase 6 `MixedProcessLayout` contract remains the canonical way to determine continuous and categorical branches.

CrabNet's learned categorical-embedding DKL remains an explicit specialized variant and is not replaced by this generic builder.

## Latent feature contract

`resolve_material_latent_dim` validates the boundary between a material/process feature extractor and the GP kernel:

- the feature extractor must be a `torch.nn.Module`,
- an explicit `latent_dim` must be positive,
- when `feature_extractor.output_dim` exists it must agree with `latent_dim`,
- when `latent_dim` is omitted, `output_dim` is used.

This prevents family wrappers from silently constructing a GP kernel over the wrong feature width.

## Compatibility guarantees

Phase 8 does **not** change:

- public CrabNet / Roost / ALIGNN / CHGNet / M3GNet / MACE classes,
- existing `model_type` strings,
- GP covariance or likelihood mathematics,
- encoder checkpoint loading,
- frozen / partial / full training semantics,
- mixed categorical-process behavior,
- correlated multitask behavior,
- partial-target semantics,
- known-noise semantics,
- structure feature caches,
- FastAPI / Web schemas,
- acquisition functions,
- saved-model module paths.

Existing family wrappers can migrate to the common builder incrementally after contract tests demonstrate equivalence.

## Tests

Phase 8 contract tests cover:

- surrogate spec validation,
- latent-dimension resolution,
- non-mixed exact-GP construction,
- DKL semantic reuse of the same exact-GP backend,
- fixed known-noise delegation,
- mixed GP construction and categorical dimension preservation,
- invalid mixed/non-mixed categorical configurations.

## Next phase

Phase 9 introduces a pretrained-model capability layer. The capability metadata will distinguish at least:

- representation/embedding availability,
- direct property prediction availability,
- checkpoint/model-name loading,
- device/dtype support,
- residual-GP suitability.

That layer prepares Phase 10's generic pretrained-prediction + GP-residual architecture.
