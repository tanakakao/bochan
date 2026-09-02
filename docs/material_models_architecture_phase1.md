# Material model architecture Phase 1

## Goal

Phase 1 freezes the migration design for material-aware Gaussian-process models before any package move or behavioral refactor.

The current implementation places generic deep GP/DKL models and material-domain models in `bochan.models.regression.gaussian.deep`. That package now contains composition models, crystal-structure models, mixed material/process variants, multitask variants, and shared material infrastructure. The migration therefore separates *material domain* from *surrogate family* without changing existing public behavior.

Phase 1 is documentation-only. No model implementation, import path, `model_type`, FastAPI/Web behavior, fitting semantics, acquisition behavior, partial-observation handling, or `train_Yvar` behavior changes in this phase.

## Architectural decision

Material-specific regression code will move under a dedicated package:

```text
src/bochan/models/regression/gaussian/
├── deep/
│   ├── deepgp.py
│   ├── deepkernel.py
│   ├── deepkernel_configurable.py
│   └── deepkerneldeepgp.py
└── materials/
    ├── common/
    ├── composition/
    │   ├── crabnet/
    │   └── roost/
    └── structure/
        ├── alignn/
        ├── chgnet/
        ├── m3gnet/
        └── mace/
```

`deep` remains the home of generic deep GP / deep-kernel machinery. `materials` owns material-domain adapters and their Gaussian-process wrappers.

The primary classification axis is the material input contract:

- `composition`: fixed-vocabulary composition/fraction inputs, optionally combined with continuous process variables.
- `structure`: crystal/atomistic structure bank inputs addressed by discrete structure index, optionally combined with process variables.

This avoids treating all neural material models as generic "deep" models and gives future structure foundation models (for example SevenNet, MatterSim, ORB, NequIP/Allegro-style encoders) a stable extension point.

## Existing reusable infrastructure

The migration must preserve and promote existing abstractions rather than reimplement them.

### Material encoder contract

`src/bochan/composition/encoders/base.py` already defines `MaterialEncoder`, an `nn.Module` contract with an `output_dim` property and encoder-specific `forward(...)` signature. It is used by both composition and structure representations despite currently living under `bochan.composition`.

Target location:

```text
bochan.models.regression.gaussian.materials.common.protocols
```

or, if kept reusable outside Gaussian regression, a future top-level `bochan.materials.encoders` package. Phase 2 will choose the least disruptive physical move while preserving the current `bochan.composition.MaterialEncoder` import as a compatibility re-export.

### Material/process fusion

`bochan.composition.MaterialProcessFusion` and `build_material_process_fusion(...)` are already consumed by the Gaussian material feature extractors.

Target architectural role:

```text
materials/common/fusion.py
```

The existing public composition import remains available during migration.

### Shared GP feature extraction

`src/bochan/models/regression/gaussian/deep/material.py` already provides:

- `CompositionMaterialInputTransform`
- `_BaseMaterialGPFeatureExtractor`
- `MaterialGPFeatureExtractor`
- encoder training-mode handling (`frozen`, `partial`, `full`)
- material/process fusion
- latent projection
- composition validation and process-only normalization

Target locations:

```text
materials/common/feature_extractor.py
materials/composition/input_transform.py
materials/composition/feature_extractor.py
```

### Shared structure feature extraction

`src/bochan/models/regression/gaussian/deep/structure.py` already provides:

- structure-bank validation
- structure-index input validation
- process-only normalization
- `_StructureGPFeatureExtractor`
- frozen structure embedding cache
- encoder parameter-version invalidation
- pickle-safe non-persistent cache handling
- unique-structure batching

Target locations:

```text
materials/structure/input.py
materials/structure/feature_extractor.py
```

This common structure layer is the correct basis for ALIGNN, CHGNet, M3GNet, MACE, and future atomistic encoders.

## Current package inventory

### Generic deep models: remain under `gaussian/deep`

| Current file | Phase-1 classification | Planned action |
| --- | --- | --- |
| `deep/deepgp.py` | generic deep GP | remain in `deep` |
| `deep/deepkernel.py` | generic DKL infrastructure | remain in `deep` |
| `deep/deepkernel_configurable.py` | generic configurable DKL GP | remain in `deep` |
| `deep/deepkerneldeepgp.py` | generic deep-kernel/deep-GP composition | remain in `deep` |
| `deep/multitask_fixed_noise.py` | generic multitask/fixed-noise support used by material models | move only if material-specific ownership is proven; otherwise remain generic |

### Shared material infrastructure

| Current file | New architectural location | Notes |
| --- | --- | --- |
| `deep/material.py` | `materials/common` + `materials/composition` | split common feature-extractor policy from composition-specific transforms |
| `deep/structure.py` | `materials/structure` | shared structure-bank and structure-feature infrastructure |
| `composition/encoders/base.py` | material encoder common contract | preserve current import compatibility |
| `composition/encoders/fusion.py` | material/process fusion common layer | not composition-specific in behavior |

## Composition model family

### CrabNet

Current Gaussian files:

```text
deep/crabnet.py
deep/crabnet_mixed.py
deep/crabnet_mixed_dkl.py
deep/crabnet_multitask.py
```

Current encoder:

```text
composition/encoders/crabnet.py
```

Current public Gaussian models include:

- `CrabNetGPModel`
- `CrabNetDKLModel`
- `CrabNetMixedGPModel`
- `CrabNetMixedDKLModel`
- `CrabNetMultiTaskGPModel`
- `CrabNetMultiTaskDKLModel`
- `CrabNetMixedMultiTaskGPModel`
- `CrabNetMixedMultiTaskDKLModel`

Target package:

```text
materials/composition/crabnet/
├── __init__.py
├── encoder.py          # initially compatibility/adaptation boundary
├── gp.py
├── mixed.py            # transitional; later folded into common fusion
└── multitask.py        # transitional; later common surrogate layer
```

The existing `crabnet.py` already resolves/freeze-configures `CrabNetEncoder`, supports checkpoint loading, partial/full encoder training for DKL, and uses `MaterialGPFeatureExtractor`. Those responsibilities should be separated in later phases, but Phase 3 first performs a behavior-preserving relocation.

### Roost

Current Gaussian file:

```text
deep/roost.py
```

Current encoder:

```text
composition/encoders/roost.py
```

Current public Gaussian models:

- `RoostGPModel`
- `RoostDKLModel`

Target package:

```text
materials/composition/roost/
├── __init__.py
├── encoder.py
└── gp.py
```

Roost already uses the same composition `MaterialGPFeatureExtractor` contract as CrabNet. Its fine-tuning logic differs because Aviary/Roost exposes element embeddings, graph/message-passing modules, and pooling modules. Those backend-specific layer-selection rules stay with the Roost family; only the common training policy moves to `materials/common`.

## Structure model family

### ALIGNN

Current Gaussian files:

```text
deep/alignn.py
deep/alignn_mixed.py
deep/alignn_multitask.py
```

Related structure/encoder files:

```text
composition/encoders/alignn.py
structure/alignn.py
```

Current public Gaussian models:

- `ALIGNNGPModel`
- `ALIGNNDKLModel`
- `ALIGNNMixedGPModel`
- `ALIGNNMixedDKLModel`
- `ALIGNNMultiTaskGPModel`
- `ALIGNNMultiTaskDKLModel`
- `ALIGNNMixedMultiTaskGPModel`
- `ALIGNNMixedMultiTaskDKLModel`

Target package:

```text
materials/structure/alignn/
├── __init__.py
├── encoder.py
├── adapter.py
├── gp.py
├── mixed.py            # transitional
└── multitask.py        # transitional
```

ALIGNN uses a structure-graph bank and the shared `_StructureGPFeatureExtractor`. Existing graph conversion/adapter responsibilities must not be mixed with GP surrogate code during migration.

### CHGNet

Current Gaussian files:

```text
deep/chgnet.py
deep/chgnet_multitask.py
```

Current encoder:

```text
composition/encoders/chgnet.py
```

Current public Gaussian models:

- `CHGNetGPModel`
- `CHGNetDKLModel`
- `CHGNetMixedGPModel`
- `CHGNetMixedDKLModel`
- `CHGNetMultiTaskGPModel`
- `CHGNetMultiTaskDKLModel`
- `CHGNetMixedMultiTaskGPModel`
- `CHGNetMixedMultiTaskDKLModel`

Target package:

```text
materials/structure/chgnet/
├── __init__.py
├── encoder.py
├── gp.py
└── multitask.py        # transitional
```

CHGNet contains both continuous and mixed/categorical process wrappers in the same base Gaussian module. The relocation must retain that API shape first; later common fusion/surrogate phases may reduce duplication.

### M3GNet

Current Gaussian files:

```text
deep/m3gnet.py
deep/m3gnet_multitask.py
```

Current encoder:

```text
composition/encoders/m3gnet.py
```

Current public Gaussian models:

- `M3GNetGPModel`
- `M3GNetDKLModel`
- `M3GNetMixedGPModel`
- `M3GNetMixedDKLModel`
- `M3GNetMultiTaskGPModel`
- `M3GNetMultiTaskDKLModel`
- `M3GNetMixedMultiTaskGPModel`
- `M3GNetMixedMultiTaskDKLModel`

Target package:

```text
materials/structure/m3gnet/
├── __init__.py
├── encoder.py
├── gp.py
└── multitask.py        # transitional
```

The default MatGL M3GNet model-name choice and backend-specific graph-layer fine-tuning rules remain model-family concerns.

### MACE

Current Gaussian files:

```text
deep/mace.py
deep/mace_mixed.py
deep/mace_multitask.py
```

Current encoder:

```text
composition/encoders/mace.py
```

Current public Gaussian models:

- `MACEGPModel`
- `MACEDKLModel`
- `MACEMixedGPModel`
- `MACEMixedDKLModel`
- `MACEMultiTaskGPModel`
- `MACEMultiTaskDKLModel`
- `MACEMixedMultiTaskGPModel`
- `MACEMixedMultiTaskDKLModel`

Target package:

```text
materials/structure/mace/
├── __init__.py
├── encoder.py
├── gp.py
├── mixed.py            # transitional
└── multitask.py        # transitional
```

MACE-specific model loading (`model_name`), interaction/product block selection, pooling/head configuration, and fine-tuning selection remain family-specific. Future pretrained direct prediction / zero-shot / residual-GP support belongs in this model family plus shared `materials/common/pretrained.py` and `materials/common/residual.py` contracts, not in generic `deep`.

## Encoder-location issue discovered in Phase 1

The current package name `bochan.composition.encoders` is broader than its actual contents. It includes:

- composition encoders: CrabNet, Roost
- structure encoders: ALIGNN, CHGNet, M3GNet, MACE
- generic material encoder base contract
- generic material/process fusion

Therefore the long-term architecture must not simply move Gaussian wrappers while leaving all encoder ownership permanently under `composition`.

Migration rule:

1. preserve `bochan.composition.*` public imports during the initial relocation;
2. establish canonical material-family ownership under `materials/composition` and `materials/structure`;
3. only after compatibility tests pass, turn the old encoder modules into re-export shims or move reusable encoder contracts to a neutral top-level material package.

The exact physical ownership of encoders is intentionally deferred to Phase 2/3 because those encoders may also be consumed outside Gaussian regression.

## Target common layers

The following responsibilities should become common *after* relocation proves behavior equivalence.

### `materials/common/protocols.py`

Owns neutral contracts/capabilities such as:

- `MaterialEncoder`
- material domain metadata (`composition`, `structure`)
- future pretrained capabilities

### `materials/common/fusion.py`

Owns material/process feature fusion. It must support at least current concatenation semantics and preserve gradient flow through both material and process features.

### `materials/common/training.py`

Owns generic encoder training-mode policy:

- frozen
- partial
- full

Backend-specific decisions about *which* layers to unfreeze remain in model-family modules.

### `materials/common/surrogate.py`

Future layer for shared GP/DKL wrapper mechanics. This is deliberately not introduced during the initial package move because it has the highest regression risk.

### `materials/common/multitask.py`

Future adapter for independent/correlated multitask behavior, including current observation-aware semantics:

- partial targets
- known `train_Yvar`
- failed/pending experiment lifecycle
- output-specific training masks

### `materials/common/pretrained.py`

Future capability contract for models that can supply pretrained material representations and/or direct physical-property predictions.

### `materials/common/residual.py`

Future shared residual-surrogate composition:

```text
pretrained/base prediction + GP residual -> corrected posterior
```

This should be reusable across MACE, CHGNet, M3GNet and future structure foundation models.

### `materials/common/registry.py`

Future model-family registry separating material family from surrogate mode. Public names such as `mace_gp` and `crabnet_gp` remain stable while internally resolving to:

```text
family=mace, domain=structure, surrogate=gp
family=crabnet, domain=composition, surrogate=gp
```

## Compatibility policy

The migration is governed by the following invariants.

1. Existing `model_type` strings remain unchanged.
2. Existing exported model class names remain unchanged.
3. Existing FastAPI and Web request/response schemas remain unchanged.
4. Existing optimizer/model factory call sites must not require users to import from the new package.
5. Existing direct imports from `bochan.models.regression.gaussian.deep.<material-model>` remain functional through compatibility re-exports for at least the migration window.
6. Existing `bochan.composition` encoder imports remain functional.
7. GP posterior, DKL training, mixed input, multitask, known-noise (`train_Yvar`), partial-target and failed/pending semantics must be behavior-equivalent before and after relocation.
8. Optional material-model dependencies must remain lazily/optionally imported; importing core `bochan` must not force MACE/CHGNet/MatGL/ALIGNN dependencies to be installed.
9. Pickle/model-save compatibility must be considered before old module paths are removed. Compatibility modules should remain until serialized artifacts can be migrated or version-gated safely.

## Dependency-direction rule

The new package must avoid cyclic ownership.

Preferred direction:

```text
material encoder/backend
        ↓
materials common feature/fusion contract
        ↓
generic gaussian deep-kernel/surrogate infrastructure
        ↓
public material GP/DKL wrapper
```

Generic `gaussian.deep` code must not import concrete CrabNet/MACE/etc. families.

Material-family packages may depend on generic Gaussian infrastructure.

## Migration map

| Current path | Target canonical path | Phase | Compatibility strategy |
| --- | --- | ---: | --- |
| `deep/material.py` | `materials/common/*` + `materials/composition/*` | 2-3 | old module re-exports |
| `deep/structure.py` | `materials/structure/*` | 2/4 | old module re-exports |
| `deep/crabnet.py` | `materials/composition/crabnet/gp.py` | 3 | shim |
| `deep/crabnet_mixed.py` | `materials/composition/crabnet/mixed.py` | 3 | shim |
| `deep/crabnet_mixed_dkl.py` | `materials/composition/crabnet/mixed.py` | 3 | shim |
| `deep/crabnet_multitask.py` | `materials/composition/crabnet/multitask.py` | 3 | shim |
| `deep/roost.py` | `materials/composition/roost/gp.py` | 3 | shim |
| `deep/alignn.py` | `materials/structure/alignn/gp.py` | 4 | shim |
| `deep/alignn_mixed.py` | `materials/structure/alignn/mixed.py` | 4 | shim |
| `deep/alignn_multitask.py` | `materials/structure/alignn/multitask.py` | 4 | shim |
| `deep/chgnet.py` | `materials/structure/chgnet/gp.py` | 4 | shim |
| `deep/chgnet_multitask.py` | `materials/structure/chgnet/multitask.py` | 4 | shim |
| `deep/m3gnet.py` | `materials/structure/m3gnet/gp.py` | 4 | shim |
| `deep/m3gnet_multitask.py` | `materials/structure/m3gnet/multitask.py` | 4 | shim |
| `deep/mace.py` | `materials/structure/mace/gp.py` | 4 | shim |
| `deep/mace_mixed.py` | `materials/structure/mace/mixed.py` | 4 | shim |
| `deep/mace_multitask.py` | `materials/structure/mace/multitask.py` | 4 | shim |
| `composition/encoders/crabnet.py` | canonical composition encoder ownership | 3+ | preserve old import |
| `composition/encoders/roost.py` | canonical composition encoder ownership | 3+ | preserve old import |
| `composition/encoders/alignn.py` | canonical structure encoder ownership | 4+ | preserve old import |
| `composition/encoders/chgnet.py` | canonical structure encoder ownership | 4+ | preserve old import |
| `composition/encoders/m3gnet.py` | canonical structure encoder ownership | 4+ | preserve old import |
| `composition/encoders/mace.py` | canonical structure encoder ownership | 4+ | preserve old import |
| `composition/encoders/base.py` | neutral material encoder contract | 2+ | preserve old import |
| `composition/encoders/fusion.py` | neutral material fusion contract | 2+ | preserve old import |

## Revised implementation phases

### Phase 2: package skeleton and neutral contracts

- add `gaussian/materials/{common,composition,structure}` packages;
- establish canonical common import surface;
- avoid concrete model moves;
- add import/optional-dependency contract tests.

### Phase 3: composition relocation

- move CrabNet and Roost Gaussian wrappers;
- preserve `gaussian.deep` compatibility modules;
- preserve `bochan.composition` encoder imports;
- add old/new import equivalence tests;
- no surrogate behavior refactor yet.

### Phase 4: structure relocation

- move ALIGNN, CHGNet, M3GNet and MACE wrappers;
- relocate structure-common feature infrastructure;
- preserve structure caches and serialization behavior;
- add old/new import equivalence tests.

### Phase 5: encoder/training policy normalization

- promote neutral encoder contract;
- centralize frozen/partial/full lifecycle;
- retain backend-specific layer discovery per family.

### Phase 6: material/process fusion normalization

- centralize mixed feature composition;
- keep categorical handling explicit where required;
- reduce per-family mixed-wrapper duplication.

### Phase 7: multitask normalization

- reuse common material encoders with standard observation-aware multitask paths;
- preserve partial target and known `train_Yvar` semantics.

### Phase 8: GP/DKL surrogate normalization

- reduce duplicated family wrappers only after package/import stability is established;
- retain public model classes as thin configured wrappers if useful for compatibility.

### Phase 9: pretrained capability layer

- define representation vs direct-prediction capabilities;
- cover optional model loading, device/dtype and checkpoint policy.

### Phase 10: residual GP layer

- generalize pretrained prediction + GP correction;
- make it reusable across compatible structure families.

### Phase 11: family registry

- map stable public `model_type` names to family/domain/surrogate specifications;
- remove scattered family-specific factory branching where safe.

### Phase 12: deprecation/compatibility cleanup

- keep old imports as explicit shims;
- only introduce deprecation warnings after downstream callers and serialization are audited.

### Phase 13: contract-test consolidation

- parameterized encoder contracts;
- GP/DKL shape/device/dtype contracts;
- mixed/process contracts;
- multitask/known-noise/partial-target contracts;
- optional dependency/import contracts;
- save/load compatibility tests.

## Phase 1 acceptance criteria

Phase 1 is complete when:

- every material-specific file currently in `gaussian/deep` has a target family;
- generic deep GP/DKL files are explicitly excluded from the material move;
- CrabNet/Roost are classified as composition models;
- ALIGNN/CHGNet/M3GNet/MACE are classified as structure models;
- existing reusable `MaterialEncoder`, fusion, composition feature extraction and structure feature extraction are recognized as migration primitives;
- compatibility constraints for imports, model types, APIs, optional dependencies and serialization are explicit;
- later commonization work is sequenced after behavior-preserving relocation.

No runtime code changes are required for Phase 1.