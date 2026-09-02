# Material model architecture — Phase 5

## Goal

Phase 5 defines one canonical training-policy contract for material encoders while preserving backend-specific layer discovery and all existing public model behavior.

The shared lifecycle is:

- `frozen`: all encoder parameters have `requires_grad=False`; encoder remains in evaluation mode even while the surrogate trains;
- `partial`: the encoder remains globally in evaluation mode, selected backend-specific submodules are trainable and follow the parent model train/eval state;
- `full`: all encoder parameters are trainable and the complete encoder follows the parent model train/eval state.

## Canonical API

`bochan.models.regression.gaussian.materials.common.training` provides:

- `EncoderTrainingMode`
- `EncoderTrainingPolicy`
- `unique_module_parameters`
- `configure_encoder_parameters`
- `apply_encoder_train_mode`
- `apply_encoder_training_policy`

The contract is intentionally backend-neutral. It does not know how a CrabNet transformer layer, Roost graph block, ALIGNN layer, CHGNet atom-convolution layer, M3GNet graph layer, or MACE interaction/product block is discovered.

## Responsibility boundary

Backend family code remains responsible for selecting partial-training modules. The common policy is responsible for what happens after those modules are selected.

```text
backend-specific layer discovery
            ↓
EncoderTrainingPolicy
            ↓
requires_grad state
train/eval lifecycle
            ↓
material GP/DKL feature extractor
```

This prevents backend-specific architecture knowledge from leaking into `materials/common` while removing duplicated lifecycle semantics.

## Compatibility

Phase 5 does not change:

- public material model class names;
- `model_type` strings;
- GP/DKL covariance or likelihood construction;
- encoder checkpoint loading;
- backend-specific partial-layer selection;
- mixed-process handling;
- multitask behavior;
- structure feature-cache policy;
- partial-target or known `train_Yvar` semantics;
- FastAPI/Web request or response schemas;
- acquisition or candidate-generation behavior;
- historical module paths used by saved model artifacts.

The existing `_BaseMaterialGPFeatureExtractor` already provides one shared lifecycle across composition and structure families. Phase 5 formalizes that behavior as the canonical `materials/common` policy contract so later relocation/commonization phases can use a stable API without re-defining frozen/partial/full semantics per family.

## Invariants covered by tests

The Phase 5 contract tests verify that:

1. frozen mode disables gradients for all encoder parameters;
2. frozen mode keeps the encoder in evaluation mode during surrogate training;
3. full mode enables gradients for all encoder parameters;
4. full mode follows parent train/eval state;
5. partial mode enables gradients only for selected modules;
6. partial mode keeps the encoder globally in evaluation mode while selected modules follow parent state;
7. changing policies cannot leave stale trainable parameters behind;
8. shared parameters are deduplicated when multiple selected modules reference them;
9. invalid partial/full/frozen configurations fail early.

## Next phase

Phase 6 will normalize material/process fusion. The main target is to reduce family-specific mixed-wrapper duplication while retaining explicit categorical-process behavior and the current BoTorch optimization contracts.
