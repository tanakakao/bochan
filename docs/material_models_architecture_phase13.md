# Material model architecture Phase 13

Phase 13 closes the staged material-model architecture migration by consolidating the cross-cutting contract tests established in Phases 2-12.

## Goal

The purpose of this phase is not another implementation move. It establishes a regression boundary around the architecture that now exists:

- canonical `gaussian.materials` package surface;
- historical `gaussian.deep.*` serialization paths;
- neutral material encoder and fusion contracts;
- frozen / partial / full encoder training policy;
- mixed material/process dimension layout;
- correlated vs independent multitask semantics;
- GP / DKL surrogate construction;
- pretrained capability metadata;
- residual-GP preprocessing and posterior boundary;
- family registry metadata and lazy class resolution.

## Consolidated integration contract

`tests/models/regression/gaussian/materials/test_architecture_contract_phase13.py` verifies that the independent contracts compose through the public API rather than only passing in isolation.

The integration contract checks:

1. Legacy `bochan.composition.encoders` contracts and canonical material contracts remain the exact same Python objects.
2. Mixed input layout preserves material coordinates, numeric process variables, and categorical process variables as distinct responsibilities.
3. Multitask and surrogate specifications can be composed without importing a concrete material backend.
4. Residual target construction preserves NaN observations so observation-aware Gaussian code remains the sole owner of missing-target semantics.
5. Residual-GP capability requires verified direct prediction support.
6. The family registry advertises the exact current variant matrix: CrabNet / ALIGNN / CHGNet / M3GNet / MACE expose the eight GP/DKL/mixed/multitask variants, while Roost currently exposes GP and DKL only.
7. Registry entries remain conservative: no family advertises direct prediction or residual-GP support until a concrete `DirectMaterialPredictor` adapter is verified.
8. Registry, canonical namespace, and historical namespace resolve to the same GP/DKL class objects.
9. Serialization-protected classes retain historical `gaussian.deep.*` `__module__` values and class pickle round trips continue to resolve them.

## Existing focused contract suites

The Phase 13 integration test complements rather than replaces the focused tests already added during the migration:

- `test_encoder_training_policy.py`
- `test_process_layout.py`
- `test_multitask_contracts.py`
- `test_surrogate_contracts.py`
- pretrained capability contract tests from Phase 9
- residual-GP contract tests from Phase 10
- registry contract tests from Phase 11
- `test_compatibility_phase12.py`
- composition and structure canonical import contract tests from Phases 3-4

Focused tests remain the correct place for detailed behavior. Phase 13 exists to catch architectural drift between those layers.

## Required regression invariants

Future material-model work should preserve the following unless an explicit compatibility migration is planned:

- public `model_type` strings remain stable;
- canonical material classes remain compatible with historical imports;
- historical module paths needed by saved models remain importable;
- optional backend packages are not required merely to import neutral material metadata;
- partial observations and `train_Yvar` continue to flow through the established observation-aware Gaussian path;
- categorical process variables retain mixed-kernel semantics;
- correlated multitask models remain distinct from independent multi-output models;
- GP vs DKL remains primarily an encoder-training-policy distinction over the shared Gaussian backend;
- residual-GP uncertainty comes from the residual GP while the pretrained baseline is deterministic unless a future probabilistic baseline contract explicitly changes this;
- family registry capability declarations must reflect verified Bochan adapters, not only features available somewhere in a third-party package.

## Migration status after Phase 13

The architectural migration is considered stable at the package/contract level. Concrete material implementations intentionally remain under `gaussian.deep.*` because their historical module names are part of the current serialization contract. The canonical `gaussian.materials.*` namespaces and registry are the preferred discovery/import surface for new code.

A future physical implementation move should be treated as a separate serialization migration and should not be bundled into ordinary model feature work.
