# Model package ownership

Model code is organized along two axes: **model family** and **cross-cutting strategy**.

## Family-owned concrete models

Likelihood- or task-specific implementations stay under their owning family, for example:

- `regression/gaussian`, `regression/beta`, `regression/gamma`, `regression/count`
- `classification/binary`, `classification/multiclass`
- `ordinal`

Mixed variants that are specific to one concrete model remain with that family.

## Cross-cutting strategy packages

- `multitask/`: correlated outputs/tasks, task-feature adapters, shared ICM/Kronecker infrastructure, and validation.
- `multioutput/`: wrappers that aggregate independently fitted single-output models.
- `multifidelity/`: shared fidelity-axis abstractions and adapters. Concrete Gaussian/Bernoulli/etc. multi-fidelity models remain family-owned unless their implementation is genuinely cross-family.

This distinction is intentional: **multi-output does not imply multi-task correlation**.

## Adding a new strategy

1. Put family-independent mechanics in a dedicated top-level strategy package.
2. Keep likelihood-specific concrete models under the family tree.
3. Depend from family code toward shared strategy code; adapters may explicitly wrap family models when that is their role.
4. Do not add forwarding modules at removed paths. Update imports and the model registry directly.
5. Keep public data-shape contracts documented at the strategy boundary.

This layout lets future multi-fidelity, transfer-learning, and related strategies grow without adding more one-off modules directly under `models/`.
