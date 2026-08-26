# ALIGNN Phase 3: tabular structure optimization

Phase 3 connects the Phase-1 ALIGNN GP/DKL models and the Phase-2 crystal-structure adapters to the canonical pandas tabular optimizer.

## Data contract

The user-facing DataFrame keeps a stable structure identifier rather than a continuous structure coordinate:

```text
phase     temperature    pressure    property
alpha        950            1.0        ...
beta        1050            1.2        ...
```

A `structure_catalog` maps each identifier to its crystal structure:

```python
structure_catalog = {
    "alpha": alpha_structure,
    "beta": beta_structure,
    "gamma": gamma_structure,
}
```

The catalog insertion order defines the canonical model indices:

```text
alpha -> 0
beta  -> 1
gamma -> 2
```

The fitted ALIGNN tensor is always ordered as:

```text
[structure_index, process_1, process_2, ...]
```

The structure index is discrete. It is never treated as a differentiable continuous design variable.

## Python API

```python
from bochan.tabular import TabularBayesianOptimizer

optimizer = TabularBayesianOptimizer(
    task_type="regression",
    model_type="alignn_gp",
    input_cols=["phase", "temperature", "pressure"],
    target_cols="property",
    structure_col="phase",
    structure_catalog={
        "alpha": alpha_structure,
        "beta": beta_structure,
        "gamma": gamma_structure,
    },
    bounds={
        "temperature": [800.0, 1300.0],
        "pressure": [0.5, 2.0],
    },
    model_kwargs={
        "checkpoint": checkpoint,
        "latent_dim": 32,
    },
)
optimizer.fit(frame)
```

`structure_col` is automatically encoded using the existing tabular category-map machinery. Candidate DataFrames decode the integer model coordinate back to the original structure ID.

`structure_graph_builder=` may be supplied when graph construction needs a custom or preconfigured `ALIGNNGraphBuilder`. Without it, the Phase-2 default builder is used.

## Candidate generation

The structure selector is automatically expanded into BoTorch fixed-feature assignments. For three structures the optimizer receives the equivalent of:

```python
fixed_features_list = [
    {0: 0.0},
    {0: 1.0},
    {0: 2.0},
]
```

The core optimizer dispatch therefore switches to the mixed fixed-feature path while gradients remain available for all continuous process dimensions.

```python
candidates, acq_value = optimizer.candidate(
    acq_name="logei",
    q=3,
)
```

A subset of structures can be explored explicitly:

```python
candidates, acq_value = optimizer.candidate(
    acq_name="logei",
    q=2,
    structure_ids=["alpha", "gamma"],
)
```

Users should not supply `fixed_features_list` for the structure coordinate directly. The tabular ALIGNN adapter owns that mapping so structure IDs and graph-bank indices cannot drift apart.

## ALIGNN-DKL

Use `alignn_dkl` when the structure encoder should be fine-tuned jointly with the GP representation:

```python
optimizer = TabularBayesianOptimizer(
    model_type="alignn_dkl",
    ...,
    model_kwargs={
        "checkpoint": checkpoint,
        "encoder_training": "partial",
    },
)
```

`encoder_training="partial"` maps to one final trainable graph-convolution block. `encoder_training="full"` maps to full representation-backbone fine-tuning. `alignn_gp` always freezes the encoder.

## Current Phase-3 scope

Phase 3 intentionally keeps the first production contract narrow:

- regression with one target
- one crystal-structure selector
- continuous process variables
- explicit bounds for every process variable
- structure enumeration through fixed features
- JARVIS / ASE / pymatgen / mapping / trusted local file structures through the Phase-2 adapter

Not yet included in this phase:

- composition + generated structure joint optimization
- categorical process variables
- multi-output ALIGNN
- structure generation or topology mutation
- FastAPI file-upload persistence and Web selectors
- saved structure catalogs/artifacts

Those can be layered on top of the canonical structure-index contract without changing the ALIGNN model tensor layout.
