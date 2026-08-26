# ALIGNN Phase 7: independent multi-output regression

Phase 7 extends the structure-aware ALIGNN GP/DKL stack to multiple continuous
outputs while preserving the existing public model names and FastAPI routes.

## 1. Model semantics

When `target_cols` contains more than one continuous target, bochan builds one
independent ALIGNN regression model per target and wraps them in BoTorch
`ModelListGP`:

```text
                         +-> ALIGNN-GP/DKL for strength -----+
structure + process X --+                                   +-> ModelListGP
                         +-> ALIGNN-GP/DKL for conductivity -+
```

The output models share the same immutable crystal graph bank and structure-ID
mapping, but their learnable model state is independent:

- separate ALIGNN encoder object per output;
- separate projection and GP parameters per output;
- separate outcome transform per output;
- separate frozen representation cache for `alignn_gp`;
- separate fine-tuning state for `alignn_dkl`.

This is an **independent-output** model, not a correlated multitask GP. It does
not directly learn a covariance between `strength` and `conductivity`. This is a
conservative first-line multi-output model and integrates directly with the
existing bochan multi-objective acquisition stack.

## 2. Public Tabular API

No new model type is introduced. Multiple targets are detected from
`target_cols`:

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="alignn_gp",
    input_cols=["phase", "temperature", "pressure"],
    target_cols=["strength", "conductivity"],
    structure_col="phase",
    structure_catalog=structure_catalog,
    bounds={
        "temperature": [850.0, 1150.0],
        "pressure": [0.5, 2.0],
    },
)
bo.fit(df)
```

For two or more targets, the fitted parent model is exposed as a
`multi_objective` bundle containing independent regression submodels. The
multi-output structure is derived automatically; do not pass
`multi_output_config` manually for tabular ALIGNN.

A single target retains the existing single-output behavior.

## 3. Mixed process variables

Categorical process variables continue to use the Phase-3 contract:

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="alignn_gp",
    input_cols=["phase", "temperature", "furnace", "pressure", "atmosphere"],
    categorical_cols=["furnace", "atmosphere"],
    target_cols=["strength", "conductivity"],
    structure_col="phase",
    structure_catalog=structure_catalog,
    bounds={
        "temperature": [850.0, 1150.0],
        "pressure": [0.5, 2.0],
    },
)
```

The model tensor contract remains:

```text
0  structure_index       graph selector, never a categorical GP dimension
1  temperature           continuous
2  furnace               categorical process dimension
3  pressure              continuous
4  atmosphere            categorical process dimension
```

Every output submodel uses the same process categorical dimensions. Candidate
optimization still admits only the currently observed **joint** process-category
assignments rather than constructing an unseen Cartesian product.

## 4. Multi-objective Bayesian optimization

Multi-output ALIGNN can be used with the normal bochan EHVI / NEHVI acquisition
flow. For example:

```python
import torch
from bochan.api import DataContext

candidates, acq_value = bo.candidate(
    acq_name="nehvi",
    q=3,
    objective_mode="multi_output",
    objective_outputs=["strength", "conductivity"],
    objective_directions=["maximize", "maximize"],
    data_context=DataContext(
        X_baseline=bo.train_X,
        Y_baseline=bo.train_Y,
        ref_point=torch.tensor(
            [90.0, 1.8],
            dtype=bo.train_Y.dtype,
            device=bo.train_Y.device,
        ),
    ),
)
```

The reference point must have the same objective dimension as the selected
outputs and must be defined in the objective space expected by the acquisition
function.

A multi-output model can also optimize a single selected property by using a
scalar objective, for example `objective_output="strength"`.

## 5. ALIGNN-DKL

`model_type="alignn_dkl"` is also supported with multiple targets. Each output
receives its own trainable ALIGNN encoder copy:

```text
strength      -> encoder A -> projection A -> GP A
conductivity  -> encoder B -> projection B -> GP B
```

This isolation is intentional. Sharing one trainable encoder between independent
GP submodels would make per-output optimization and serialization semantics
ambiguous. A shared correlated representation model can be introduced later as
a distinct multitask architecture if required.

The existing training controls remain:

```python
model_kwargs={"encoder_training": "partial"}
```

or

```python
model_kwargs={"encoder_training": "full"}
```

and are applied independently to every output submodel.

## 6. FastAPI

The existing endpoint is unchanged:

```text
POST /api/v1/tabular/alignn/models
```

A multi-output request uses a target list:

```json
{
  "input_cols": ["phase", "temperature", "pressure"],
  "target_cols": ["strength", "conductivity"],
  "structure_col": "phase",
  "structure_catalog": {},
  "bounds": {
    "temperature": [850.0, 1150.0],
    "pressure": [0.5, 2.0]
  },
  "model_config": {
    "task_type": "regression",
    "model_type": "alignn_gp"
  }
}
```

The response `metadata.alignn` reports:

```text
multi_output       true
num_outputs        number of target columns
output_names       target column names
output_dependency  independent
output_models      per-output model / encoder metadata
```

`predict`, `candidates`, `ask`, `tell`, `save`, and `load` keep their existing
routes. `ask` registers one pending row with an unobserved value for every output,
and `tell` accepts the target columns as the multi-output observation.

## 7. Persistence

The existing `.bochan.pt` artifact format is reused. A multi-output artifact
retains:

- `ModelListGP` and all independent ALIGNN submodels;
- target names and target order;
- structure catalog / graph bank;
- category maps;
- current observation state, including pending rows;
- Phase-6 frozen representation-cache policy.

Derived structure-feature cache tensors remain excluded from artifacts and are
rebuilt lazily after loading.

## 8. What is not implemented

Phase 7 does not add a correlated ALIGNN multitask model. In particular, it does
not yet provide an ALIGNN equivalent of `KroneckerMultiTaskGP` or a learned
cross-output task covariance. Such a model should use a distinct architecture and
model contract rather than silently changing `alignn_gp` / `alignn_dkl` semantics.
