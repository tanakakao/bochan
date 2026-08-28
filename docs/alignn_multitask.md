# ALIGNN correlated multitask models

Bochan provides two distinct ways to model multiple continuous material properties from crystal structure and process conditions.

## Independent multi-output vs correlated multitask

| Public `model_type` | Output model | Output dependence | Encoder policy |
| --- | --- | --- | --- |
| `alignn_gp` | one ALIGNN-GP per output in `ModelListGP` | independent | one frozen encoder per output |
| `alignn_dkl` | one ALIGNN-DKL per output in `ModelListGP` | independent | one trainable encoder per output |
| `alignn_multitask` | one wide-output ALIGNN-GP | correlated through `MultitaskKernel` | one shared frozen encoder |
| `alignn_multitask_dkl` | one wide-output ALIGNN-DKL | correlated through `MultitaskKernel` | one shared trainable encoder |

The multitask variants are intended for properties that may share statistical structure, for example strength and conductivity measured for the same crystal/process rows. The learned task covariance is a predictive statistical relationship; it is not evidence of a causal relationship between properties.

## Architecture

For two targets such as `strength` and `conductivity`, `alignn_multitask` keeps the target tensor wide:

```text
crystal structure + process conditions
             |
       shared ALIGNN encoder
             |
       shared latent features
             |
    continuous or mixed base kernel
             |
       MultitaskKernel
        /          \
 strength       conductivity
```

The GP learns a task covariance matrix jointly with the input-space covariance. This differs from the independent `ModelListGP` path, where each output has its own encoder, latent projection, and GP.

## Public model names and mixed inputs

There are no separate public `alignn_mixed_multitask*` model names. As with the existing ALIGNN GP/DKL API, Bochan derives the internal model from `categorical_cols`:

```text
model_type="alignn_multitask"
  no categorical process columns -> ALIGNNMultiTaskGPModel
  categorical process columns    -> ALIGNNMixedMultiTaskGPModel

model_type="alignn_multitask_dkl"
  no categorical process columns -> ALIGNNMultiTaskDKLModel
  categorical process columns    -> ALIGNNMixedMultiTaskDKLModel
```

The crystal structure selector remains feature 0 and is never treated as an ordinary categorical-kernel dimension. It indexes the structure graph bank. Only genuine categorical process variables are placed in `cat_dims`.

For an input table such as:

```text
phase | temperature | furnace | pressure | strength | conductivity
```

with `structure_col="phase"` and `categorical_cols=["furnace"]`, the fitted model coordinates are:

```text
0 phase          -> structure graph selector
1 temperature    -> continuous process
2 furnace        -> categorical process kernel
3 pressure       -> continuous process
```

so `dataset.cat_dims == [0, 2]` while the model receives `cat_dims == [2]`.

## Tabular example

```python
from bochan.tabular import TabularBayesianOptimizer

optimizer = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="alignn_multitask",
    input_cols=["phase", "temperature", "pressure", "furnace"],
    categorical_cols=["furnace"],
    target_cols=["strength", "conductivity"],
    structure_col="phase",
    structure_catalog=structure_catalog,
    bounds={
        "temperature": [850.0, 1150.0],
        "pressure": [0.5, 2.0],
    },
    model_kwargs={
        "checkpoint": checkpoint,
        "latent_dim": 32,
    },
)
optimizer.fit(data)
```

For shared encoder fine-tuning, use:

```python
model_type="alignn_multitask_dkl"
model_kwargs={
    "checkpoint": checkpoint,
    "encoder_training": "partial",  # or "full"
    "latent_dim": 32,
}
```

`alignn_multitask` always freezes the ALIGNN encoder. `alignn_multitask_dkl` fine-tunes the one encoder shared by all output tasks.

## Multi-objective Bayesian optimization

The correlated model exposes a standard BoTorch multi-output posterior, so Bochan's existing multi-objective acquisitions can operate on it. For example:

```python
candidates, acq_value = optimizer.candidate(
    acq_name="nehvi",
    q=3,
    objective_mode="multi_output",
    objective_outputs=["strength", "conductivity"],
    objective_directions=["maximize", "maximize"],
)
```

Structure candidates are still enumerated discretely, categorical process assignments are fixed during each mixed optimization subproblem, and only continuous process dimensions are relaxed by the acquisition optimizer.

## `fit(X, y)`

Wide array targets are supported as well. The fitted `dataset.Y.shape[-1]` is authoritative:

```python
optimizer.fit(X, y)  # y.shape == [n, 2]
```

If target names were not supplied, Bochan assigns stable names `y0`, `y1`, and so on. Correlated multitask model types require at least two continuous target columns.

## FastAPI

The existing ALIGNN routes are reused; no multitask-specific route is required:

```text
POST /api/v1/tabular/alignn/models
POST /api/v1/tabular/alignn/models/{model_id}/predict
POST /api/v1/tabular/alignn/models/{model_id}/candidates
POST /api/v1/tabular/alignn/models/{model_id}/ask
POST /api/v1/tabular/alignn/models/{model_id}/tell
POST /api/v1/tabular/alignn/models/{model_id}/save
POST /api/v1/tabular/alignn/models/load
```

A fit request selects correlated multitask behavior through `model_config.model_type`:

```json
{
  "model_config": {
    "task_type": "multi_objective",
    "model_type": "alignn_multitask",
    "model_kwargs": {
      "latent_dim": 32
    }
  },
  "target_cols": ["strength", "conductivity"]
}
```

`metadata.alignn` distinguishes the dependency model explicitly:

```text
output_dependency = "correlated"
shared_encoder = true
num_outputs = 2
task_kernel = "MultitaskKernel"
```

Independent `alignn_gp` / `alignn_dkl` multi-output models continue to report `output_dependency="independent"`.

## Persistence and observation lifecycle

Multitask models use the same trusted `.bochan.pt` artifact contract as other fitted ALIGNN models. The shared ALIGNN encoder, task covariance parameters, structure graph bank, process category maps, observations, and pending rows are serialized together.

`ask` registers a pending target row with width equal to the number of tasks, and `tell` accepts one value for every target before optional refitting.

## Current scope

- pure-PyTorch ALIGNN structure encoding
- known discrete crystal structures
- continuous and categorical process conditions
- correlated continuous regression targets
- GP and DKL variants
- multi-objective acquisition support
- FastAPI through predict/candidate/ask/tell/save/load

A Web UI is intentionally outside the crystal-structure model scope.
