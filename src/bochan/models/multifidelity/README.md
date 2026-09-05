# Gaussian Multi-Fidelity

`bochan` provides a generic, materials-independent Gaussian multi-fidelity subsystem built on BoTorch / GPyTorch.

## Model configuration

Single-output continuous input:

```python
from bochan.api import ModelConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="normal",
    model_kwargs={
        "fidelity_features": [-1],
        "target_fidelities": {-1: 1.0},
    },
)
```

Multiple fidelity dimensions are supported by the Gaussian surrogate:

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="normal",
    model_kwargs={
        "fidelity_features": [-2, -1],
        "target_fidelities": {
            -2: 1.0,
            -1: 1.0,
        },
    },
)
```

Mixed input:

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="mixed",
    cat_dims=[1, 3],
    model_kwargs={
        "fidelity_features": [-2, -1],
        "target_fidelities": {-2: 1.0, -1: 1.0},
    },
)
```

Negative indices are resolved against the input dimension and duplicate / categorical-overlap indices are rejected. BoTorch constructs one data-fidelity kernel per configured fidelity feature.

## Query fidelity modes

Target-only optimization fixes every configured fidelity dimension to its target:

```python
OptimizeConfig()
```

For a single fidelity dimension, discrete fidelity search enumerates explicitly allowed query fidelities:

```python
OptimizeConfig(fidelity_values=[0.25, 0.5, 1.0])
```

and continuous fidelity search jointly optimizes design variables and fidelity:

```python
OptimizeConfig(optimize_fidelity=True)
```

Phase 59 deliberately keeps discrete and continuous query-fidelity search restricted to exactly one fidelity dimension. Multi-dimensional fidelity assignments / joint fidelity optimization are introduced in Phase 60. `fidelity_values` and `optimize_fidelity=True` remain mutually exclusive.

## Multi-fidelity acquisitions

Single-output models support MFKG and MF-MES:

```python
AcquisitionConfig(name="mfkg")
AcquisitionConfig(name="qmfmes")
```

The configured `target_fidelities` defines the terminal high-fidelity objective used by the acquisition projection. Candidate-time fidelity selection is controlled independently by `OptimizeConfig`.

Independent multi-output models additionally support MF-HVKG and MOMF:

```python
AcquisitionConfig(name="mfhvkg")
AcquisitionConfig(name="momf")
```

MOMF augments the physical objectives with a fidelity/trust objective and applies fidelity-dependent evaluation cost.

## Cost-aware optimization

Known affine evaluation cost can be supplied through `cost_config`:

```python
AcquisitionConfig(
    name="mfkg",
    acqf_kwargs={
        "cost_config": {
            "kind": "affine",
            "fixed_cost": 1.0,
            "fidelity_weights": {-1: 4.0},
        }
    },
)
```

For multiple fidelity dimensions, explicit weights can already be supplied for each fidelity feature. Generalized known-cost configuration is expanded further in Phase 61.

Negative fidelity indices are resolved against the model dimension.

## Multi-output

Independent multi-output MF is supported through `ModelListGP`:

```python
from bochan.api import MultiOutputConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="normal",
    model_kwargs={
        "fidelity_features": [-2, -1],
        "target_fidelities": {-2: 1.0, -1: 1.0},
    },
    multi_output_config=MultiOutputConfig(
        output_names=["property_a", "property_b"],
    ),
)
```

All output submodels must share the same fidelity features, target fidelities, input mode, and categorical dimensions.

## Known observation variance

`train_Yvar` is supported for single- and independent multi-output MF models. When an optimizer was fitted with known observation variance, `tell()` requires `new_Yvar` and preserves the noise data during refit.

## FastAPI

FastAPI fitting uses the same `model_type="multifidelity_gp"` payload. Candidate requests may specify MF settings either in the canonical nested configs or through request convenience fields.

## Current v2 boundary after Phase 59

- one or more continuous fidelity features in Gaussian surrogate models
- continuous or mixed design inputs
- inferred or known observation noise
- independent multi-output via `ModelListGP`
- target-fixed optimization supports all configured fidelity targets
- discrete / continuous query-fidelity search remains single-dimensional until Phase 60
- MFKG / MF-MES single-output; MF-HVKG / MOMF multi-output
- affine known cost model
- CPU and CUDA-compatible tensor/device handling

Planned v2 extensions include multi-dimensional query-fidelity optimization, generalized and learned cost models, correlated multi-output MF, and discrete information sources.
