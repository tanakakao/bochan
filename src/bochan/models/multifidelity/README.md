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

Mixed input:

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="mixed",
    cat_dims=[1, 3],
    model_kwargs={
        "fidelity_features": [-1],
        "target_fidelities": {-1: 1.0},
    },
)
```

Gaussian Multi-Fidelity v1 formally supports one continuous fidelity feature. Negative indices are resolved against the input dimension and duplicate / categorical-overlap indices are rejected.

## Query fidelity modes

Target-only optimization fixes the fidelity to the configured target:

```python
OptimizeConfig()
```

Discrete fidelity search enumerates explicitly allowed query fidelities:

```python
OptimizeConfig(fidelity_values=[0.25, 0.5, 1.0])
```

Continuous fidelity search jointly optimizes design variables and fidelity:

```python
OptimizeConfig(optimize_fidelity=True)
```

`fidelity_values` and `optimize_fidelity=True` are mutually exclusive.

## Multi-fidelity acquisitions

Single-output models support MFKG and MF-MES:

```python
AcquisitionConfig(name="mfkg")
AcquisitionConfig(name="qmfmes")
```

The configured `target_fidelities` defines the terminal high-fidelity objective used by the acquisition projection. Candidate-time fidelity selection is controlled independently by `OptimizeConfig`.

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

Negative fidelity indices are resolved against the model dimension, so `-1` consistently means the last input feature for both model and cost configuration.

## Multi-output

Independent multi-output MF is supported through `ModelListGP`:

```python
from bochan.api import MultiOutputConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    input_type="normal",
    model_kwargs={
        "fidelity_features": [-1],
        "target_fidelities": {-1: 1.0},
    },
    multi_output_config=MultiOutputConfig(
        output_names=["property_a", "property_b"],
    ),
)
```

All output submodels must share the same fidelity feature, target fidelity, input mode, and categorical dimensions. Ordinary multi-objective acquisitions such as EHVI / NEHVI / NParEGO can use the shared fidelity optimization modes. MFKG and MF-MES remain single-output in v1.

## Known observation variance

`train_Yvar` is supported for single- and independent multi-output MF models. When an optimizer was fitted with known observation variance, `tell()` requires `new_Yvar` and preserves the noise data during refit.

## FastAPI

FastAPI fitting uses the same `model_type="multifidelity_gp"` payload. Candidate requests may specify MF settings either in the canonical nested configs or through request convenience fields:

```json
{
  "acquisition_config": {"name": "mfkg"},
  "optimize_fidelity": true,
  "target_fidelity": 1.0,
  "cost_config": {
    "fixed_cost": 1.0,
    "fidelity_weights": {"-1": 4.0}
  }
}
```

## Current v1 boundaries

- one continuous fidelity feature
- Gaussian regression
- continuous or mixed design inputs
- inferred or known observation noise
- independent multi-output via `ModelListGP`
- MFKG / MF-MES for single-output models
- affine known cost model
- target-fixed, discrete-fidelity, and continuous-fidelity candidate optimization
- CPU and CUDA-compatible tensor/device handling

Future extensions can add multiple fidelity dimensions, learned/custom cost models, correlated multi-output MF models, and dedicated multi-objective information-value acquisitions such as MF-HVKG.
