# Gaussian Multi-Fidelity v2

This document summarizes the generic Gaussian Multi-Fidelity stack completed in Phases 43–66. The subsystem is materials-independent and is intended for ordinary tabular Bayesian optimization, mixed variables, multiple fidelities, correlated outputs, and discrete information sources.

## 1. Continuous fidelity

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

For several continuous fidelity dimensions, provide all columns in `fidelity_features`. The target-fidelity projection, discrete enumeration, and continuous joint optimization resolve negative feature indices against the final model dimension.

## 2. Mixed input

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

Categorical assignments and discrete fidelity assignments are crossed before `optimize_acqf_mixed` is called. A fidelity column must not also be declared categorical.

## 3. Independent and correlated multi-output MF

The historical multi-output path creates one MF GP per physical output and combines them with `ModelListGP`. This is the most flexible path when outputs have missing values or different observation sets.

For block-design data where every output is observed at every `X`, Phase 64 adds correlated output modeling:

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="multifidelity_gp",
    model_kwargs={
        "fidelity_features": [-1],
        "target_fidelities": {-1: 1.0},
        "correlated_outputs": True,
    },
)
```

The correlated path uses `KroneckerMultiTaskGP` with an ICM task covariance and a fidelity-aware data covariance. It therefore learns output correlation instead of assuming conditional independence. In v2 this path is continuous-input, fully observed block-design, and inferred-noise only.

The explicit alias `model_type="correlated_multifidelity_gp"` selects the same correlated surrogate.

## 4. Discrete information sources

When simulation A, simulation B, and experiment are distinct sources without a natural continuous ordering, use the source model instead of forcing them into a continuous fidelity coordinate.

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="multisource_gp",
    model_kwargs={
        "source_feature": -1,
        "source_values": [0, 1, 2],
        "target_source": 2,
        "source_names": {
            0: "simulation_a",
            1: "simulation_b",
            2: "experiment",
        },
    },
)
```

`information_source_gp` is an alias. The source axis is modeled by `MultiTaskGP` / ICM, so source ids are task labels rather than ordered physical fidelity values. The model exposes the common MF metadata bridge, allowing the existing target projection and discrete optimizer infrastructure to be reused.

## 5. Query-fidelity optimization

Discrete fidelity or source search:

```python
from bochan.api import OptimizeConfig

opt_config = OptimizeConfig(
    fidelity_values=[0.25, 0.5, 1.0],
)
```

For multiple fidelity dimensions:

```python
opt_config = OptimizeConfig(
    fidelity_values={
        -2: [0.25, 0.5, 1.0],
        -1: [0.5, 1.0],
    },
)
```

If only physically valid combinations should be searched, provide `fidelity_assignments` instead of the Cartesian product.

Continuous joint optimization keeps all configured fidelity dimensions free:

```python
opt_config = OptimizeConfig(optimize_fidelity=True)
```

These query modes are mutually exclusive.

## 6. Cost models

Known affine cost remains the backward-compatible default:

```python
from bochan.models.multifidelity import FidelityCostConfig

cost = FidelityCostConfig(
    kind="affine",
    fixed_cost=1.0,
    fidelity_weights={-1: 9.0},
)
```

Other v2 modes are:

- `kind="fixed"`: one constant evaluation cost.
- `kind="callable"`: Python callable `X -> cost` for nonlinear known cost.
- `kind="learned_gp"`: GP model of observed cost, optionally in log-cost space.
- `kind="discrete_source"`: explicit cost per information-source id.

`callable` is a Python-API concept and is intentionally not serialized as executable code through FastAPI.

## 7. Acquisition functions

The v2 Gaussian MF stack supports:

- MFKG / qMFKG for cost-aware single-objective knowledge gradient.
- MF-MES / qMF-MES for cost-aware max-value entropy search.
- MF-HVKG / qMF-HVKG for multi-objective hypervolume knowledge gradient.
- MOMF / qMOMF for multi-objective multi-fidelity EHVI-style optimization with a fidelity/trust objective.

MF-HVKG and MOMF work with both independent `ModelListGP` multi-output models and Phase 64 correlated multi-output models. Phase 66 no longer relies exclusively on `ModelBundle.metadata["multi_output"]`; the concrete model's `num_outputs` contract is also recognized.

## 8. Benchmark protocol

Do not compare MF strategies by iteration count alone. An iteration at high fidelity can cost orders of magnitude more than one at low fidelity. Phase 66 provides common cost-normalized traces in `bochan.models.multifidelity`.

Single objective:

```python
from bochan.models.multifidelity import single_objective_cost_trace

trace = single_objective_cost_trace(
    strategy="mfkg",
    values=observed_objective,
    costs=observed_cost,
    maximize=True,
)
```

Report **best objective vs cumulative cost**.

Multi-objective:

```python
from bochan.models.multifidelity import multi_objective_cost_trace

trace = multi_objective_cost_trace(
    strategy="mfhvkg",
    Y=observed_Y,
    costs=observed_cost,
    ref_point=ref_point,
)
```

Report **hypervolume vs cumulative cost**. When a trusted reference or oracle hypervolume is available, also report **inference hypervolume regret vs cumulative cost** with `inference_hv_regret_cost_trace`.

A recommended benchmark compares at least:

1. high-fidelity-only BO,
2. MFKG,
3. MF-MES,
4. MF-HVKG for multi-objective problems,
5. MOMF for multi-objective problems.

Use identical initial target-fidelity observations where possible, identical random seeds, the same total cost budget, and multiple replications. Aggregate traces on a common cost grid only after each run has been represented on its native cumulative-cost axis.

## 9. v2 scope boundaries

The completed v2 stack intentionally keeps the following boundaries explicit:

- correlated MF requires fully observed block-design outputs;
- correlated mixed/categorical MF remains outside Phase 64–66;
- MOMF currently uses exactly one fidelity/trust dimension;
- multi-source ids are discrete task labels and should not be interpreted as ordered fidelity values;
- learned cost quality depends on the representativeness of observed cost data.

These boundaries are explicit errors rather than silent approximations.
