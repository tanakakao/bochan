# 03d. Multi-objective Entropy Search (MO-MES / MO-JES)

bochan exposes BoTorch's native Pareto multi-objective entropy-search acquisitions through the high-level API.

- MO-MES: `qLowerBoundMultiObjectiveMaxValueEntropySearch`
- MO-JES: `qLowerBoundMultiObjectiveJointEntropySearch`

These are distinct from applying scalar MES / JES to a multi-output posterior after scalarization. MO-MES and MO-JES retain the Pareto objective space.

## 1. Selection guide

| Problem | Recommended acquisition |
|---|---|
| Learn about a scalar optimum value | MES |
| Learn about a scalar optimum location and value | JES |
| Learn about Pareto-optimal outputs | MO-MES / MESMO |
| Learn about Pareto-optimal input-output pairs | MO-JES |
| Improve future Pareto hypervolume | HVKG |

MO-MES measures information about Pareto-optimal outputs. MO-JES additionally includes the corresponding Pareto-optimal inputs.

## 2. API names

### MO-MES

```python
AcquisitionConfig(name="mo_mes")
AcquisitionConfig(name="mesmo")
```

Equivalent aliases include:

```text
qmo_mes
qmesmo
multi_objective_mes
qmulti_objective_mes
```

### MO-JES

```python
AcquisitionConfig(name="mo_jes")
```

Equivalent aliases include:

```text
qmo_jes
multi_objective_jes
qmulti_objective_jes
```

The canonical BoTorch class names are also registered.

## 3. Automatic Pareto sampling

When auxiliary inputs are omitted, bochan uses BoTorch's public utilities:

```text
model + bounds
    ↓
sample_optimal_points
    ↓
pareto_sets / pareto_fronts
    ↓
compute_sample_box_decomposition
    ↓
hypercell_bounds
```

`sample_optimal_points` draws posterior paths and optimizes each path to approximate Pareto sets and fronts. bochan does not reimplement that algorithm.

Defaults are:

```python
DataContext(
    bounds=bounds,
    extra={
        "mo_entropy_num_pareto_samples": 8,
        "mo_entropy_num_pareto_points": 8,
        "mo_entropy_num_samples": 64,
        "mo_entropy_estimation_type": "LB",
    },
)
```

Supported entropy estimators follow BoTorch: `"0"`, `"LB"`, `"LB2"`, and `"MC"`.

The posterior-path optimizer can be configured with:

```python
DataContext(
    bounds=bounds,
    extra={
        "mo_entropy_optimizer_kwargs": {
            "pop_size": 1024,
            "max_tries": 10,
        },
    },
)
```

Advanced users can pass a `sample_optimal_points`-compatible callable through `mo_entropy_optimizer`.

## 4. MO-MES

MO-MES requires `hypercell_bounds` at construction time. Normally bochan generates them automatically.

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_mes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

An explicit box decomposition is preserved:

```python
AcquisitionConfig(
    name="mo_mes",
    acqf_kwargs={"hypercell_bounds": hypercell_bounds},
)
```

When `hypercell_bounds` is supplied, auxiliary Pareto sampling is skipped and `bounds` are not required for that preprocessing step.

## 5. MO-JES

MO-JES requires:

- `pareto_sets`
- `pareto_fronts`
- `hypercell_bounds`

All three are generated automatically when omitted.

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

You may provide Pareto samples and let bochan compute only the box decomposition:

```python
AcquisitionConfig(
    name="mo_jes",
    acqf_kwargs={
        "pareto_sets": pareto_sets,
        "pareto_fronts": pareto_fronts,
    },
)
```

`pareto_sets` and `pareto_fronts` must be supplied together.

If `hypercell_bounds` is also explicit, it must correspond to the same Pareto samples. Therefore bochan does not allow the ambiguous case of supplying only `hypercell_bounds` while auto-generating unrelated Pareto samples.

## 6. Objective semantics

MO-MES / MO-JES operate directly on the model outputs as Pareto objectives. They do not consume a scalar BoTorch objective.

The high-level path therefore rejects:

- `AcquisitionConfig.objective`
- `objective_config`
- `objective_factory`
- `posterior_transform`
- automatic `MultiObjectiveConfig` scalarization

If multiple outputs should be combined into one utility, use scalar MES / JES instead:

```text
multi-output → scalar utility → MES / JES
```

For native Pareto information acquisition:

```text
multi-output → Pareto objectives → MO-MES / MO-JES
```

## 7. Maximization and minimization

BoTorch's Pareto sampler and box decomposition expose one `maximize` flag for all objectives. bochan defaults to maximization:

```python
DataContext(
    bounds=bounds,
    extra={"mo_entropy_maximize": True},
)
```

Set it to `False` when all objectives are minimized.

```python
extra={"mo_entropy_maximize": False}
```

Mixed directions are not silently transformed by this native path. If some objectives are maximized and others minimized, transform the model outputs first so that all objectives share one direction.

## 8. q > 1

The lower-bound MO-MES / MO-JES acquisitions can evaluate `q > 1`, but the lower bound is not necessarily monotone when batch elements are added. BoTorch's information-theoretic tutorial optimizes batch MO-MES / MO-JES sequentially and greedily.

bochan therefore automatically sets `sequential=True` for `q > 1`.

```python
OptimizeConfig(q=3)
```

is normalized to sequential candidate generation for MO-MES and MO-JES.

## 9. Formal support scope

The automatic Pareto-sampling path is centered on the contract of BoTorch `sample_optimal_points`: compatible Gaussian GP models, continuous bounds, and homogeneous regression objectives.

### Automatic generation

- at least two outputs
- BoTorch-compatible Gaussian GP model
- continuous inputs
- unconstrained Pareto objectives
- one common optimization direction

### Mixed / categorical inputs

BoTorch's default `sample_optimal_points` optimizer searches continuous bounds. bochan therefore does not auto-generate Pareto samples for mixed / categorical spaces.

Advanced users can provide auxiliary values explicitly:

- MO-MES: `hypercell_bounds`
- MO-JES: `pareto_sets`, `pareto_fronts`, `hypercell_bounds`

Candidate optimization itself may still use bochan's normal mixed-space optimizer settings.

## 10. Constraints

BoTorch's box-decomposition utility has lower-level constrained capabilities, but a correct high-level workflow also needs explicit objective/constraint-output separation and compatible Pareto sampling.

This PR does not silently ignore constraints. The automatic MO-MES / MO-JES path rejects configured constraints so constrained entropy search can be added later as a coherent feature.

## 11. Computational cost

MO-MES / MO-JES are generally more expensive than hypervolume-improvement criteria because each BO iteration may require multiple posterior paths, Pareto optimization for each path, and a box decomposition.

They are most attractive when physical experiments are expensive and the information gained about the Pareto frontier is itself valuable.
