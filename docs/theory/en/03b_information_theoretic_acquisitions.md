# 03b. Information-theoretic and look-ahead acquisitions

This chapter summarizes KG, MES, JES, MO-MES, MO-JES, and HVKG in bochan's high-level API. Unlike EI-style criteria that focus on immediate improvement, these acquisitions value information about optimum values, optimum locations, Pareto fronts, or future terminal decisions.

See `03c_knowledge_gradient.md` for KG `current_value`, multi-output scalarization, pending-point, and one-shot semantics. See `03d_multiobjective_entropy_search.md` for native Pareto multi-objective entropy search.

## 1. Max-value Entropy Search (MES)

MES measures information gained about an unknown scalar optimum value `f*`:

```math
\alpha_{MES}(x)=I(y_x; f^*\mid \mathcal D).
```

```python
AcquisitionConfig(name="mes")
AcquisitionConfig(name="qmes")
```

Both resolve to BoTorch `qMaxValueEntropy`.

When `candidate_set` is omitted, bochan asks BoTorch's registered input constructor to generate it from `DataContext.bounds`.

```python
DataContext(bounds=bounds, extra={"mes_candidate_size": 2000})
```

The default is 1000. For `q > 1`, the high-level API automatically enables sequential optimization.

A multi-output model must provide an explicit `posterior_transform`. This is scalar MES applied after output scalarization; it is not Pareto multi-objective MES.

## 2. Joint Entropy Search (JES)

JES measures information about a scalar optimum location-value pair `(x*, f*)`:

```math
\alpha_{JES}(x)=I(y_x;(x^*,f^*)\mid\mathcal D).
```

```python
AcquisitionConfig(name="jes")
AcquisitionConfig(name="qjes")
```

Both resolve to BoTorch `qJointEntropySearch`.

If `optimal_inputs` and `optimal_outputs` are omitted, bochan uses BoTorch's input constructor to draw posterior optimal samples.

```python
DataContext(bounds=bounds, extra={"jes_num_optima": 64})
```

Explicit `optimal_inputs` and `optimal_outputs` must be supplied together. Multi-output JES requires an explicit `posterior_transform`.

## 3. Native Pareto multi-objective Entropy Search

### MO-MES / MESMO

```python
AcquisitionConfig(name="mo_mes")
AcquisitionConfig(name="mesmo")
```

These resolve to BoTorch `qLowerBoundMultiObjectiveMaxValueEntropySearch`, which values information about Pareto-optimal outputs.

### MO-JES

```python
AcquisitionConfig(name="mo_jes")
```

This resolves to BoTorch `qLowerBoundMultiObjectiveJointEntropySearch`, which values information about Pareto-optimal input-output pairs.

MO-MES / MO-JES retain the multi-output Pareto space rather than scalarizing it. When auxiliary quantities are omitted, bochan uses BoTorch `sample_optimal_points` and `compute_sample_box_decomposition` to generate posterior Pareto samples and hypercell bounds.

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

For `q > 1`, bochan uses sequential greedy candidate generation, following BoTorch's information-theoretic tutorial. The formal automatic-generation scope is continuous-input, Gaussian-GP-compatible, unconstrained regression with at least two objectives. Mixed / categorical spaces require explicit auxiliary values.

See `03d_multiobjective_entropy_search.md` for the full contract.

## 4. Hypervolume Knowledge Gradient (HVKG)

HVKG is a multi-objective Knowledge Gradient that values how an observation can improve the future Pareto frontier and hypervolume.

```python
AcquisitionConfig(name="hvkg")
AcquisitionConfig(name="qhvkg")
```

Both resolve to BoTorch `qHypervolumeKnowledgeGradient` and require at least two objective outputs.

Reference points are resolved in this order:

1. `AcquisitionConfig.acqf_kwargs["ref_point"]`
2. `DataContext.ref_point`
3. an automatically inferred point from observed multi-objective values

When `current_value` is omitted, bochan delegates its computation to BoTorch's registered HVKG input constructor. Explicit values are preserved.

HVKG is a one-shot acquisition, so the high-level API keeps joint optimization rather than converting it to sequential optimization.

## 5. Objective semantics

- **MES / JES** operate on a scalar posterior. Multi-output use requires an explicit `posterior_transform`.
- **KG** uses a scalar terminal objective. Multi-output KG requires an explicit `objective`, `objective_config`, `objective_factory`, or `posterior_transform`.
- **MO-MES / MO-JES** use the model outputs directly as Pareto objectives and do not consume a scalar `objective` or `posterior_transform`.
- **HVKG** preserves the multi-objective dimension and values future hypervolume.

Multi-output therefore does not automatically imply native multi-objective entropy search:

```text
multi-output → scalar utility → MES / JES / KG
multi-output → Pareto objectives → MO-MES / MO-JES / HVKG
```

## 6. Task routing

Short aliases `kg`, `mes`, `jes`, `mo_mes`, `mo_jes`, and `hvkg` are centered on Gaussian regression posterior semantics. They are not silently redirected to binary, multiclass, or ordinal models.

For classification-oriented information acquisition, use bochan's task-specific BALD and predictive-entropy acquisitions.

Automatic MO-MES / MO-JES Pareto sampling requires homogeneous regression objectives. Hybrid and mixed-categorical automatic Pareto sampling are outside the formal support scope.

## 7. Examples

### KG

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="kg"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mes"),
    OptimizeConfig(q=3),
    data_context=DataContext(bounds=bounds),
)
```

### JES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MO-MES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_mes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### MO-JES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

### HVKG

```python
context = DataContext(
    bounds=bounds,
    multi_objective=MultiObjectiveConfig(ref_point=ref_point),
)

X_next, value = optimizer.candidate(
    AcquisitionConfig(name="hvkg"),
    OptimizeConfig(q=1),
    data_context=context,
)
```

## 8. Selection guide

| Problem | Information / look-ahead | Improvement-based |
|---|---|---|
| Scalar objective / utility | KG / MES / JES | LogEI / LogNEI |
| Native Pareto multi-objective | MO-MES / MO-JES / HVKG | LogEHVI / LogNEHVI |
| Scalarized multi-objective | KG / MES / JES | LogNParEGO |

More specifically:

| Goal | Acquisition |
|---|---|
| Improve expected terminal decision value | KG |
| Reduce uncertainty about a scalar optimum value | MES |
| Identify a scalar optimum location and value jointly | JES |
| Learn about Pareto-optimal outputs | MO-MES / MESMO |
| Learn about Pareto-optimal inputs and outputs | MO-JES |
| Improve future Pareto hypervolume | HVKG |

These information-theoretic and look-ahead acquisitions are generally more computationally demanding than immediate-improvement criteria, but are attractive when physical experiments are expensive and the information gained from each experiment is valuable.
