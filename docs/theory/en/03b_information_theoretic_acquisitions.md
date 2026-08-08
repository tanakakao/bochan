# 03b. Information-theoretic and look-ahead acquisitions

This chapter describes MES, JES, and HVKG in bochan's high-level API. Unlike EI-style criteria that focus on immediate improvement, these acquisitions value information about the optimum or the future Pareto frontier.

Knowledge Gradient (KG) is also integrated as a look-ahead acquisition. See `03c_knowledge_gradient.md` for its automatic `current_value`, multi-output scalarization, pending-point, and one-shot optimization semantics.

## 1. Max-value Entropy Search (MES)

MES measures information gained about the unknown optimum value `f*`:

```math
\alpha_{MES}(x)=I(y_x; f^*\mid \mathcal D).
```

Use:

```python
AcquisitionConfig(name="mes")
AcquisitionConfig(name="qmes")
```

Both resolve to BoTorch `qMaxValueEntropy`.

When `candidate_set` is omitted, bochan asks BoTorch's registered acquisition input constructor to generate it from `DataContext.bounds`. The size is controlled with:

```python
DataContext(bounds=bounds, extra={"mes_candidate_size": 2000})
```

The default is 1000. For `q > 1`, the high-level API automatically enables sequential optimization, matching qMES batch construction requirements.

A multi-output model must provide an explicit `posterior_transform` so the scalar quantity whose maximum is being learned is unambiguous.

## 2. Joint Entropy Search (JES)

JES measures information about the joint optimum location and value `(x*, f*)`:

```math
\alpha_{JES}(x)=I(y_x;(x^*,f^*)\mid\mathcal D).
```

Use:

```python
AcquisitionConfig(name="jes")
AcquisitionConfig(name="qjes")
```

Both resolve to BoTorch `qJointEntropySearch`.

If `optimal_inputs` and `optimal_outputs` are omitted, bochan uses BoTorch's input constructor to draw posterior optimal samples. The number of optimum samples can be configured with:

```python
DataContext(bounds=bounds, extra={"jes_num_optima": 64})
```

Explicit `optimal_inputs` and `optimal_outputs` must be supplied together. Multi-output JES requires an explicit `posterior_transform`.

## 3. Hypervolume Knowledge Gradient (HVKG)

HVKG is a multi-objective Knowledge Gradient that values how an observation can improve the future Pareto frontier and hypervolume.

```python
AcquisitionConfig(name="hvkg")
AcquisitionConfig(name="qhvkg")
```

Both resolve to BoTorch `qHypervolumeKnowledgeGradient`. HVKG requires at least two objective outputs.

### Reference point precedence

Reference points are resolved in this order:

1. `AcquisitionConfig.acqf_kwargs["ref_point"]`
2. `DataContext.ref_point`
3. an automatically inferred point from observed multi-objective values

An explicit value is never overwritten by a context or inferred value.

### Current value

When `current_value` is omitted, bochan delegates its computation to BoTorch's registered HVKG input constructor. Explicit `current_value` is preserved.

```python
AcquisitionConfig(
    name="hvkg",
    acqf_kwargs={"num_fantasies": 8, "num_pareto": 10},
)
```

HVKG is a one-shot acquisition, so the high-level API keeps joint optimization rather than converting it to sequential optimization.

## 4. Objective semantics

MES and JES do not silently consume bochan MC objectives. If a multi-output posterior needs scalarization, use an explicit `posterior_transform`.

KG uses a scalar terminal objective. Single-output KG can use the posterior mean directly; multi-output KG requires an explicit `objective`, `objective_config`, `objective_factory`, or `posterior_transform`.

HVKG preserves the objective dimension. Generic scalarization through `MultiObjectiveConfig.scalarization_weights` is disabled on the cloned HVKG context so hypervolume is computed in multi-objective space.

## 5. Task routing

The short aliases `kg`, `mes`, `jes`, and `hvkg` are restricted to regression-like posterior semantics. They are not silently redirected to binary, multiclass, or ordinal models because probability and utility spaces require task-specific information criteria.

For classification-oriented information acquisition, use bochan's task-specific BALD and predictive-entropy acquisitions.

## 6. Examples

### KG

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="kg"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

`current_value` is inferred automatically from the bounds. See `03c_knowledge_gradient.md` for details.

### MES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mes"),
    OptimizeConfig(q=3),
    data_context=DataContext(bounds=bounds),
)
```

`q=3` automatically uses sequential optimization.

### JES

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(
        bounds=bounds,
        extra={"jes_num_optima": 64},
    ),
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

## 7. Selection guide

| Goal | Acquisition |
|---|---|
| Improve expected terminal decision value | KG |
| Reduce uncertainty about the optimum value | MES |
| Identify optimum location and value jointly | JES |
| Improve the future multi-objective Pareto frontier | HVKG |
| Optimize immediate scalar improvement | LogEI / LogNEI |
| Optimize immediate hypervolume improvement | LogEHVI / LogNEHVI |

KG, MES, JES, and HVKG are typically more computationally demanding than improvement-based acquisitions, but can be attractive when each physical experiment is expensive and information value matters.
