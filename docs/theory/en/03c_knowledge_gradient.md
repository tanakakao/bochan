# 03c. Knowledge Gradient (KG)

This chapter documents Knowledge Gradient (KG) in bochan's high-level API.
KG was already routed to BoTorch's `qKnowledgeGradient`; this integration adds
permanent high-level defaults and validation at the same level as MES, JES, and
HVKG.

## 1. Role

KG is a look-ahead acquisition that values a candidate by the expected
improvement in the best terminal decision after observing that candidate.

```python
AcquisitionConfig(name="kg")
AcquisitionConfig(name="qkg")
```

resolve to BoTorch `qKnowledgeGradient`.

`qKnowledgeGradient` is a one-shot acquisition: candidate variables and the
fantasy-model terminal optimizers are optimized jointly.

## 2. Automatic current value

`current_value` is the current expected best terminal objective under the
observed data. Without it, BoTorch qKG returns the expected best value after
augmenting the data rather than the actual KG difference.

Therefore bochan computes `current_value` automatically from `DataContext.bounds`
through BoTorch's registered qKG input constructor when it is not supplied.

```python
acq = AcquisitionConfig(name="kg")
context = DataContext(bounds=bounds)
```

The default number of fantasies is 64. It can be changed through the context:

```python
context = DataContext(
    bounds=bounds,
    extra={"kg_num_fantasies": 32},
)
```

or explicitly on the acquisition:

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={"num_fantasies": 32},
)
```

An explicit acquisition value takes precedence.

## 3. Explicit current value

An explicit `current_value` is preserved and is not recomputed.

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={"current_value": current_value},
)
```

In this case bounds are not needed solely for terminal-value computation.

## 4. Pending points

When `X_pending` is present, the qKG `current_value` should be the terminal value
conditioned on those pending evaluations. BoTorch's regular
`construct_inputs_qKG(..., with_current_value=True)` does not compute a
pending-conditioned current value.

bochan therefore does not silently auto-compute `current_value` in that case.
Supply the pending-conditioned value explicitly:

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={"current_value": pending_conditioned_current_value},
)
context = DataContext(X_pending=X_pending)
```

## 5. Single-objective regression

The standard use case is single-output regression.

```python
acq = AcquisitionConfig(name="kg")
opt = OptimizeConfig(q=1)

X_next, value = optimizer.candidate(
    acq,
    opt,
    data_context=DataContext(bounds=bounds),
)
```

Without an explicit objective, qKG uses the posterior mean as the terminal
value function.

## 6. Multi-output regression

For a multi-output model, KG requires an explicit scalar terminal objective.
bochan accepts either:

- `AcquisitionConfig.objective`, `objective_factory`, or `objective_config`, or
- an explicit `posterior_transform` in `acqf_kwargs`.

If neither is supplied, the high-level API raises an error rather than guessing
a scalarization.

This is distinct from Pareto multi-objective optimization. If the outputs must
remain separate objectives, use HVKG instead.

## 7. Optimizer semantics

Because qKG is a one-shot acquisition, bochan normalizes
`OptimizeConfig.sequential=True` back to `False` for KG.

BoTorch `optimize_acqf` automatically selects its specialized one-shot KG
initializer for `qKnowledgeGradient`, so bochan does not duplicate that
initializer.

## 8. Task routing

The short aliases `kg`, `qkg`, `knowledgegradient`, and `qknowledgegradient`
are restricted to regression-like posterior semantics.

Binary, multiclass, and ordinal tasks are not silently routed to standard qKG;
use the existing task-specific BALD, predictive entropy, margin uncertainty, and
related acquisitions instead.

## 9. KG / MES / JES / HVKG

| Method | Main scope | Value of information | Optimization |
|---|---|---|---|
| KG | single objective | terminal decision value | one-shot |
| MES | single objective | information about optimum value | sequential for q > 1 |
| JES | single objective | information about optimum input and value | regular joint/q semantics |
| HVKG | multi-objective | future Pareto / hypervolume value | one-shot |
