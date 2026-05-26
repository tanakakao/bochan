# 07. Multi-objective Optimization and Constraints

Practical Bayesian optimization often involves several objectives and several constraints. This document summarizes the theoretical assumptions used by `bochan` for multi-output, multi-objective, and constrained problems.

---

## 1. Multi-output versus multi-objective

A model is multi-output when it predicts several response channels:

```text
f(x) = (f_1(x), f_2(x), ..., f_m(x)).
```

A problem is multi-objective when more than one of those outputs is optimized as an objective.

These are related but not identical:

- A multi-output model may include objectives, constraints, and auxiliary predictions.
- A multi-objective acquisition function needs objective dimensions that define a Pareto problem.
- Some outputs may need to be scalarized or transformed before optimization.

This is why `bochan` separates model outputs from objective values.

---

## 2. Pareto dominance

For maximization, a point `a` dominates a point `b` if:

```text
f_j(a) >= f_j(b) for all j
```

and

```text
f_j(a) > f_j(b) for at least one j.
```

The Pareto frontier is the set of non-dominated trade-off points.

In multi-objective BO, the goal is usually not a single optimum but a useful approximation of the Pareto frontier.

---

## 3. Hypervolume

Hypervolume measures the volume dominated by the Pareto set relative to a reference point.

For maximization, the reference point should be worse than the relevant objective values:

```text
ref_point = [r_1, ..., r_m].
```

Expected Hypervolume Improvement evaluates the expected increase in hypervolume after observing a candidate.

Common acquisition functions:

- EHVI for noiseless multi-objective BO,
- NEHVI for noisy multi-objective BO,
- qEHVI and qNEHVI for q-batch candidate selection.

---

## 4. Reference point selection

The reference point strongly affects hypervolume-based optimization.

A practical reference point should be:

1. worse than observed or expected objective values,
2. expressed in the same transformed objective space as the acquisition function,
3. updated carefully when objectives are standardized or utility-transformed,
4. documented for each example.

If outputs include probabilities or ordinal utilities, the reference point should be chosen after the objective conversion.

---

## 5. Scalarization

Scalarization converts multiple objectives into one scalar objective:

```text
s(x) = w^T f(x).
```

NParEGO uses randomized scalarizations to explore different parts of the Pareto frontier.

Scalarization is often simpler than hypervolume methods but depends strongly on the scale and weights of each objective.

For hybrid outputs, scalarization should operate on objective values, not raw model outputs.

---

## 6. Constraints

A constrained problem can be written as:

```text
maximize f(x)
subject to g_j(x) <= 0,  j = 1, ..., J.
```

There are several distinct constraint types:

### Design-space constraints

These are constraints directly on `x`, such as:

```text
x_1 + x_2 <= 1
```

or k-sparse constraints. They can often be handled inside the optimizer, through repair functions, or by candidate filtering.

### Outcome constraints

These are constraints on unknown outputs, such as:

```text
P(y_feasible = 1 | x) >= 0.9
```

or

```text
g(x) <= 0.
```

They usually require a model and a feasibility probability.

### Post-processing constraints

These are practical constraints imposed after optimization, such as rounding to valid grid values or repairing a candidate to satisfy linear constraints.

---

## 7. Classification and ordinal constraints

Classification and ordinal models are often useful as constraint models.

Examples:

```text
binary classifier: feasible / infeasible
ordinal model: quality grade must be at least class 2
multiclass model: class must not be failure mode A
```

These constraints should be expressed through objective or feasibility functions:

```text
P(feasible | x)
P(grade >= k | x)
E[utility | x]
```

rather than by treating class labels as continuous regression values.

---

## 8. Multi-output reduction

When a model produces multiple outputs, an acquisition function must know whether to:

- preserve all outputs for multi-objective optimization,
- reduce outputs by a weighted mean,
- select one output as the objective,
- convert outputs to constraints,
- compute a custom utility.

This should be handled by an objective layer whenever possible.

A useful implementation rule is:

```text
model output dimension != objective dimension necessarily.
```

For example, an ordinal model may output `K` class probabilities, but the objective may be a single expected utility.

---

## 9. Pending points and feasibility

In constrained q-batch optimization, pending points matter in two ways:

1. they may duplicate planned evaluations,
2. their unknown outcomes may affect the future feasible Pareto frontier.

Acquisition functions with explicit `X_pending` support should use it. If not available, a distance-based pending penalty can be a practical approximation.

The distance should be computed in a consistent space, especially when input transforms or dimensionality reduction are active.

---

## 10. Design guidance

A multi-objective or constrained component should document:

1. which outputs are objectives,
2. which outputs are constraints,
3. whether objectives are maximized or minimized,
4. the transformed objective scale,
5. the reference point if hypervolume is used,
6. the scalarization rule if scalarization is used,
7. the feasibility rule for constraints,
8. how classification or ordinal outputs are converted,
9. whether q-batch and `X_pending` are supported,
10. how candidate repair or rounding interacts with constraints.

These details determine whether a mathematically correct acquisition function is actually optimizing the intended practical problem.
