# 02. Bayesian Optimization

Bayesian optimization is a sequential decision-making framework for optimizing expensive black-box functions. This document defines the basic loop and the assumptions used by `bochan`.

---

## 1. Basic formulation

The standard problem is

```text
maximize f(x)
subject to x in X.
```

The function `f` is expensive, noisy, nonconvex, or unavailable in closed form. Instead of optimizing `f` directly, Bayesian optimization builds a probabilistic surrogate model from observed data and optimizes an acquisition function.

At iteration `t`:

```text
D_t = {(x_i, y_i)}_{i=1}^t
```

A surrogate model defines a posterior distribution over `f`. The next candidate is selected by

```text
x_{t+1} = argmax_x alpha_t(x),
```

where `alpha_t` is an acquisition function.

---

## 2. Surrogate model

The surrogate model should provide uncertainty, not only a point prediction.

For a Gaussian process model, the posterior at a candidate point is summarized by:

```text
mu_t(x) = E[f(x) | D_t]
sigma_t^2(x) = Var[f(x) | D_t].
```

A high mean suggests exploitation. A high variance suggests exploration.

The acquisition function converts the predictive distribution into a decision score.

---

## 3. Exploitation and exploration

Bayesian optimization works because it does not greedily choose only the current predicted best point.

- Exploitation: choose points likely to have high objective values.
- Exploration: choose points whose observations may reduce important uncertainty.

Different acquisition functions encode this balance differently:

| Acquisition | Main behavior |
|---|---|
| EI / LogEI | favors expected improvement over current best |
| NEI / LogNEI | improvement under noisy observations |
| UCB | explicit mean-plus-uncertainty trade-off |
| KG | value of information for future decisions |
| MES / PES / JES | information gain about optimum or optimal value |
| EHVI / NEHVI | expected hypervolume improvement in multi-objective BO |
| NParEGO | randomized scalarization for multi-objective BO |

---

## 4. q-batch Bayesian optimization

In many applications, several candidates are evaluated in parallel. q-batch Bayesian optimization chooses `q` points jointly:

```text
X = [x_1, ..., x_q].
```

The acquisition function evaluates the value of a batch, not only independent single points:

```text
alpha(X),  X shape = batch_shape x q x d.
```

Joint q-batch acquisition functions are important because the value of a batch is usually not the sum of independent point values. Two identical candidates are not twice as useful. This is why `X_pending`, duplicate penalties, and joint posterior samples often matter.

---

## 5. Noisy observations

In real experiments, the observed response may be noisy:

```text
y = f(x) + eps.
```

If noise is significant, the current best observed value may not be a reliable estimate of the best latent function value. Noise-aware acquisition functions such as NEI are often preferred.

A practical distinction:

- EI assumes the best known value is meaningful.
- NEI integrates over uncertainty in the latent true values at baseline points.

This is why experiment data often benefits from NEI or related noisy criteria.

---

## 6. Constraints

Many practical BO problems have constraints:

```text
maximize f(x)
subject to g_j(x) <= 0, j = 1, ..., J.
```

Constraints can be handled in several ways:

1. hard constraints in the optimizer or candidate generator,
2. feasibility-weighted acquisition values,
3. model-based constraint probabilities,
4. repair functions or post-processing functions,
5. rejection or filtering after candidate generation.

`bochan` should keep these roles separate. A linear design-space constraint is not the same as a probabilistic output constraint.

---

## 7. Multi-objective Bayesian optimization

For multiple objectives,

```text
f(x) = (f_1(x), ..., f_m(x)),
```

there is usually no single best point. Instead, the goal is to approximate the Pareto frontier.

Common approaches include:

- hypervolume improvement, such as EHVI or NEHVI,
- scalarization, such as NParEGO,
- preference-based utility functions,
- constrained multi-objective acquisition functions.

In `bochan`, multi-objective optimization is especially important because outputs may come from regression, classification, ordinal models, or hybrid model lists. Objectives must make those outputs comparable.

---

## 8. Relationship to active learning and level-set estimation

Bayesian optimization is not the only sequential design goal.

Active learning asks:

```text
Where should we observe to improve the model?
```

Level-set estimation asks:

```text
Where is the boundary f(x) = tau or region f(x) >= tau?
```

These goals can share the same surrogate model infrastructure but require different acquisition functions. `bochan` intentionally keeps these categories close because implementation details such as q-batch shapes, pending points, objectives, and posterior handling are shared.

---

## 9. Practical implementation loop

A minimal BO loop in this repository typically has the following structure:

```text
1. collect train_X, train_Y
2. construct model
3. construct MLL and fit model
4. construct objective if needed
5. construct acquisition function
6. optimize acquisition function under bounds and constraints
7. evaluate target function at candidates
8. append new data and repeat
```

The theoretical object is simple, but the implementation becomes difficult when we add:

- q-batch shapes,
- multi-output posteriors,
- classification or ordinal likelihoods,
- input transforms,
- categorical dimensions,
- noisy baselines,
- constraints,
- input perturbation,
- risk aggregation.

The rest of the theory documents explain these components separately.
