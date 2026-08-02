# `bochan` Theory Reference

This directory is a chapter-based reference for the mathematics, statistical
assumptions, and implementation contracts used by `bochan`.

The documentation connects four layers:

1. mathematical problem formulation;
2. probabilistic model and inference;
3. sequential decision criterion;
4. concrete `bochan` / BoTorch implementation.

Each topic has one primary chapter. Other chapters refer to that chapter instead
of repeating the same derivation. Every model or acquisition chapter contains a
theory-to-implementation correspondence section.

---

## Part I. Foundations and sequential decisions

| Ch. | File | Primary responsibility |
|---:|---|---|
| 00 | `00_overview.md` | Architecture, terminology, four mathematical spaces, notation, reading paths, and posterior contracts. |
| 01 | `01_gaussian_process_models.md` | GP priors, kernels, conditioning, marginal likelihood, variational inference, transforms, posterior sampling, and numerical stability. |
| 02 | `02_bayesian_optimization.md` | BO problem, regret, sequential loop, noisy observations, q-batch, pending points, look-ahead, stopping, and evaluation. |
| 03 | `03_acquisition_functions.md` | PI, EI, LogEI, NEI, UCB, Thompson sampling, KG, multi-step look-ahead, objectives, constraints, and acquisition optimization. |
| 04 | `04_active_learning.md` | Entropy, epistemic/aleatoric uncertainty, BALD, margin, probability variance, IPV/NIPV, joint and heteroscedastic Active Learning. |
| 05 | `05_level_set_estimation.md` | Level sets, boundaries, confidence sets, external losses, multi-threshold and multi-output regions, stopping, and evaluation. |
| 06 | `06_classification_and_ordinal_bo.md` | Probability and utility objectives, transformed EI/PI/UCB, discrete-output constraints, calibration, and risk-sensitive class decisions. |
| 07 | `07_multi_objective_and_constraints.md` | Pareto dominance, hypervolume, EHVI/NEHVI, scalarization, chance constraints, feasibility, and repair. |
| 08 | `08_input_perturbation_and_risk.md` | Mean/worst-case robustness, VaR, CVaR, chance constraints, perturbation distributions, reduction order, and `q * n_w`. |
| 09 | `09_shape_conventions.md` | Canonical tensor-axis, posterior, sample, objective, q-batch, DeepGP, ensemble, class, boundary, and perturbation contracts. |

## Part II. Model families and implementation details

| Ch. | File | Primary responsibility |
|---:|---|---|
| 10 | `10_regression_models_and_likelihoods.md` | Gaussian, Beta, Gamma, Poisson, Negative-Binomial, mixed-input, and multi-output regression. |
| 11 | `11_classification_models.md` | Binary and multiclass likelihoods, variational inference, probability marginalization, calibration, posterior contracts, and source mapping. |
| 12 | `12_ordinal_models.md` | Ordered-logit likelihood, cutpoints, identifiability, quadrature, class probabilities, utility summaries, calibration, and posterior contracts. |
| 13 | `13_heteroscedastic_and_robust_models.md` | Known versus learned noise, residual-noise approximation, label-noise interpretations, RRP, outliers, and robust-model source paths. |
| 14 | `14_deep_and_high_dimensional_models.md` | DKL, DeepGP, SAAS, PCA, REMBO, VAE-GP, assumptions, failure modes, and validation. |
| 15 | `15_heterogeneous_multi_output.md` | Independent and shared-latent heterogeneous models, `OutputSpec`, `HybridMultiOutputModel`, and proxy-posterior limitations. |
| 16 | `16_level_set_mathematics_and_implementation.md` | Exact formulas and source mappings for current regression, binary, multiclass, ordinal, multi-output, heteroscedastic, and robust LSE classes. |

---

## Topic ownership

To avoid duplication, the chapters use the following ownership rules:

- GP probability theory and ELBO: Chapter 01.
- BO acquisition mathematics: Chapter 03.
- Active Learning criteria: Chapter 04.
- LSE problem definition and losses: Chapter 05.
- Classification/ordinal decision objectives: Chapter 06.
- Pareto and constraints: Chapter 07.
- Risk measures and perturbation sampling: Chapter 08.
- Tensor axes and reduction order: Chapter 09.
- Model likelihoods and posterior contracts: Chapters 10–15.
- Current LSE class formulas and naming caveats: Chapter 16.

---

## Recommended reading tracks

### Standard Gaussian-process BO

1. Chapters 00 and 01
2. Chapter 10
3. Chapters 02 and 03
4. Chapter 09

### Binary or multiclass work

1. Chapters 00, 01, and 11
2. Chapter 06 for BO, or Chapter 04 for Active Learning
3. Chapters 05 and 16 for LSE
4. Chapter 09

### Ordinal work

1. Chapters 00, 01, and 12
2. Chapter 06 for BO, or Chapter 04 for Active Learning
3. Chapters 05 and 16 for LSE
4. Chapter 09

### Heteroscedastic or robust experiments

1. Chapters 01, 10, and 13
2. Chapter 08 for robust decision criteria
3. Chapter 04 or 16 for sequential learning
4. Chapter 09

### Multi-objective or heterogeneous outputs

1. Chapter 07
2. Chapter 15
3. Chapters 06 and 12 for discrete utility channels
4. Chapter 09

### Deep or high-dimensional models

1. Chapter 01 and the relevant response-model chapter
2. Chapter 14
3. Chapters 02 and 03
4. Chapter 09

---

## Core posterior contracts

Current model families do not share one universal meaning of `posterior()`.

| Model family | Current primary contract |
|---|---|
| Gaussian regression | Gaussian response/latent posterior, optionally including observation noise. |
| Binary classification | `posterior()` returns probability-space posterior; `latent_posterior()` returns latent GP. |
| Multiclass classification | `posterior()` returns class-probability posterior; `latent_posterior()` returns class-wise latent GP. |
| Ordinal | `posterior()` returns scalar latent GP; `class_probs()` returns ordered-class probabilities. |
| Hybrid | `posterior(..., output_mode=...)` returns a mode-specific `HybridPosterior`. |

A reported variance can mean latent epistemic variance, noisy predictive
variance, Bernoulli/categorical observation variance, posterior probability
variance, class-utility variance, auxiliary noise, or proxy variance. The model
chapter identifies the meaning.

A common `[..., q, m]` shape also does not imply cross-output covariance.
Outputs can be independent ModelList channels, correlated multitask channels, or
heterogeneous channels stacked after transformation.

---

## Canonical reduction axes

The following axes may coexist:

```text
posterior samples
model / ensemble batch
BoTorch t-batch
q candidates
input perturbations
outputs or tasks
classes
ordinal boundaries
```

Reduction order is part of the mathematical definition. Chapter 09 is the
canonical interface reference.

---

## Documentation standard

Each chapter should identify:

1. problem and random variables;
2. assumptions and likelihood;
3. derivation or core equations;
4. uncertainty interpretation;
5. non-equivalent alternatives;
6. tensor consequences;
7. implementation classes and source paths;
8. approximations and limitations;
9. validation metrics;
10. references.

When a familiar name such as ICU, heteroscedastic classification, or hybrid
posterior is implemented as a practical proxy, the documentation states the
actual formula and posterior contract rather than relying on the name alone.
