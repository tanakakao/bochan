# Theory documentation

This directory summarizes the theoretical background behind `bochan`.

The goal is not to replace BoTorch or Gaussian process textbooks. The goal is to connect three layers that are easy to separate in practice:

1. the mathematical problem formulation,
2. the BoTorch implementation concepts, and
3. the API design used in this repository.

The documents are divided into introductory chapters and detailed model / implementation chapters. The introductory chapters explain the common concepts. The detailed chapters define the equations, posterior spaces, approximations, source-code mappings, and limitations of the current implementation.

## Documents

### Introductory and cross-cutting chapters

| File | Topic |
|---|---|
| `00_overview.md` | Overall design philosophy and problem taxonomy. |
| `01_gaussian_process_models.md` | Gaussian process models, predictive posteriors, and latent functions. |
| `02_bayesian_optimization.md` | Bayesian optimization loop, exploitation, exploration, and q-batch selection. |
| `03_acquisition_functions.md` | Acquisition function families used for optimization, active learning, and boundary search. |
| `04_active_learning.md` | Difference between Bayesian optimization and active learning. |
| `05_level_set_estimation.md` | Introductory Level-set Estimation and boundary-oriented acquisition functions. |
| `06_classification_and_ordinal_bo.md` | Introductory binary, multiclass, and ordinal Bayesian optimization. |
| `07_multi_objective_and_constraints.md` | Multi-output, multi-objective, Pareto optimization, and constraints. |
| `08_input_perturbation_and_risk.md` | Input perturbation, robustness, VaR, CVaR, and risk-aware objectives. |
| `09_shape_conventions.md` | Tensor shape conventions shared by models, objectives, and acquisition functions. |

### Detailed model and implementation chapters

| File | Topic |
|---|---|
| `10_regression_models_and_likelihoods.md` | Exact and variational GP regression, Gaussian and non-Gaussian likelihoods, kernels, ARD, and multi-output regression. |
| `11_classification_models.md` | Binary and multiclass GP classification, probability versus latent posteriors, entropy, BALD, and implementation contracts. |
| `12_ordinal_models.md` | Ordered-logit GP models, cutpoints, class probabilities, expected utility, ordinal boundaries, and ordinal BO / AL. |
| `13_heteroscedastic_and_robust_models.md` | Known and learned noise, heteroscedastic regression / classification / ordinal models, RRP, outliers, and input uncertainty. |
| `14_deep_and_high_dimensional_models.md` | Deep Kernel Learning, DeepGP, SAAS, PCA, REMBO, VAE-GP, assumptions, and validation. |
| `15_heterogeneous_multi_output.md` | Independent and correlated heterogeneous outputs, HybridMultiOutputModel, objective-space conversion, and proxy posterior limitations. |
| `16_level_set_mathematics_and_implementation.md` | Detailed LSE losses and formulas for regression, binary, multiclass, ordinal, multi-output, heteroscedastic, robust, and q-batch methods. |

## Reading order

For users who want the minimum theoretical path, read:

1. `00_overview.md`
2. `01_gaussian_process_models.md`
3. `02_bayesian_optimization.md`
4. `03_acquisition_functions.md`
5. `09_shape_conventions.md`

For users implementing or reviewing models, continue with:

1. `10_regression_models_and_likelihoods.md`
2. `11_classification_models.md`
3. `12_ordinal_models.md`
4. `13_heteroscedastic_and_robust_models.md`
5. `14_deep_and_high_dimensional_models.md`
6. `15_heterogeneous_multi_output.md`

For users implementing Active Learning or Level-set Estimation, read:

1. `04_active_learning.md`
2. `05_level_set_estimation.md`
3. `11_classification_models.md`
4. `12_ordinal_models.md`
5. `16_level_set_mathematics_and_implementation.md`
6. `08_input_perturbation_and_risk.md`

For multi-objective or heterogeneous experiments, read:

1. `07_multi_objective_and_constraints.md`
2. `12_ordinal_models.md`
3. `15_heterogeneous_multi_output.md`
4. `16_level_set_mathematics_and_implementation.md`

## Posterior-space contracts

The detailed chapters treat the following as part of the public mathematical contract:

- whether `posterior()` returns a latent function, response distribution, class probability, or objective-space proxy;
- whether observation noise is included;
- whether inference is exact marginal likelihood or a variational ELBO;
- whether multiple outputs are independent, correlated, or only stacked after an objective transformation;
- whether a reported variance is epistemic, observation, class-utility, or a proxy variance;
- how `q`, output, class, model-batch, Monte Carlo, and `q * n_w` dimensions are reduced.

Current model families do not all use the same default posterior space. In particular, binary and multiclass wrappers return probability-space posteriors from `posterior()`, while the base ordinal wrapper returns a latent scalar posterior and exposes probabilities through `class_probs()`. Generic acquisition code must account for this difference.

## Documentation style

Each theory document follows the same pattern when possible:

1. what problem is being solved,
2. how it is written mathematically,
3. which statistical assumptions and approximations are used,
4. how it maps to BoTorch / GPyTorch concepts,
5. how it maps to concrete `bochan` classes and files,
6. what posterior and tensor-shape contracts must be respected,
7. where the current implementation is an approximation rather than an exact generative model.

This structure is intentional. Many errors in practical Bayesian optimization are not conceptual errors but interface errors: latent versus predictive outputs, probability versus utility, epistemic versus observation variance, scalar versus heterogeneous multi-output objectives, or `q` versus `q * n_w` shapes. These documents make those assumptions explicit.