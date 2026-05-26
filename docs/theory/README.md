# Theory documentation

This directory summarizes the theoretical background behind `bochan`.

The goal is not to replace BoTorch or Gaussian process textbooks. The goal is to connect three layers that are easy to separate in practice:

1. the mathematical problem formulation,
2. the BoTorch implementation concepts, and
3. the API design used in this repository.

## Documents

| File | Topic |
|---|---|
| `00_overview.md` | Overall design philosophy and problem taxonomy. |
| `01_gaussian_process_models.md` | Gaussian process models, predictive posteriors, and latent functions. |
| `02_bayesian_optimization.md` | Bayesian optimization loop, exploitation, exploration, and q-batch selection. |
| `03_acquisition_functions.md` | Acquisition function families used for optimization, active learning, and boundary search. |
| `04_active_learning.md` | Difference between Bayesian optimization and active learning. |
| `05_level_set_estimation.md` | Level-set estimation and boundary-oriented acquisition functions. |
| `06_classification_and_ordinal_bo.md` | Binary, multiclass, and ordinal Bayesian optimization. |
| `07_multi_objective_and_constraints.md` | Multi-output, multi-objective, Pareto optimization, and constraints. |
| `08_input_perturbation_and_risk.md` | Input perturbation, robustness, VaR, CVaR, and risk-aware objectives. |
| `09_shape_conventions.md` | Tensor shape conventions shared by models, objectives, and acquisition functions. |

## Reading order

For users who want the minimum theoretical path, read the files in this order:

1. `00_overview.md`
2. `01_gaussian_process_models.md`
3. `02_bayesian_optimization.md`
4. `03_acquisition_functions.md`
5. `09_shape_conventions.md`

For users working on practical extensions, continue with:

1. `04_active_learning.md`
2. `05_level_set_estimation.md`
3. `06_classification_and_ordinal_bo.md`
4. `07_multi_objective_and_constraints.md`
5. `08_input_perturbation_and_risk.md`

## Documentation style

Each theory document follows the same pattern when possible:

1. what problem is being solved,
2. how it is written mathematically,
3. how it maps to BoTorch concepts,
4. how `bochan` extends or unifies the idea,
5. what tensor shapes must be respected.

This structure is intentional. Many errors in practical Bayesian optimization are not conceptual errors but interface errors: latent versus predictive outputs, probability versus utility, scalar versus multi-output objectives, or `q` versus `q * n_w` shapes. These documents make those assumptions explicit.
