# acquisition package

This package contains unified acquisition functions and objective classes under
`bochan.acquisition`.

The current design separates acquisition functions by model family and task:

- Bayesian optimization
- active learning
- level-set estimation
- objective / score transformation

Legacy class names and compatibility aliases are intentionally not treated as
public API. Public classes should use the unified naming scheme described below.

---

## Current layout

```text
acquisition/
├── objective/
│   ├── regression.py
│   ├── binary.py
│   ├── ordinal.py
│   └── hybrid.py
│
├── feasible/
│   ├── constraints.py
│   └── wrapper.py
│
├── regression/
│   ├── bayesian_optimization/
│   ├── active_learning/
│   └── levelset_estimation/
│
├── binary/
│   ├── bayesian_optimization/
│   ├── active_learning/
│   └── levelset_estimation/
│
├── ordinal/
│   ├── bayesian_optimization/
│   ├── active_learning/
│   └── levelset_estimation/
│
└── non_gaussian/
    ├── _stats.py
    ├── active_learning/
    │   ├── single_output.py
    │   ├── multi_output.py
    │   ├── hetero_single_output.py
    │   └── hetero_multi_output.py
    ├── levelset_estimation/
    │   ├── single_output.py
    │   ├── multi_output.py
    │   ├── hetero_single_output.py
    │   └── hetero_multi_output.py
    └── bayesian_optimization/
        ├── single_output.py
        ├── multi_output.py
        ├── hetero_single_output.py
        └── hetero_multi_output.py
```

Notes:

- `binary/` is the binary-classification acquisition family.
- `non_gaussian/` is a top-level acquisition family, not a subpackage of
  `regression/`. It is intended for Poisson, Beta, Gamma, and Negative Binomial
  GP-style wrappers.
- Some `non_gaussian` files are intentionally placeholders to keep the directory
  structure aligned with the other acquisition families.

---

## Naming policy

Only unified names are public. Legacy class names and compatibility aliases are
intentionally removed unless explicitly documented.

Public acquisition names follow these patterns:

| Family | Naming pattern |
|---|---|
| Gaussian / standard regression | `qRegression...` |
| Binary classification | `qBinary...` |
| Ordinal regression | `qOrdinal...` |
| Non-Gaussian regression | `qNonGaussian...` |
| Multi-output regression | `qMultiOutputRegression...` |
| Multi-output binary classification | `qMultiOutputBinary...` |
| Multi-output ordinal regression | `qMultiOutputOrdinal...` |
| Heteroscedastic regression | `qHeteroRegression...` |
| Heteroscedastic binary classification | `qHeteroBinary...` |
| Heteroscedastic ordinal regression | `qHeteroOrdinal...` |
| Heteroscedastic multi-output regression | `qHeteroMultiOutputRegression...` |
| Heteroscedastic multi-output binary classification | `qHeteroMultiOutputBinary...` |
| Heteroscedastic multi-output ordinal regression | `qHeteroMultiOutputOrdinal...` |

Objective classes are placed in `acquisition/objective/`.

---

## Objective package

Objective classes convert posterior samples or acquisition scores into the scale
expected by BoTorch acquisition functions.

```python
from bochan.acquisition.objective import (
    RegressionScalarObjective,
    RegressionLinearMCObjective,
    MultiOutputRegressionInputPerturbationObjective,
    BinaryClassificationScoreObjective,
    MultiOutputBinaryClassificationScoreObjective,
    MultiOutputBinaryClassificationInputPerturbationObjective,
    OrdinalInputPerturbationExpectedUtilityObjective,
    MultiOutputOrdinalInputPerturbationObjective,
    OrdinalScoreObjective,
    MultiOutputOrdinalScoreObjective,
    HybridObjectiveSpec,
    make_hybrid_scalar_objective,
    make_hybrid_multi_output_objective,
)
```

Typical responsibilities:

- scalarization of regression outputs
- probability / utility conversion for binary and ordinal models
- input-perturbation aggregation from `q * n_w` back to `q`
- risk aggregation such as mean / VaR / CVaR
- qEHVI / qNEHVI / qNParEGO compatible multi-output transformation

### Hybrid objective helpers

`objective/hybrid.py` is a thin adapter around the regression objective classes.
It does not reimplement binary or ordinal utility conversion. The intended
responsibility split is:

- `OutputSpec`: output meaning and task-specific conversion metadata
  (`task_type`, `positive_class`, `utility_values`, etc.).
- `HybridObjectiveSpec`: optimization setting for each objective output
  (`direction`, `weight`, `eq_target`).

For hybrid models, `HybridMultiOutputModel.posterior(..., output_mode="objective")`
already converts regression / binary / ordinal / multiclass outputs into an
objective-space `[..., q, m]` tensor. Therefore the hybrid objective helpers reuse:

- `RegressionScalarObjective`
- `RegressionLinearMCObjective`
- `MultiOutputRegressionInputPerturbationObjective`

Example:

```python
from bochan.acquisition.objective import make_hybrid_scalar_objective

objective = make_hybrid_scalar_objective(
    model=hybrid_model,
    output="strength",
    direction="maximize",
)
```

Multi-output / perturbation example:

```python
from bochan.acquisition.objective import HybridObjectiveSpec, make_hybrid_multi_output_objective

objective = make_hybrid_multi_output_objective(
    model=hybrid_model,
    specs=[
        HybridObjectiveSpec("strength", direction="maximize", weight=1.0),
        HybridObjectiveSpec("cost", direction="minimize", weight=0.5),
        HybridObjectiveSpec("quality_rank", direction="maximize", weight=2.0),
    ],
    n_w=8,
    risk_type="cvar",
    alpha=0.8,
)
```

The same helpers can also be used with non-hybrid multi-output models by using
integer output indices instead of output names.

---

## Bayesian optimization

### Standard regression

For standard regression, prefer BoTorch's existing acquisition functions when
possible, for example:

- `qExpectedImprovement`
- `qNoisyExpectedImprovement`
- `qUpperConfidenceBound`
- `qProbabilityOfImprovement`
- `qLogExpectedImprovement`
- `qLogNoisyExpectedImprovement`
- `qExpectedHypervolumeImprovement`
- `qNoisyExpectedHypervolumeImprovement`
- `qNParEGO`

The local regression BO package is used only where custom behavior is needed,
such as heteroscedastic or wrapper-specific behavior.

### Binary / ordinal Bayesian optimization

Binary and ordinal models often require probability / utility conversion before
standard BoTorch BO logic can be applied. These families therefore provide
custom acquisition classes and objectives.

| Family | Binary | Ordinal |
|---|---|---|
| Probability / feasibility | `qBinaryProbabilityOfFeasibility` | utility / feasibility objectives |
| Expected improvement style | binary BO wrappers | ordinal utility wrappers |
| qEHVI | `qMultiOutputBinaryExpectedHypervolumeImprovement` | `qMultiOutputOrdinalExpectedHypervolumeImprovement` |
| qNEHVI | `qMultiOutputBinaryNoisyExpectedHypervolumeImprovement` | `qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement` |
| qNParEGO | `qMultiOutputBinaryNParEGO` | `qMultiOutputOrdinalNParEGO` |

### Non-Gaussian Bayesian optimization

`acquisition/non_gaussian/bayesian_optimization/` currently contains placeholder
modules only.

This is intentional. Standard BoTorch BO acquisitions can generally be used with
Poisson / Beta / Gamma / Negative Binomial wrappers if the model exposes a
response-scale `posterior` and an appropriate objective is supplied where needed.
Therefore, qEI / qNEI / qUCB / qPI / qEHVI / qNEHVI / qNParEGO are not
reimplemented under `non_gaussian` at this stage.

---

## Active learning families

| Family | Meaning |
|---|---|
| Predictive Entropy | Select points with ambiguous predictions. |
| BALD / MI | Select points with high model-information gain. |
| Variance | Select points with high posterior / probability / utility variance. |
| Margin Uncertainty | Select points near a decision or ordinal boundary. |
| Integrated Posterior Variance | Select points expected to reduce global uncertainty. |

### Regression active learning

Main single-output classes include:

- `qRegressionPredictiveEntropy`
- `qRegressionBALD`
- `qRegressionPosteriorVariance`
- `qRegressionNegIntegratedPosteriorVariance`
- `qRegressionIntegratedPosteriorVarianceProxy`

The true `qRegressionNegIntegratedPosteriorVariance` delegates to BoTorch's
`qNegIntegratedPosteriorVariance` when available. The proxy variant is intended
for custom models that do not support `fantasize()`.

### Binary active learning

Main single-output classes include:

- `qBinaryPredictiveEntropy`
- `qBinaryBALD`
- `qBinaryJointBALD`
- `qBinaryGreedyJointBALD`
- `qBinaryProbabilityVariance`
- `qBinaryMarginUncertainty`

### Ordinal active learning

Ordinal active learning follows the same design idea as binary classification,
but scores are computed from class probabilities, utilities, or ordinal boundary
uncertainty.

### Non-Gaussian active learning

Implemented under:
