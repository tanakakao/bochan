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
│   └── ordinal.py
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
    ClassificationScoreObjective,
    MultiOutputClassificationScoreObjective,
    MultiOutputClassificationInputPerturbationObjective,
    OrdinalInputPerturbationExpectedUtilityObjective,
    MultiOutputOrdinalInputPerturbationObjective,
    OrdinalScoreObjective,
    MultiOutputOrdinalScoreObjective,
)
```

Typical responsibilities:

- scalarization of regression outputs
- probability / utility conversion for binary and ordinal models
- input-perturbation aggregation from `q * n_w` back to `q`
- risk aggregation such as mean / VaR / CVaR
- qEHVI / qNEHVI / qNParEGO compatible multi-output transformation

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

```text
acquisition/non_gaussian/active_learning/single_output.py
```

Current public classes:

- `qNonGaussianResponseMeanVariance`
- `qNonGaussianPosteriorVariance`
- `qNonGaussianExpectedObservationVariance`
- `qNonGaussianTotalObservationVariance`
- `qNonGaussianExpectedObservationEntropy`
- `qNonGaussianPredictiveEntropyProxy`
- `qNonGaussianBALDProxy`

These classes are shared across Poisson, Beta, Gamma, and Negative Binomial
wrappers. They use `latent_posterior(X)` and `model.likelihood(function_samples)`
to estimate response-scale statistics.

Important interpretation:

- `qNonGaussianResponseMeanVariance` measures reducible uncertainty in the
  response mean, i.e. uncertainty induced by the latent GP.
- `qNonGaussianExpectedObservationVariance` measures conditional observation
  noise and may over-select intrinsically noisy regions.
- `qNonGaussianTotalObservationVariance` combines reducible latent uncertainty
  and irreducible observation noise.
- `qNonGaussianPredictiveEntropyProxy` and `qNonGaussianBALDProxy` are
  moment-matched proxies, not exact mixture-entropy implementations.

The following files currently exist as placeholders to keep the package layout
consistent:

- `active_learning/multi_output.py`
- `active_learning/hetero_single_output.py`
- `active_learning/hetero_multi_output.py`

---

## Level-set estimation families

| Family | Meaning |
|---|---|
| Straddle | Select points near a target boundary and with high uncertainty. |
| Joint Straddle | Batch-aware straddle using joint uncertainty. |
| ICU | Integrated contour uncertainty around a boundary. |
| Boundary Variance | Variance weighted by boundary proximity. |
| Probability of Exceedance | Boundary / threshold probability score. |
| Class Entropy | Entropy-based boundary exploration for classification / ordinal models. |

### Regression level-set estimation

Main single-output classes include:

- `RegressionLevelSetScoreObjective`
- `qRegressionStraddle`
- `qRegressionJointStraddle`
- `qRegressionICU`
- `qRegressionBoundaryVariance`
- `qRegressionProbabilityOfExceedance`

### Binary / ordinal level-set estimation

Binary and ordinal level-set acquisitions use class probabilities, class entropy,
margin uncertainty, and boundary-aware scores. Ordinal variants additionally
support ordinal boundary selection and utility-based scoring.

### Non-Gaussian level-set estimation

Implemented under:

```text
acquisition/non_gaussian/levelset_estimation/single_output.py
```

Current public classes:

- `qNonGaussianStraddle`
- `qNonGaussianBoundaryVariance`
- `qNonGaussianICU`
- `qNonGaussianProbabilityOfExceedance`

These classes operate on the response mean scale. For example,
`qNonGaussianStraddle` uses:

```text
score(x) = beta * std_response_mean(x) - |E[y | x] - threshold|
```

The following files currently exist as placeholders to keep the package layout
consistent:

- `levelset_estimation/multi_output.py`
- `levelset_estimation/hetero_single_output.py`
- `levelset_estimation/hetero_multi_output.py`

---

## Heteroscedastic variants

Heteroscedastic acquisition functions apply noise-aware or robust scoring such as:

```text
adjusted_sample = mean + beta * (sample - mean) - noise_penalty * sigma_noise
```

Use heteroscedastic variants when the observation noise depends on the input or
when noisy regions should be avoided.

Current heteroscedastic implementations exist mainly for regression, binary, and
ordinal families. Non-Gaussian heteroscedastic files are present as placeholders
until the non-Gaussian model API exposes a stable separation between latent
uncertainty and input-dependent observation noise.

---

## Input perturbation and risk objectives

Many acquisition classes accept an `objective` argument. For input perturbation,
objectives are often used to aggregate expanded scores:

```text
q * n_w -> q
```

Common settings include:

- `risk_type=None`: mean aggregation
- `risk_type="var"`: VaR-style aggregation
- `risk_type="cvar"`: CVaR-style aggregation
- `alpha`: tail fraction for VaR / CVaR

The same idea is used across regression, binary, ordinal, and non-Gaussian
acquisition families where pointwise scores are available.

---

## Import examples

```python
from bochan.acquisition.regression.active_learning import qRegressionPosteriorVariance
from bochan.acquisition.regression.levelset_estimation import qRegressionStraddle

from bochan.acquisition.binary.active_learning import qBinaryBALD
from bochan.acquisition.ordinal.active_learning import qOrdinalPredictiveEntropy

from bochan.acquisition.non_gaussian.active_learning import (
    qNonGaussianResponseMeanVariance,
    qNonGaussianBALDProxy,
)
from bochan.acquisition.non_gaussian.levelset_estimation import (
    qNonGaussianStraddle,
    qNonGaussianProbabilityOfExceedance,
)
```

---

## Google-style docstrings

公開 API として使う統一名の acquisition / objective class には、Google スタイルの
docstring を追加する方針です。

確認例:

```python
from bochan.acquisition.binary.bayesian_optimization import qBinaryProbabilityOfFeasibility

help(qBinaryProbabilityOfFeasibility)
```

または Jupyter / VS Code 上で class 名にカーソルを合わせると、主な `Args`、
`Forward Args`、`Returns`、`Notes` が確認できます。

対象は旧名 alias ではなく、統一名のみです。
