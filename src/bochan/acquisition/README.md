# acquisition package

This package contains unified acquisition functions, feasibility wrappers, and
objective classes under `bochan.acquisition`.

The current design separates acquisition functions by model family and task:

- Bayesian optimization
- active learning
- level-set estimation / boundary exploration
- feasibility and constraint handling
- objective / score transformation

Previous class names are not treated as
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
├── multiclass/
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

- `regression/` is the standard Gaussian / continuous-output acquisition family.
- `binary/` is the binary-classification acquisition family.
- `multiclass/` is the unordered multi-class probability acquisition family.
- `ordinal/` is separate from multiclass because ordered cutpoints, boundary
  selection, and expected utility are central to ordinal workflows.
- `non_gaussian/` is a top-level acquisition family, not a subpackage of
  `regression/`. It is intended for Poisson, Beta, Gamma, and Negative Binomial
  GP-style wrappers.
- `feasible/` contains reusable feasibility constraints and wrapper logic for
  constrained acquisition functions.
- Some `non_gaussian/bayesian_optimization` modules are intentionally placeholders
  to keep the directory structure aligned with the other acquisition families.

---

## Naming policy

Only unified names are public. Previous aliases are
intentionally removed unless explicitly documented.

Public acquisition names follow these patterns:

| Family | Naming pattern |
|---|---|
| Gaussian / standard regression | `qRegression...` |
| Binary classification | `qBinary...` |
| Multiclass classification | `qMulticlass...` |
| Ordinal regression | `qOrdinal...` |
| Non-Gaussian regression | `qNonGaussian...` |
| Multi-output regression | `qMultiOutputRegression...` |
| Multi-output binary classification | `qMultiOutputBinary...` |
| Multi-output multiclass classification | `qMultiOutputMulticlass...` |
| Multi-output ordinal regression | `qMultiOutputOrdinal...` |
| Heteroscedastic regression | `qHeteroRegression...` |
| Heteroscedastic binary classification | `qHeteroBinary...` |
| Heteroscedastic multiclass classification | `qHeteroMulticlass...` |
| Heteroscedastic ordinal regression | `qHeteroOrdinal...` |
| Heteroscedastic multi-output regression | `qHeteroMultiOutputRegression...` |
| Heteroscedastic multi-output binary classification | `qHeteroMultiOutputBinary...` |
| Heteroscedastic multi-output multiclass classification | `qHeteroMultiOutputMulticlass...` |
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

- scalarization of regression outputs;
- probability / utility conversion for binary and ordinal models;
- input-perturbation aggregation from `q * n_w` back to `q`;
- risk aggregation such as mean / VaR / CVaR;
- qEHVI / qNEHVI / qNParEGO compatible multi-output transformation.

### Hybrid objective helpers

`objective/hybrid.py` is a thin adapter around the regression objective classes.
It does not reimplement binary, ordinal, or multiclass utility conversion. The
intended responsibility split is:

- `OutputSpec`: output meaning and task-specific conversion metadata
  (`task_type`, `positive_class`, `utility_values`, `target_class`, etc.).
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

## Feasibility and constraints

`acquisition/feasible` contains reusable pieces for constrained workflows:

- constraint helpers that convert model outputs or posterior samples into a
  feasible / infeasible score;
- wrapper logic that can combine a base acquisition value with feasibility
  probabilities or penalties.

Use this layer when the same feasibility definition should be shared by multiple
acquisition functions instead of being reimplemented inside each family-specific
class.

---

## Bayesian optimization

### Standard regression

For standard regression, prefer BoTorch's existing acquisition functions when
possible, for example:

- `qExpectedImprovement`
- `qLogExpectedImprovement`
- `qNoisyExpectedImprovement`
- `qLogNoisyExpectedImprovement`
- `qUpperConfidenceBound`
- `qProbabilityOfImprovement`
- `qKnowledgeGradient`
- `qMultiStepLookahead`
- `qExpectedHypervolumeImprovement`
- `qNoisyExpectedHypervolumeImprovement`
- `qNParEGO`

The local regression BO package is used only where custom behavior is needed,
such as heteroscedastic or wrapper-specific behavior.

### Binary / multiclass / ordinal Bayesian optimization

Classification and ordinal models often require probability / utility conversion
before standard BoTorch BO logic can be applied. These families therefore provide
custom acquisition classes and objectives.

| Family | Single-output BO | Multi-output BO |
|---|---|---|
| Binary | `qBinaryProbabilityOfFeasibility`, `qBinaryExpectedImprovement`, `qBinaryProbabilityOfImprovement`, `qBinaryUpperConfidenceBound` | `qMultiOutputBinaryExpectedHypervolumeImprovement`, `qMultiOutputBinaryNoisyExpectedHypervolumeImprovement`, `qMultiOutputBinaryNParEGO` |
| Multiclass | `qMulticlassProbabilityOfFeasibility`, `qMulticlassExpectedImprovement`, `qMulticlassProbabilityOfImprovement`, `qMulticlassUpperConfidenceBound` | `qMultiOutputMulticlassExpectedHypervolumeImprovement`, `qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement`, `qMultiOutputMulticlassNParEGO`, plus scalar EI / PI / UCB variants |
| Ordinal | `qOrdinalExpectedImprovement`, `qOrdinalProbabilityOfImprovement`, `qOrdinalUpperConfidenceBound`, `qOrdinalProbabilityOfFeasibility` | `qMultiOutputOrdinalExpectedHypervolumeImprovement`, `qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement`, `qMultiOutputOrdinalNParEGO` |

Heteroscedastic variants follow the same naming pattern by adding `Hetero` or
`HeteroMultiOutput` after the leading `q`.

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
| JointBALD / GreedyJointBALD | Select batches using joint or greedy information gain. |
| Variance | Select points with high posterior / probability / utility variance. |
| Margin Uncertainty | Select points near a decision, class, or ordinal boundary. |
| Integrated Posterior Variance | Select points expected to reduce global uncertainty. |

### Regression active learning

Main classes include:

- `qRegressionPredictiveEntropy`
- `qRegressionBALD`
- `qRegressionPosteriorVariance`
- `qRegressionNegIntegratedPosteriorVariance`
- `qRegressionIntegratedPosteriorVarianceProxy`
- `qMultiOutputRegressionPredictiveEntropy`
- `qMultiOutputRegressionBALD`
- `qMultiOutputRegressionPosteriorVariance`
- `qHeteroRegressionPredictiveEntropy`
- `qHeteroRegressionBALD`
- `qHeteroRegressionPosteriorVariance`
- `qHeteroMultiOutputRegressionIntegratedPosteriorVarianceProxy`

The true `qRegressionNegIntegratedPosteriorVariance` delegates to BoTorch's
`qNegIntegratedPosteriorVariance` when available. Proxy variants are intended for
custom models that do not support `fantasize()`.

### Binary active learning

Main classes include:

- `qBinaryPredictiveEntropy`
- `qBinaryBALD`
- `qBinaryJointBALD`
- `qBinaryGreedyJointBALD`
- `qBinaryProbabilityVariance`
- `qBinaryMarginUncertainty`
- `qBinaryFantasyNegIntegratedPosteriorVariance`
- `qMultiOutputBinaryPredictiveEntropy`
- `qMultiOutputBinaryBALD`
- `qMultiOutputBinaryIntegratedPosteriorVarianceProxy`
- `qHeteroBinaryPredictiveEntropy`
- `qHeteroBinaryBALD`
- `qHeteroBinaryIntegratedPosteriorVariance`

### Multiclass active learning

Multiclass active learning works on class-probability predictions over unordered
classes. Main classes include:

- `qMulticlassPredictiveEntropy`
- `qMulticlassBALD`
- `qMulticlassJointBALD`
- `qMulticlassGreedyJointBALD`
- `qMulticlassProbabilityVariance`
- `qMulticlassMarginUncertainty`
- `qMulticlassIntegratedPosteriorVarianceProxy`
- `qMultiOutputMulticlassPredictiveEntropy`
- `qMultiOutputMulticlassBALD`
- `qMultiOutputMulticlassJointBALD`
- `qHeteroMulticlassPredictiveEntropy`
- `qHeteroMulticlassBALD`
- `qHeteroMulticlassIntegratedPosteriorVarianceProxy`
- `qHeteroMultiOutputMulticlassPredictiveEntropy`
- `qHeteroMultiOutputMulticlassBALD`

Multi-output multiclass acquisitions accept output aggregation settings such as
`output_reduction="mean"`, `"sum"`, `"max"`, `"min"`, or `"weighted_mean"`
where implemented by the individual class.

### Ordinal active learning

Ordinal active learning follows the same design idea as binary classification,
but scores are computed from class probabilities, utilities, or ordinal boundary
uncertainty. Main classes include:

- `qOrdinalPredictiveEntropy`
- `qOrdinalBALD`
- `qOrdinalUtilityVariance`
- `qOrdinalMarginUncertainty`
- `qOrdinalFantasyNegIntegratedPosteriorVariance`
- `qMultiOutputOrdinalPredictiveEntropy`
- `qMultiOutputOrdinalBALD`
- `qMultiOutputOrdinalUtilityVariance`
- `qMultiOutputOrdinalIntegratedPosteriorVarianceProxy`
- `qHeteroOrdinalPredictiveEntropy`
- `qHeteroOrdinalBALD`
- `qHeteroOrdinalIntegratedPosteriorVariance`
- `qHeteroMultiOutputOrdinalIntegratedPosteriorVarianceProxy`

### Non-Gaussian active learning

Implemented under `acquisition/non_gaussian/active_learning/`.

Main classes include:

- `qNonGaussianResponseMeanVariance`
- `qNonGaussianPosteriorVariance`
- `qNonGaussianExpectedObservationVariance`
- `qNonGaussianTotalObservationVariance`
- `qNonGaussianExpectedObservationEntropy`
- `qNonGaussianPredictiveEntropyProxy`
- `qNonGaussianBALDProxy`

These classes are intended for latent-GP models with non-Gaussian response-scale
likelihoods. They distinguish latent posterior uncertainty from response mean,
expected observation variance, total observation variance, and entropy-like
proxy scores.

---

## Level-set estimation / boundary exploration

Level-set acquisitions seek points near a threshold, class boundary, ordinal
boundary, or feasibility frontier.

### Regression level-set estimation

Main classes include:

- `qRegressionStraddle`
- `qRegressionJointStraddle`
- `qRegressionICU`
- `qRegressionBoundaryVariance`
- `qRegressionProbabilityOfExceedance`
- `qMultiOutputRegressionStraddle`
- `qMultiOutputRegressionICU`
- `qHeteroRegressionStraddle`
- `qHeteroRegressionBoundaryVariance`
- `qHeteroMultiOutputRegressionProbabilityOfExceedance`

### Binary level-set estimation

Main classes include:

- `qBinaryLatentStraddleAcquisition`
- `qBinaryJointLatentStraddleAcquisition`
- `qBinaryICUAcquisition`
- `qBinaryBoundaryVarianceAcquisition`
- `qBinaryClassEntropyAcquisition`
- `qMultiOutputBinaryLatentStraddleAcquisition`
- `qHeteroBinaryLatentStraddleAcquisition`
- `qHeteroMultiOutputBinaryBoundaryVarianceAcquisition`

### Multiclass level-set estimation

Multiclass level-set estimation is usually based on target-class probability or
class-boundary ambiguity. Main classes include:

- `qMulticlassLatentStraddleAcquisition`
- `qMulticlassJointLatentStraddleAcquisition`
- `qMulticlassICUAcquisition`
- `qMulticlassBoundaryVarianceAcquisition`
- `qMulticlassClassEntropyAcquisition`
- `qMulticlassProbabilityOfExceedance`
- `qMulticlassLevelSetUncertainty`
- `qMultiOutputMulticlassLatentStraddleAcquisition`
- `qMultiOutputMulticlassProbabilityOfExceedance`
- `qHeteroMulticlassLevelSetUncertainty`
- `qHeteroMultiOutputMulticlassLevelSetUncertainty`

### Ordinal level-set estimation

Ordinal level-set estimation can target ordered boundaries such as class `1 | 2`
or an expected-utility threshold. Main classes include:

- `qOrdinalLatentStraddleAcquisition`
- `qOrdinalJointLatentStraddleAcquisition`
- `qOrdinalICUAcquisition`
- `qOrdinalBoundaryVarianceAcquisition`
- `qOrdinalClassEntropyAcquisition`
- `qMultiOutputOrdinalLatentStraddleAcquisition`
- `qHeteroOrdinalLatentStraddleAcquisition`
- `qHeteroMultiOutputOrdinalProbabilityOfExceedance`
- `qHeteroMultiOutputOrdinalLevelSetUncertainty`
- `qHeteroMultiOutputOrdinalStraddle`
- `qHeteroMultiOutputOrdinalBoundaryVariance`

### Non-Gaussian level-set estimation

Implemented under `acquisition/non_gaussian/levelset_estimation/`.

Main classes include:

- `qNonGaussianStraddle`
- `qNonGaussianBoundaryVariance`
- `qNonGaussianICU`
- `qNonGaussianProbabilityOfExceedance`

These classes operate on response-scale quantities, but may use
`latent_posterior(X)` plus likelihood transformation internally when response
uncertainty must be estimated from latent GP samples.

---

## Contextual aliases used by the API

The high-level API resolves short names according to `task_type`, `model_type`,
and whether the fitted model is multi-output.

Examples:

```python
AcquisitionConfig(name="BALD")       # qRegressionBALD / qBinaryBALD / qMulticlassBALD / qOrdinalBALD
AcquisitionConfig(name="Variance")   # posterior / probability / utility / response variance by family
AcquisitionConfig(name="Straddle")   # regression straddle or latent classification / ordinal straddle
AcquisitionConfig(name="EI")         # BoTorch qEI or family-specific EI
```

Supported contextual short names include:

```python
"BALD"
"JointBALD"
"GreedyJointBALD"
"PredictiveEntropy"
"Entropy"
"Variance"
"PosteriorVariance"
"NIPV"
"Margin"
"MarginUncertainty"
"Straddle"
"JointStraddle"
"ICU"
"BoundaryVariance"
"ClassEntropy"
"ProbabilityOfExceedance"
"PoE"
"LevelSetUncertainty"
"LevelSet"
"EI"
"PI"
"UCB"
"PoF"
"EHI"
"EHVI"
"NEHVI"
"NParEGO"
"KG"
"MultiStepLookahead"
"Lookahead"
```

Important restrictions:

- `KG`, `qKG`, `MultiStepLookahead`, and `Lookahead` are regression / hybrid
  aliases only.
- `EHI`, `EHVI`, `NEHVI`, and `NParEGO` require a multi-output setting when they
  resolve to binary / multiclass / ordinal or heteroscedastic multi-output
  classes.
- `PoF` resolves only for binary, multiclass, and ordinal probability-style BO.
- `NIPV` maps to the best available true or proxy implementation for each family.

---

## Implementation guidance

When adding a new acquisition function:

1. Put the class under the correct family and task directory.
2. Use the public naming scheme, for example `qMulticlass...` or
   `qHeteroMultiOutputOrdinal...`.
3. Keep `forward(X)` q-batch safe and return a score shaped like BoTorch
   acquisitions expect.
4. Prefer shared utilities for posterior sampling, output reduction, risk
   aggregation, and feasibility handling.
5. Register the class in `bochan.api.acquisition_registry` if it should be
   available from `AcquisitionConfig(name=...)`.
6. Avoid reimplementing BoTorch standard acquisitions unless the family requires
   probability, utility, response-scale, or heteroscedastic behavior that the
   standard class does not provide directly.
