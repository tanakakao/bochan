# acquisition package

This package contains unified acquisition functions under `acquisition/`.

## Layout

```text
acquisition/
├── objective/
├── regression/
│   ├── bayesian_optimization/
│   ├── active_learning/
│   └── levelset_estimation/
├── classification/
│   ├── bayesian_optimization/
│   ├── active_learning/
│   └── levelset_estimation/
└── ordinal/
    ├── bayesian_optimization/
    ├── active_learning/
    └── levelset_estimation/
```

## Naming policy

Only unified names are public. Legacy class names and compatibility aliases are intentionally removed.

Public acquisition names follow these patterns:

- `qRegression...`
- `qBinary...`
- `qOrdinal...`
- `qMultiOutputRegression...`
- `qMultiOutputBinary...`
- `qMultiOutputOrdinal...`
- `qHeteroRegression...`
- `qHeteroBinary...`
- `qHeteroOrdinal...`
- `qHeteroMultiOutputRegression...`
- `qHeteroMultiOutputBinary...`
- `qHeteroMultiOutputOrdinal...`

Objective classes are placed in `acquisition/objective/`.

## Bayesian Optimization: multi-output

| Family | Regression | Classification | Ordinal |
|---|---|---|---|
| Utility / Feasibility | BoTorch standard | `qMultiOutputBinaryProbabilityOfFeasibility` | `qMultiOutputOrdinalUtilityObjective` |
| qEHVI | BoTorch standard | `qMultiOutputBinaryExpectedHypervolumeImprovement` | `qMultiOutputOrdinalExpectedHypervolumeImprovement` |
| qNEHVI | BoTorch standard | `qMultiOutputBinaryNoisyExpectedHypervolumeImprovement` | `qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement` |
| qNParEGO | BoTorch standard | `qMultiOutputBinaryNParEGO` | `qMultiOutputOrdinalNParEGO` |
| Hetero qEHVI | `qHeteroMultiOutputRegressionExpectedHypervolumeImprovement` | `qHeteroMultiOutputBinaryExpectedHypervolumeImprovement` | `qHeteroMultiOutputOrdinalExpectedHypervolumeImprovement` |
| Hetero qNEHVI | `qHeteroMultiOutputRegressionNoisyExpectedHypervolumeImprovement` | `qHeteroMultiOutputBinaryNoisyExpectedHypervolumeImprovement` | `qHeteroMultiOutputOrdinalNoisyExpectedHypervolumeImprovement` |
| Hetero qNParEGO | `qHeteroMultiOutputRegressionNParEGO` | `qHeteroMultiOutputBinaryNParEGO` | `qHeteroMultiOutputOrdinalNParEGO` |

## Objective examples

```python
from acquisition.objective import (
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

## Active Learning families

| Family | Meaning |
|---|---|
| Predictive Entropy | Select points with ambiguous predictions. |
| BALD / MI | Select points with high model-information gain. |
| Variance | Select points with high posterior / probability / utility variance. |
| Margin Uncertainty | Select points near a decision or ordinal boundary. |
| Integrated Posterior Variance | Select points expected to reduce global uncertainty. |

## Level-set Estimation families

| Family | Meaning |
|---|---|
| Straddle | Select points near a target boundary and with high uncertainty. |
| Joint Straddle | Batch-aware straddle using joint uncertainty. |
| ICU | Integrated contour uncertainty around a boundary. |
| Boundary Variance | Variance weighted by boundary proximity. |
| Class Entropy | Entropy-based boundary exploration for classification / ordinal models. |

## Heteroscedastic variants

Hetero acquisition functions apply noise-aware or robust scoring such as:

```text
adjusted_sample = mean + beta * (sample - mean) - noise_penalty * sigma_noise
```

Use hetero variants when the observation noise depends on the input or when noisy regions should be avoided.

---

## Google-style docstrings

この版では、公開 API として使う統一名の acquisition / objective class に Google スタイルの docstring を追加しています。

確認例:

```python
from acquisition.classification.bayesian_optimization import qBinaryProbabilityOfFeasibility

help(qBinaryProbabilityOfFeasibility)
```

または Jupyter / VS Code 上で class 名にカーソルを合わせると、主な `Args`、`Forward Args`、`Returns`、`Notes` が確認できます。

対象は旧名 alias ではなく、統一名のみです。
