# bochan API

`bochan.api` は、BoTorch / GPyTorch ベースのモデル構築・学習・獲得関数生成・候補点最適化を、アプリケーションや外部 API から扱いやすい形にまとめるための高レベル API です。

研究・開発段階では関数単位で細かく扱い、アプリ化・API化するときは `BayesianOptimizer` クラスから文字列中心で操作することを想定しています。

---

## 1. 設計方針

内部処理は、基本的に次の4段階に分かれます。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)

acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer` は、この流れを薄く包むクラスです。

```python
bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
    model_registry=MODEL_REGISTRY,
)

bo.fit(train_X, train_Y)

posterior = bo.predict(test_X)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
)
```

### 1.1 API の責務分離

| 設定クラス | 役割 |
|---|---|
| `ModelConfig` | モデルの種類、カテゴリ変数、input transform、モデル固有引数を指定する |
| `FitConfig` | 学習回数、learning rate、maxiter などを指定する |
| `InputTransformConfig` | Normalize / input perturbation を簡易指定する |
| `MultiOutputConfig` | multi-output / hybrid の出力ごとのモデル構成を指定する |
| `AcquisitionConfig` | 獲得関数を文字列またはクラスで指定する |
| `DataContext` | `X_baseline`, `best_f`, `ref_point`, `partitioning` など獲得関数側の文脈を渡す |
| `MultiObjectiveConfig` | EHVI / NEHVI / NParEGO などの多目的設定を渡す |
| `OptimizeConfig` | 候補点最適化の q, restart 数, optimizer 種類, 制約を指定する |
| `CandidateRepairConfig` | grid rounding, k-sparse, 制約補修用の post-processing を自動生成する |

### 1.2 目標とする外部 API

外部 API では、基本的に **文字列 + 最小限の kwargs** で指定できるようにします。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="deepkernel",
    cat_dims=[2],
    model_kwargs={
        "feature_extractor": feature_extractor,
        "num_inducing_points": 64,
    },
)

fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)

acq_config = AcquisitionConfig(name="BALD")

opt_config = OptimizeConfig(
    optimizer="torch",
    q=3,
    num_restarts=10,
    raw_samples=512,
    optimizer_kwargs={
        "method": "adam",
        "options": {
            "lr": 0.03,
            "num_steps": 200,
        },
    },
)
```

---

## 2. 最小例: regression + EI

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)

fit_config = FitConfig(
    maxiter=128,
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
    model_registry=MODEL_REGISTRY,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "best_f": train_Y.max(),
    },
)

opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

`name="EI"` は regression では BoTorch の `qExpectedImprovement` に解決されます。

---

## 3. model registry

`ModelConfig` では、`task_type` と `model_type` を文字列で指定します。
実際のクラス解決には `model_registry` を使います。

### 3.1 nested registry

推奨は nested registry です。

```python
MODEL_REGISTRY = {
    "normal": {
        "regression": {
            "base": SingleTaskGP,
            "deepgp": DeepGPModel,
            "deepkernel": DeepKernelGPModel,
            "deepgpdeepkernel": DeepKernelDeepGPModel,
            "saas": SaasSingleTaskGP,
            "pca": PCASingleTaskGP,
            "rembo": REMBOSingleTaskGP,
            "rrp": SafeRobustRelevancePursuitSingleTaskGP,
            "hetero": HeteroscedasticSingleTaskGP,
        },
        "binary": {
            "base": BinaryClassificationGPModel,
            "deepgp": BinaryClassificationDeepGPModel,
            "deepkernel": DeepKernelBinaryClassificationGPModel,
            "deepgpdeepkernel": DeepKernelBinaryClassificationDeepGPModel,
            "saas": SaasBinaryClassificationGPModel,
            "pca": PCABinaryClassificationGPModel,
            "rembo": REMBOBinaryClassificationGPModel,
            "rrp": OutlierRelevancePursuitBinaryClassificationGPModel,
            "hetero": HeteroscedasticBinaryClassificationGPModel,
        },
        "ordinal": {
            "base": OrdinalGPModel,
            "deepgp": OrdinalDeepGPModel,
            "deepkernel": DeepKernelOrdinalGPModel,
            "deepgpdeepkernel": DeepKernelOrdinalDeepGPModel,
            "saas": SaasOrdinalGPModel,
            "pca": PCAOrdinalGPModel,
            "rembo": REMBOOrdinalGPModel,
            "rrp": OutlierRelevancePursuitOrdinalGPModel,
            "hetero": HeteroscedasticOrdinalGPModel,
        },
        "multiclass": {
            "base": MulticlassClassificationGPModel,
        },
    },
    "mixed": {
        "regression": {
            "base": MixedSingleTaskGP,
            "deepgp": DeepMixedGPModel,
            "deepkernel": DeepKernelMixedGPModel,
            "deepgpdeepkernel": DeepKernelDeepMixedGPModel,
            "saas": SaasMixedSingleTaskGP,
            "pca": PCAMixedSingleTaskGP,
            "rembo": REMBOMixedSingleTaskGP,
            "rrp": SafeRobustRelevancePursuitMixedSingleTaskGP,
            "hetero": HeteroscedasticMixedSingleTaskGP,
        },
        "binary": {
            "base": BinaryClassificationMixedGPModel,
            "deepgp": BinaryClassificationMixedDeepGPModel,
            "deepkernel": DeepKernelBinaryClassificationMixedGPModel,
            "deepgpdeepkernel": DeepKernelBinaryClassificationMixedDeepGPModel,
            "saas": SaasBinaryClassificationMixedGPModel,
            "pca": PCABinaryClassificationMixedGPModel,
            "rembo": REMBOBinaryClassificationMixedGPModel,
            "rrp": OutlierRelevancePursuitBinaryClassificationMixedGPModel,
            "hetero": HeteroscedasticBinaryClassificationMixedGPModel,
        },
        "ordinal": {
            "base": OrdinalMixedGPModel,
            "deepgp": OrdinalMixedDeepGPModel,
            "deepkernel": DeepKernelOrdinalMixedGPModel,
            "deepgpdeepkernel": DeepKernelOrdinalMixedGPModel,
            "saas": SaasOrdinalMixedGPModel,
            "pca": PCAOrdinalMixedGPModel,
            "rembo": REMBOOrdinalMixedGPModel,
            "rrp": OutlierRelevancePursuitOrdinalMixedGPModel,
            "hetero": HeteroscedasticOrdinalMixedGPModel,
        },
    },
}
```

`cat_dims` が空なら `"normal"`、非空なら `"mixed"` 側からクラスを解決します。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    cat_dims=[2],
)
```

この場合は、以下を参照します。

```python
MODEL_REGISTRY["mixed"]["binary"]["base"]
```

### 3.2 flat registry

flat registry にも対応しています。

```python
MODEL_REGISTRY = {
    ("normal", "regression", "base"): SingleTaskGP,
    ("mixed", "regression", "base"): MixedSingleTaskGP,
    ("normal", "binary", "base"): BinaryClassificationGPModel,
}
```

---

## 4. `ModelConfig`

### 4.1 基本形

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)
```

標準的な `task_type` は次です。

```python
"regression"
"binary"
"multiclass"
"ordinal"
"hybrid"
"multi_objective"
```

標準的な `model_type` は次です。

```python
"base"
"deepgp"
"deepkernel"
"deepgpdeepkernel"
"saas"
"pca"
"rembo"
"rrp"
"hetero"
```

### 4.2 model 固有引数

モデル固有の引数は `model_kwargs` に渡します。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={
        "num_inducing_points": 64,
    },
)
```

### 4.3 mixed model

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[2, 5],
)
```

`cat_dims` があると `input_type="mixed"` として扱われます。

### 4.4 deepkernel

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="deepkernel",
    model_kwargs={
        "feature_extractor": feature_extractor,
        "feature_dim": 8,
    },
)
```

### 4.5 PCA / REMBO

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="pca",
    model_kwargs={
        "latent_dim": 5,
        "standardize_X": True,
    },
)
```

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="rembo",
    model_kwargs={
        "latent_dim": 4,
        "projection_matrix": projection_matrix,
    },
)
```

### 4.6 model class を直接渡す高度な使い方

registry を使わずにモデルクラスを直接渡すこともできます。

```python
model_config = ModelConfig(
    model_cls=SingleTaskGP,
    task_type="regression",
    model_type="base",
)
```

通常の API 利用では、`model_cls` よりも `model_registry` + 文字列指定を推奨します。

---

## 5. `FitConfig`

`FitConfig` は、基本的には epoch 数や learning rate だけを指定します。
`fit_func`, `mll_factory`, `mll_cls` は、通常は指定不要です。

### 5.1 exact GP

```python
fit_config = FitConfig(
    maxiter=128,
)
```

`regression`, `base` などの exact GP では、内部で `ExactMarginalLogLikelihood` + `fit_gpytorch_mll` が使われます。

### 5.2 classification / ordinal / deep 系

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)
```

binary / ordinal / multiclass / deep GP / deep kernel などでは、`task_type` と `model_type` に応じて専用 fit helper が選ばれます。

### 5.3 fit の自動解決

| 条件 | 自動選択 |
|---|---|
| `task_type="regression"` かつ exact GP | `ExactMarginalLogLikelihood` + `fit_gpytorch_mll` |
| `task_type="binary"` | `VariationalELBO` + `fit_binary_classifier_mll` |
| `task_type="binary", model_type="rrp"` | `fit_rrp_binary_classifier_mll` |
| `task_type="ordinal"` | `make_ordinal_mll` + `fit_ordinal_mll` |
| `task_type="ordinal", model_type="rrp"` | `fit_rrp_ordinal_mll` |
| `task_type="multiclass"` | `VariationalELBO` + `fit_multiclass_mll` |
| `model_type` に `deepgp` を含む | `fit_deepgp_mll` |
| `model_type` に `deepkernel` を含む | `fit_deepkernel_mll` |
| model に `make_mll()` がある | `model.make_mll()` を優先 |

### 5.4 高度な上書き

必要なら従来通り明示指定できます。

```python
fit_config = FitConfig(
    mll_factory=custom_mll_factory,
    fit_func=custom_fit_func,
    fit_kwargs={
        "num_epochs": 500,
    },
)
```

---

## 6. input transform / input perturbation

`InputTransformConfig` を使うと、`build_input_transform(...)` を API 側で自動生成できます。

### 6.1 入力摂動なし

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        perturbation=False,
    ),
)
```

### 6.2 入力摂動あり

```python
N_W = 8
STD_DEV = 0.1

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        perturbation=True,
        n_w=N_W,
        std=STD_DEV,
    ),
)
```

これは内部的に以下と同等です。

```python
from bochan.models.transforms.input import build_input_transform

intf = build_input_transform(
    train_X=train_X,
    bounds=torch.cat(
        [
            train_X.min(dim=0, keepdim=True).values,
            train_X.max(dim=0, keepdim=True).values,
        ],
        dim=0,
    ),
    perturbation=True,
    categorical_idx=None,
    n_w=8,
    std=0.1,
)
```

### 6.3 mixed model での input perturbation

`cat_dims` がある場合は、`categorical_idx` として自動的に使われます。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    cat_dims=[2],
    input_transform_config=InputTransformConfig(
        perturbation=True,
        n_w=8,
        std=0.1,
    ),
)
```

内部では以下のように呼ばれます。

```python
build_input_transform(
    train_X=train_X,
    bounds=bounds,
    perturbation=True,
    categorical_idx=[2],
    n_w=8,
    std=0.1,
)
```

### 6.4 明示的に bounds を渡す

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        perturbation=True,
        n_w=8,
        std=0.1,
        bounds=bounds,
    ),
)
```

### 6.5 input perturbation と acquisition objective

入力摂動を使うと、候補点 `q` が内部的に `q * n_w` に展開されます。
そのため、獲得関数側で `q * n_w -> q` に戻す objective が必要です。

回帰では、例えば `RegressionScalarObjective` を使います。

```python
from bochan.acquisition.objective import RegressionScalarObjective

objective = RegressionScalarObjective(
    n_w=8,
    risk_type=None,  # None / "var" / "cvar"
    alpha=0.8,
    maximize=True,
)

acq_config = AcquisitionConfig(
    name="EI",
    objective=objective,
    acqf_kwargs={
        "best_f": train_Y.max(),
    },
)
```

CVaR を使う場合は次です。

```python
objective = RegressionScalarObjective(
    n_w=8,
    risk_type="cvar",
    alpha=0.8,
    maximize=True,
)
```

---

## 7. acquisition registry

`AcquisitionConfig.name` は文字列で指定できます。

```python
acq_config = AcquisitionConfig(name="EI")
```

`acqf_cls` を直接渡すこともできますが、通常の API 利用では文字列指定を推奨します。

```python
acq_config = AcquisitionConfig(
    name="qEI",
    acqf_cls=qExpectedImprovement,
)
```

### 7.1 BoTorch 系 alias

| name | 解決先 |
|---|---|
| `EI`, `qEI`, `ExpectedImprovement` | `qExpectedImprovement` |
| `LogEI`, `qLogEI` | `qLogExpectedImprovement` |
| `NEI`, `qNEI` | `qNoisyExpectedImprovement` |
| `PI`, `qPI` | `qProbabilityOfImprovement` |
| `UCB`, `qUCB` | `qUpperConfidenceBound` |
| `KG`, `qKG` | `qKnowledgeGradient` |
| `MultiStepLookahead`, `qMultiStepLookahead`, `Lookahead` | `qMultiStepLookahead` |
| `EHI`, `EHVI`, `qEHVI` | `qExpectedHypervolumeImprovement` |
| `NEHVI`, `qNEHVI` | `qNoisyExpectedHypervolumeImprovement` |
| `NParEGO`, `qNParEGO` | `qExpectedImprovement` + scalarization objective |

### 7.2 task context に応じる短縮 alias

`BayesianOptimizer` 経由では、`bundle.task_type` を見て短縮名を自動解決します。

```python
AcquisitionConfig(name="BALD")
```

| task_type | `BALD` の解決先 |
|---|---|
| `regression` | `qRegressionBALD` |
| `binary` | `qBinaryBALD` |
| `ordinal` | `qOrdinalBALD` |

同様に、以下も task context に応じて解決されます。

```python
"BALD"
"PredictiveEntropy"
"Entropy"
"Variance"
"PosteriorVariance"
"Margin"
"MarginUncertainty"
"Straddle"
"JointStraddle"
"ICU"
"BoundaryVariance"
"ClassEntropy"
"ProbabilityOfExceedance"
"PoE"
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
```

### 7.3 binary の例

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
)

acq_config = AcquisitionConfig(name="BALD")
# -> qBinaryBALD
```

```python
acq_config = AcquisitionConfig(name="EI")
# -> qBinaryExpectedImprovement
```

### 7.4 ordinal の例

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
)

acq_config = AcquisitionConfig(name="BALD")
# -> qOrdinalBALD
```

```python
acq_config = AcquisitionConfig(name="EI")
# -> qOrdinalExpectedImprovement
```

### 7.5 hybrid の例

`task_type="hybrid"` の場合、獲得関数側では regression として解決されます。

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(...),
)

AcquisitionConfig(name="EI")
# -> BoTorch qExpectedImprovement

AcquisitionConfig(name="EHI")
# -> BoTorch qExpectedHypervolumeImprovement

AcquisitionConfig(name="KG")
# -> BoTorch qKnowledgeGradient
```

`KG` / `MultiStepLookahead` は regression / hybrid 用です。
binary / ordinal で短縮指定した場合は、誤解決を避けるためエラーになります。

### 7.6 登録済み獲得関数名を確認する

```python
from bochan.api import available_acqf_names

names = available_acqf_names()
```

---

## 8. 単目的例

### 8.1 regression + UCB

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)

fit_config = FitConfig(maxiter=128)

acq_config = AcquisitionConfig(
    name="UCB",
    acqf_kwargs={
        "beta": 0.2,
    },
)
```

### 8.2 binary + BALD

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={
        "num_inducing_points": 64,
    },
)

fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)

acq_config = AcquisitionConfig(
    name="BALD",
)
```

### 8.3 binary + EI

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "best_f": best_f,
    },
)
```

binary model では `qBinaryExpectedImprovement` に解決されます。

### 8.4 ordinal + MarginUncertainty

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    model_kwargs={
        "num_classes": 3,
    },
)

fit_config = FitConfig(
    num_epochs=500,
    lr=0.03,
)

acq_config = AcquisitionConfig(
    name="Margin",
)
```

ordinal model では `qOrdinalMarginUncertainty` に解決されます。

---

## 9. multi-output / multi-objective

### 9.1 homogeneous multi-output regression

`train_Y` が `n x m` の場合、`MultiOutputConfig` を使って出力ごとに submodel を作ります。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            "regression",
            "regression",
            "regression",
        ],
    ),
)
```

### 9.2 binary multi-output

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            "binary",
            "binary",
            "binary",
        ],
    ),
)
```

### 9.3 output ごとに詳細指定

```python
model_config = ModelConfig(
    task_type="binary",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {
                "name": "defect_a",
                "task_type": "binary",
                "model_type": "base",
                "model_kwargs": {
                    "num_inducing_points": 64,
                },
            },
            {
                "name": "defect_b",
                "task_type": "binary",
                "model_type": "deepkernel",
                "model_kwargs": {
                    "feature_extractor": feature_extractor,
                },
            },
        ],
    ),
)
```

### 9.4 qEHVI

```python
mo_config = MultiObjectiveConfig(
    ref_point=ref_point,
    Y_baseline=train_Y,
    auto_partitioning=True,
)

data_context = DataContext(
    bounds=bounds,
    X_baseline=train_X,
    multi_objective=mo_config,
)

acq_config = AcquisitionConfig(name="EHI")

opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
)
```

`ref_point` と `Y_baseline` があれば、`partitioning` は自動生成されます。

### 9.5 qNEHVI

```python
mo_config = MultiObjectiveConfig(
    ref_point=ref_point,
    Y_baseline=train_Y,
)

data_context = DataContext(
    bounds=bounds,
    X_baseline=train_X,
    multi_objective=mo_config,
)

acq_config = AcquisitionConfig(name="NEHVI")

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
)
```

### 9.6 NParEGO

```python
mo_config = MultiObjectiveConfig(
    Y_baseline=train_Y,
    scalarization_weights=weights,
    auto_scalarization=True,
)

data_context = DataContext(
    bounds=bounds,
    X_baseline=train_X,
    best_f=best_f_scalarized,
    multi_objective=mo_config,
)

acq_config = AcquisitionConfig(name="NParEGO")

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
)
```

---

## 10. hybrid model

hybrid は、regression / binary / ordinal など異なる task type を同時に扱うための multi-output wrapper です。

### 10.1 文字列だけで指定する

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            "regression",
            "binary",
            "ordinal",
        ],
        use_hybrid=True,
    ),
)
```

この場合、内部では各出力が次のように解釈されます。

```text
"regression" -> task_type="regression", model_type="base"
"binary"     -> task_type="binary",     model_type="base"
"ordinal"    -> task_type="ordinal",    model_type="base"
```

### 10.2 出力ごとに詳細指定する

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {
                "name": "strength",
                "task_type": "regression",
                "model_type": "base",
                "output_spec_kwargs": {
                    "sign": 1.0,
                },
            },
            {
                "name": "defect",
                "task_type": "binary",
                "model_type": "base",
                "model_kwargs": {
                    "num_inducing_points": 64,
                },
                "output_spec_kwargs": {
                    "sign": -1.0,
                    "positive_class": 1,
                },
            },
            {
                "name": "rank",
                "task_type": "ordinal",
                "model_type": "base",
                "model_kwargs": {
                    "num_classes": 3,
                },
                "output_spec_kwargs": {
                    "sign": 1.0,
                    "utility_values": [0.0, 1.0, 2.0],
                },
            },
        ],
        use_hybrid=True,
    ),
)
```

### 10.3 出力ごとに fit 設定を変える

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {
                "name": "strength",
                "task_type": "regression",
                "fit_config": FitConfig(maxiter=128),
            },
            {
                "name": "defect",
                "task_type": "binary",
                "fit_config": FitConfig(num_epochs=300, lr=0.01),
            },
            {
                "name": "rank",
                "task_type": "ordinal",
                "model_kwargs": {
                    "num_classes": 3,
                },
                "fit_config": FitConfig(num_epochs=800, lr=0.03),
            },
        ],
        use_hybrid=True,
    ),
)
```

---

## 11. candidate optimizer

`OptimizeConfig.optimizer` で最適化手法を選べます。

### 11.1 BoTorch 標準

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf",
    q=3,
    num_restarts=10,
    raw_samples=256,
)
```

### 11.2 BoTorch mixed

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=3,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=fixed_features_list,
)
```

### 11.3 evo optimizer

```python
opt_config = OptimizeConfig(
    optimizer="evo",
    q=3,
    num_restarts=10,
    raw_samples=256,
    optimizer_kwargs={
        "method": "ga",  # "ga", "pso", "sa", "cmaes"
        "options": {
            "pop_size": 128,
            "num_generations": 100,
            "penalty_factor": 1e3,
        },
    },
)
```

`optimizer="evo"` は `bochan.optim.optimize_acqf_evo` を使います。

### 11.4 torch optimizer

```python
opt_config = OptimizeConfig(
    optimizer="torch",
    q=3,
    num_restarts=10,
    raw_samples=512,
    optimizer_kwargs={
        "method": "adam",  # "adam", "adamw", "sgd", "rmsprop", "lbfgs"
        "options": {
            "lr": 0.03,
            "num_steps": 200,
            "penalty_factor": 1e3,
        },
    },
)
```

`optimizer="torch"` は `bochan.optim.optimize_acqf_torch` を使います。

### 11.5 evo mixed

```python
opt_config = OptimizeConfig(
    optimizer="evo_mixed",
    q=3,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=fixed_features_list,
    optimizer_kwargs={
        "method": "ga",
        "categorical_features": {
            2: [0, 1, 2],
            5: [0, 1],
        },
        "enumerate_categorical_features": True,
        "options": {
            "pop_size": 128,
            "num_generations": 100,
        },
    },
)
```

### 11.6 torch mixed

```python
opt_config = OptimizeConfig(
    optimizer="torch_mixed",
    q=3,
    num_restarts=10,
    raw_samples=512,
    fixed_features_list=fixed_features_list,
    optimizer_kwargs={
        "method": "adam",
        "categorical_features": {
            2: [0, 1, 2],
            5: [0, 1],
        },
        "options": {
            "lr": 0.03,
            "num_steps": 200,
        },
    },
)
```

### 11.7 optimizer 一覧

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="optimize_acqf_mixed")
OptimizeConfig(optimizer="evo")
OptimizeConfig(optimizer="optimize_acqf_evo")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="optimize_acqf_torch")
OptimizeConfig(optimizer="evo_mixed")
OptimizeConfig(optimizer="optimize_acqf_evo_mixed")
OptimizeConfig(optimizer="torch_mixed")
OptimizeConfig(optimizer="optimize_acqf_torch_mixed")
```

---

## 12. 制約・repair・grid rounding・k-sparse

BoTorch の制約は `OptimizeConfig` に直接渡せます。

```python
opt_config = OptimizeConfig(
    q=3,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
)
```

さらに、`CandidateRepairConfig` を使うと、`make_grid_k_sparse_post_processing_func(...)` を API 側で自動生成できます。

### 12.1 grid rounding + k-sparse + constraints

```python
opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
    repair_config=CandidateRepairConfig(
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        k=k,
        final_priority="grid",
    ),
)
```

これは内部的に以下と同等です。

```python
repair = make_grid_k_sparse_post_processing_func(
    bounds=bounds,
    numeric_indices=numeric_indices,
    steps=steps,
    comp_idx=comp_idx,
    k=k,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
)
```

### 12.2 丸めだけ

```python
opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=[],
        k=0,
    ),
)
```

### 12.3 k-sparse だけ

```python
opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        comp_idx=comp_idx,
        k=3,
        steps=None,
    ),
)
```

### 12.4 repair 側に制約を書く

```python
opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        k=k,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
    ),
)
```

優先順位は次です。

```text
CandidateRepairConfig.equality_constraints
    -> None なら OptimizeConfig.equality_constraints

CandidateRepairConfig.inequality_constraints
    -> None なら OptimizeConfig.inequality_constraints

CandidateRepairConfig.fixed_features
    -> None なら OptimizeConfig.fixed_features
```

### 12.5 追加オプション

```python
repair_config = CandidateRepairConfig(
    numeric_indices=numeric_indices,
    steps=steps,
    comp_idx=comp_idx,
    k=k,
    inequality_sense="le",
    fixed_features=fixed_features,
    final_sum_constraint=final_sum_constraint,
    diversify=True,
    diversify_kwargs={"noise_scale": 0.01},
    score="abs",
    support_selection="topk",
    sample_tau=0.2,
    sample_eps=0.05,
    max_iters=12,
    num_alternations=2,
    final_priority="grid",
    support_eps=0.0,
)
```

### 12.6 optimizer との併用

```python
opt_config = OptimizeConfig(
    optimizer="evo",
    q=3,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
    repair_config=CandidateRepairConfig(
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        k=k,
    ),
    optimizer_kwargs={
        "method": "ga",
        "options": {
            "pop_size": 128,
            "num_generations": 100,
            "penalty_factor": 1e3,
        },
    },
)
```

---

## 13. `DataContext`

`DataContext` は獲得関数生成時に必要な補助情報を渡すための入れ物です。

```python
data_context = DataContext(
    bounds=bounds,
    X_baseline=train_X,
    X_pending=X_pending,
    Y_baseline=train_Y,
    best_f=train_Y.max(),
    ref_point=ref_point,
    partitioning=partitioning,
    constraints=constraints,
)
```

`BayesianOptimizer` に `bounds` と `train_X` があれば、最低限の `DataContext` は内部で補完されます。

```python
bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
    model_registry=MODEL_REGISTRY,
)

bo.fit(train_X, train_Y)

# DataContext を明示しなくても bounds / X_baseline は補完される
candidates, acq_value = bo.candidate(acq_config, opt_config)
```

---

## 14. ask / tell 形式

```python
candidates, acq_value = bo.ask(acq_config, opt_config)

new_Y = evaluate(candidates)

bo.tell(candidates, new_Y, refit=True)
```

`bo.tell(...)` は内部の `train_X`, `train_Y` に新しい観測を追加し、`refit=True` なら再学習します。

---

## 15. 複数獲得関数の比較

同じ学習済みモデルに対して複数の獲得関数を比較できます。

```python
results = bo.compare_acquisitions(
    acq_configs=[
        AcquisitionConfig(name="EI", acqf_kwargs={"best_f": train_Y.max()}),
        AcquisitionConfig(name="UCB", acqf_kwargs={"beta": 0.2}),
        AcquisitionConfig(name="Straddle", acqf_kwargs={"threshold": 0.0}),
    ],
    opt_config=opt_config,
)

for name, result in results.items():
    print(name, result.candidates, result.acq_value)
```

---

## 16. 推奨パターン

### 16.1 アプリ / API 用

アプリや外部 API では、できるだけ文字列指定に寄せます。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    cat_dims=[2],
    input_transform_config=InputTransformConfig(
        perturbation=True,
        n_w=8,
        std=0.1,
    ),
    model_kwargs={
        "num_inducing_points": 64,
    },
)

fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)

acq_config = AcquisitionConfig(name="BALD")

opt_config = OptimizeConfig(
    optimizer="torch",
    q=3,
    num_restarts=10,
    raw_samples=512,
    optimizer_kwargs={
        "method": "adam",
        "options": {
            "lr": 0.03,
            "num_steps": 200,
        },
    },
)
```

### 16.2 研究 / デバッグ用

研究やデバッグでは、直接クラス・関数を渡して細かく制御できます。

```python
model_config = ModelConfig(
    model_cls=SingleTaskGP,
    task_type="regression",
)

fit_config = FitConfig(
    mll_cls=ExactMarginalLogLikelihood,
    fit_func=fit_gpytorch_mll,
)

acq_config = AcquisitionConfig(
    name="custom_ei",
    acqf_cls=qExpectedImprovement,
    acqf_kwargs={
        "best_f": train_Y.max(),
    },
)
```

---

## 17. よくある注意点

### 17.1 `model_type="base"` と `model_cls` は両方必要か

通常は両方は不要です。

アプリ/API では次を推奨します。

```python
ModelConfig(
    task_type="regression",
    model_type="base",
)
```

この場合は `model_registry` から `SingleTaskGP` などを解決します。

`model_cls` は registry を使わず直接指定したい場合だけ使います。

```python
ModelConfig(
    task_type="regression",
    model_cls=SingleTaskGP,
)
```

### 17.2 `FitConfig` に `fit_func` は必要か

通常は不要です。

```python
FitConfig(maxiter=128)
```

または、

```python
FitConfig(num_epochs=300, lr=0.01)
```

だけで十分です。

### 17.3 `input_transform` と `input_transform_config` の優先順位

`input_transform` を直接渡した場合は、それが優先されます。
`input_transform_config` は無視されます。

```python
ModelConfig(
    input_transform=manual_transform,
    input_transform_config=InputTransformConfig(perturbation=True),
)
```

この場合は `manual_transform` が使われます。

### 17.4 入力摂動を使ったら objective が必要か

多くの場合は必要です。
入力摂動では `q` が内部で `q * n_w` に展開されるため、獲得関数側で `q * n_w -> q` に戻す objective / risk objective が必要です。

### 17.5 `EHI` / `EHVI` / `NEHVI` は単目的でも使えるか

基本的には multi-output / multi-objective 用です。
単目的では `EI`, `NEI`, `UCB`, `PI`, `KG` などを使います。

### 17.6 binary / ordinal で `KG` は使えるか

短縮 alias としての `KG` は regression / hybrid 専用です。
binary / ordinal で `KG` を指定すると、誤って regression 用 BoTorch KG に落ちないようにエラーになります。

---

## 18. 推奨する全体テンプレート

```python
from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    CandidateRepairConfig,
    DataContext,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    OptimizeConfig,
)

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        perturbation=False,
    ),
)

fit_config = FitConfig(
    maxiter=128,
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
    model_registry=MODEL_REGISTRY,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "best_f": train_Y.max(),
    },
)

opt_config = OptimizeConfig(
    optimizer="optimize_acqf",
    q=3,
    num_restarts=10,
    raw_samples=256,
    repair_config=CandidateRepairConfig(
        numeric_indices=numeric_indices,
        steps=steps,
        comp_idx=comp_idx,
        k=k,
    ),
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
)
```
