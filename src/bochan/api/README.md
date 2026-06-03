# bochan API

`bochan.api` は、BoTorch / GPyTorch ベースのモデル構築・学習・獲得関数生成・候補点最適化を、アプリケーションや外部 API から扱いやすい形にまとめるための高レベル API です。

研究・開発段階では関数単位で細かく扱い、アプリ化・API化するときは `BayesianOptimizer` クラスまたは FastAPI ルーターから文字列中心で操作することを想定しています。

---

## 1. 全体設計

内部処理は、基本的に次の4段階に分かれます。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)

acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer` は、この流れを薄く包むクラスです。

```python
from bochan.api import BayesianOptimizer, ModelConfig, FitConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)

bo.fit(train_X, train_Y)
posterior = bo.predict(test_X)
```

以前は `model_registry=MODEL_REGISTRY` を明示的に渡していましたが、現在は通常不要です。省略時は API 標準の `DEFAULT_MODEL_REGISTRY` を内部で参照します。

```python
bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)
```

必要であれば確認・上書きできます。

```python
from bochan.api import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY
```

---

## 2. 設定クラス

| 設定クラス | 役割 |
|---|---|
| `ModelConfig` | モデルの種類、カテゴリ変数、input transform、モデル固有引数を指定する |
| `FitConfig` | 学習回数、learning rate、maxiter などを指定する |
| `InputTransformConfig` | Normalize / input perturbation を簡易指定する |
| `MultiOutputConfig` | multi-output / hybrid の出力ごとのモデル構成を指定する |
| `OutputConfig` | multi-output / hybrid の各出力を詳細指定する |
| `AcquisitionConfig` | 獲得関数を文字列またはクラスで指定する |
| `DataContext` | `X_baseline`, `best_f`, `ref_point`, `partitioning` など獲得関数側の文脈を渡す |
| `MultiObjectiveConfig` | EHVI / NEHVI / NParEGO などの多目的設定を渡す |
| `OptimizeConfig` | 候補点最適化の q, restart 数, optimizer 種類, 制約を指定する |
| `CandidateRepairConfig` | grid rounding, k-sparse, 制約補修用の post-processing を自動生成する |

---

## 3. 最小例: regression + EI

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_Y = torch.tensor([[0.0], [0.25], [1.0]], dtype=torch.double)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)

fit_config = FitConfig(maxiter=128)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
)

opt_config = OptimizeConfig(
    q=1,
    num_restarts=10,
    raw_samples=256,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
print(candidates, acq_value)
```

---

## 4. `ModelConfig`

基本的には `task_type` と `model_type` を文字列で指定します。

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

### 4.1 regression base

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)
```

### 4.2 binary base

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={
        "num_inducing_points": 64,
    },
)
```

### 4.3 ordinal base

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    model_kwargs={
        "num_classes": 3,
    },
)
```

### 4.4 mixed model

`cat_dims` がある場合は、`mixed` 側のモデルとして解決されます。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[2, 5],
)
```

binary mixed の例です。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    cat_dims=[2],
    model_kwargs={
        "num_inducing_points": 64,
    },
)
```

### 4.5 deepkernel

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

binary deepkernel の例です。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="deepkernel",
    model_kwargs={
        "feature_extractor": feature_extractor,
        "feature_dim": 8,
        "num_inducing_points": 64,
    },
)
```

### 4.6 deepgp

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="deepgp",
    model_kwargs={
        "hidden_dims": [16, 8],
        "num_inducing_points": 64,
    },
)
```

### 4.7 SAAS

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="saas",
    model_kwargs={
        "num_taus": 4,
    },
)
```

### 4.8 PCA / REMBO

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

### 4.9 RRP / hetero

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="rrp",
)
```

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="hetero",
)
```

### 4.10 model class を直接渡す高度な使い方

通常は文字列指定を推奨しますが、デバッグ時は直接渡せます。

```python
from botorch.models import SingleTaskGP

model_config = ModelConfig(
    model_cls=SingleTaskGP,
    task_type="regression",
    model_type="base",
)
```

---

## 5. `FitConfig`

通常利用では `fit_func`, `mll_factory`, `mll_cls` を指定する必要はありません。`task_type` / `model_type` / model の `make_mll()` から内部で自動選択します。

### 5.1 exact GP

```python
fit_config = FitConfig(maxiter=128)
```

### 5.2 classification / ordinal / deep 系

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)
```

### 5.3 自動解決の目安

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

```python
fit_config = FitConfig(
    mll_factory=custom_mll_factory,
    fit_func=custom_fit_func,
    fit_kwargs={"num_epochs": 500},
)
```

### 5.5 fit をスキップする

すでに学習済みのモデルや、外部で fit 済みの model を渡す場合は `skip_fit=True` にできます。

```python
fit_config = FitConfig(skip_fit=True)
```

---

## 6. input transform / input perturbation

`InputTransformConfig` を使うと、`build_input_transform(...)` を API 側で自動生成できます。

### 6.1 入力摂動なし

```python
from bochan.api import InputTransformConfig

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
    n_w=N_W,
    std=STD_DEV,
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

### 6.5 入力摂動と objective

入力摂動では `q` が内部で `q * n_w` に展開されるため、獲得関数側で `q * n_w -> q` に戻す objective が必要になる場合があります。

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
    acqf_kwargs={"best_f": train_Y.max()},
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
from botorch.acquisition.monte_carlo import qExpectedImprovement

acq_config = AcquisitionConfig(
    name="custom_ei",
    acqf_cls=qExpectedImprovement,
    acqf_kwargs={"best_f": train_Y.max()},
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

### 7.3 binary / ordinal の例

```python
model_config = ModelConfig(task_type="binary", model_type="base")
AcquisitionConfig(name="BALD")  # -> qBinaryBALD
AcquisitionConfig(name="EI")    # -> qBinaryExpectedImprovement
```

```python
model_config = ModelConfig(task_type="ordinal", model_type="base")
AcquisitionConfig(name="BALD")  # -> qOrdinalBALD
AcquisitionConfig(name="EI")    # -> qOrdinalExpectedImprovement
```

### 7.4 hybrid の例

`task_type="hybrid"` の場合、獲得関数側では regression として解決されます。

```python
AcquisitionConfig(name="EI")   # -> BoTorch qExpectedImprovement
AcquisitionConfig(name="EHI")  # -> BoTorch qExpectedHypervolumeImprovement
AcquisitionConfig(name="KG")   # -> BoTorch qKnowledgeGradient
```

`KG` / `MultiStepLookahead` は regression / hybrid 用です。binary / ordinal で短縮指定した場合は、誤解決を避けるためエラーになります。

### 7.5 登録済み獲得関数名を確認する

```python
from bochan.api import available_acqf_names

names = available_acqf_names()
```

---

## 8. 単目的実行例

### 8.1 regression + UCB

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)

fit_config = FitConfig(maxiter=128)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)
bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="UCB",
    acqf_kwargs={"beta": 0.2},
)

opt_config = OptimizeConfig(q=3, num_restarts=10, raw_samples=256)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

### 8.2 regression + KG

```python
acq_config = AcquisitionConfig(
    name="KG",
    acqf_kwargs={
        "num_fantasies": 64,
    },
)

opt_config = OptimizeConfig(
    q=1,
    num_restarts=10,
    raw_samples=256,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

### 8.3 binary + BALD

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={"num_inducing_points": 64},
)

fit_config = FitConfig(num_epochs=300, lr=0.01)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)
bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(name="BALD")
opt_config = OptimizeConfig(q=3, num_restarts=10, raw_samples=256)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

### 8.4 binary + EI

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": best_f},
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

binary model では `qBinaryExpectedImprovement` に解決されます。

### 8.5 ordinal + MarginUncertainty

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    model_kwargs={"num_classes": 3},
)

fit_config = FitConfig(num_epochs=500, lr=0.03)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)
bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(name="Margin")
opt_config = OptimizeConfig(q=3, num_restarts=10, raw_samples=256)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

---

## 9. multi-output / multi-objective

`MultiOutputConfig()` は複数出力モデルを作る合図です。多目的最適化の合図ではありません。

```text
MultiOutputConfig()
    = Y が複数列あるので、複数出力モデルを作る

MultiObjectiveConfig()
    = その複数出力を目的関数群として最適化する

AcquisitionConfig(name="EHI")
    = 多目的獲得関数を使う
```

### 9.1 homogeneous multi-output regression

`train_Y` が `n x m` の場合、`MultiOutputConfig()` を指定すると、`train_Y.shape[-1]` の数だけ親の `task_type` / `model_type` を複製します。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)
```

これは `train_Y.shape[-1] == 3` の場合、概念的には以下と同じです。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=["regression", "regression", "regression"],
    ),
)
```

### 9.2 multi-output binary

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)
```

これは `train_Y.shape[-1] == 3` の場合、概念的には以下と同じです。

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=["binary", "binary", "binary"],
    ),
)
```

### 9.3 出力ごとに詳細指定

```python
model_config = ModelConfig(
    task_type="binary",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {
                "name": "defect_a",
                "task_type": "binary",
                "model_type": "base",
                "model_kwargs": {"num_inducing_points": 64},
            },
            {
                "name": "defect_b",
                "task_type": "binary",
                "model_type": "deepkernel",
                "model_kwargs": {"feature_extractor": feature_extractor},
            },
        ],
    ),
)
```

### 9.4 hybrid model

regression / binary / ordinal など異なる task type を同時に扱う場合は hybrid を使います。

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {"name": "strength", "task_type": "regression", "model_type": "base"},
            {"name": "defect", "task_type": "binary", "model_type": "base"},
            {"name": "rank", "task_type": "ordinal", "model_type": "base"},
        ],
        use_hybrid=True,
    ),
)
```

### 9.5 hybrid の出力ごとに詳細指定

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            {
                "name": "strength",
                "task_type": "regression",
                "model_type": "base",
                "output_spec_kwargs": {"sign": 1.0},
            },
            {
                "name": "defect",
                "task_type": "binary",
                "model_type": "base",
                "model_kwargs": {"num_inducing_points": 64},
                "output_spec_kwargs": {
                    "sign": -1.0,
                    "positive_class": 1,
                },
            },
            {
                "name": "rank",
                "task_type": "ordinal",
                "model_type": "base",
                "model_kwargs": {"num_classes": 3},
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

### 9.6 出力ごとに fit 設定を変える

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
                "model_kwargs": {"num_classes": 3},
                "fit_config": FitConfig(num_epochs=800, lr=0.03),
            },
        ],
        use_hybrid=True,
    ),
)
```

### 9.7 qEHVI

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

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
opt_config = OptimizeConfig(q=3, num_restarts=10, raw_samples=256)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
)
```

### 9.8 qNEHVI

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

### 9.9 NParEGO

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

## 10. optimizer / constraints / repair

### 10.1 optimizer 選択

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

### 10.2 BoTorch 標準

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf",
    q=3,
    num_restarts=10,
    raw_samples=256,
)
```

### 10.3 BoTorch mixed

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=3,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=[
        {2: 0},
        {2: 1},
    ],
)
```

### 10.4 torch optimizer

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

### 10.5 evo optimizer

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

### 10.6 evo mixed

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

### 10.7 torch mixed

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

### 10.8 fixed_features と fixed_features_list

`fixed_features` は常に固定したい変数、`fixed_features_list` はカテゴリ・離散値の列挙パターンです。

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    fixed_features={0: 0.5},
    fixed_features_list=[
        {2: 0},
        {2: 1},
    ],
)
```

内部では以下のようにマージされます。

```python
[
    {0: 0.5, 2: 0},
    {0: 0.5, 2: 1},
]
```

同じ次元が重複した場合は `fixed_features_list` 側を優先します。

### 10.9 BoTorch 線形制約

```python
opt_config = OptimizeConfig(
    q=3,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
)
```

### 10.10 grid rounding + k-sparse + constraints repair

`CandidateRepairConfig` を使うと、`make_grid_k_sparse_post_processing_func(...)` を API 側で自動生成できます。

```python
opt_config = OptimizeConfig(
    q=3,
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

### 10.11 丸めだけ

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

### 10.12 k-sparse だけ

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

---

## 11. `DataContext`

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

---

## 12. ask / tell

```python
candidates, acq_value = bo.ask(acq_config, opt_config)

new_Y = evaluate(candidates)

bo.tell(candidates, new_Y, refit=True)
```

fit 設定を変えて再学習したい場合は次です。

```python
bo.tell(
    candidates,
    new_Y,
    refit=True,
    fit_config=FitConfig(maxiter=128),
)
```

---

## 13. 複数獲得関数の比較

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

## 14. 推奨する Python API 全体テンプレート

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

fit_config = FitConfig(maxiter=128)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=fit_config,
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
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

---

## 15. FastAPI で使う

FastAPI 統合は optional dependency です。通常の `bochan.api` import では FastAPI を要求しません。

### 15.1 インストール

```bash
pip install -e '.[api]'
```

または、パッケージとしてインストールする場合は次です。

```bash
pip install 'botorch_ext[api]'
```

### 15.2 通常版の起動

```bash
uvicorn bochan.api.fastapi:app --reload
```

### 15.3 保存・復元対応版の起動

```bash
uvicorn bochan.api.fastapi_persistent:app --reload
```

保存先は環境変数 `BOCHAN_API_MODEL_DIR` で指定できます。

```bash
export BOCHAN_API_MODEL_DIR=./saved_bo_models
```

未指定の場合は `bochan_sessions/` に保存されます。

### 15.4 OpenAPI UI

```text
http://127.0.0.1:8000/docs
```

---

## 16. FastAPI エンドポイント

すべて `/bochan` prefix 配下です。

### 16.1 通常版 / persistent 版 共通

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/bochan/health` | ヘルスチェック |
| `GET` | `/bochan/acquisitions` | 登録済み acquisition 名の一覧 |
| `GET` | `/bochan/sessions` | セッション ID 一覧 |
| `POST` | `/bochan/sessions` | 学習済み `BayesianOptimizer` セッションを作成 |
| `DELETE` | `/bochan/sessions/{session_id}` | セッション削除 |
| `POST` | `/bochan/sessions/{session_id}/predict` | セッションで予測 |
| `POST` | `/bochan/sessions/{session_id}/candidate` | セッションで候補点生成 |
| `POST` | `/bochan/sessions/{session_id}/ask` | `/candidate` の alias |
| `POST` | `/bochan/sessions/{session_id}/tell` | 観測追加と任意の再学習 |
| `POST` | `/bochan/suggest` | stateless に fit → candidate まで実行 |

### 16.2 persistent 版のみ

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/bochan/models` | 保存済みモデルファイル一覧 |
| `POST` | `/bochan/sessions/{session_id}/save` | メモリ上のセッションをファイル保存 |
| `POST` | `/bochan/sessions/load` | 保存済みセッションをロードして新しい `session_id` を発行 |

---

## 17. FastAPI: stateful session 例

### 17.1 セッション作成

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]],
    "model_config": {
      "task_type": "regression",
      "model_type": "base"
    },
    "fit_config": {
      "maxiter": 64
    }
  }'
```

レスポンス例:

```json
{
  "session_id": "...",
  "task_type": "regression",
  "model_type": "base",
  "input_type": "normal",
  "metadata": {
    "model_cls": "SingleTaskGP"
  }
}
```

### 17.2 候補点生成

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/candidate \
  -H 'Content-Type: application/json' \
  -d '{
    "acquisition_config": {
      "name": "EI",
      "acqf_kwargs": {"best_f": 1.0}
    },
    "optimize_config": {
      "q": 1,
      "num_restarts": 5,
      "raw_samples": 64
    }
  }'
```

### 17.3 予測

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "X": [[0.25], [0.75]],
    "return_type": "mean_variance"
  }'
```

### 17.4 ask / tell

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "acquisition_config": {"name": "UCB", "acqf_kwargs": {"beta": 0.2}},
    "optimize_config": {"q": 1, "num_restarts": 5, "raw_samples": 64}
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/tell \
  -H 'Content-Type: application/json' \
  -d '{
    "new_X": [[0.3]],
    "new_Y": [[0.09]],
    "refit": true,
    "fit_config": {"maxiter": 64}
  }'
```

---

## 18. FastAPI: stateless suggest 例

`/bochan/suggest` は、1リクエスト内で `fit -> acquisition -> optimize` まで実行します。セッションを残したくない単発 API に向いています。

```bash
curl -X POST http://127.0.0.1:8000/bochan/suggest \
  -H 'Content-Type: application/json' \
  -d '{
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]],
    "model_config": {
      "task_type": "regression",
      "model_type": "base"
    },
    "fit_config": {"maxiter": 64},
    "acquisition_config": {
      "name": "EI",
      "acqf_kwargs": {"best_f": 1.0}
    },
    "optimize_config": {
      "q": 1,
      "num_restarts": 5,
      "raw_samples": 64
    }
  }'
```

---

## 19. FastAPI: multi-output / hybrid / 多目的

### 19.1 multi-output

```json
{
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
  "bounds": [[0.0], [1.0]],
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "multi_output_config": {}
  },
  "fit_config": {
    "maxiter": 64
  }
}
```

### 19.2 hybrid

```json
{
  "model_config": {
    "task_type": "hybrid",
    "multi_output_config": {
      "use_hybrid": true,
      "output_configs": [
        {"name": "strength", "task_type": "regression", "model_type": "base"},
        {"name": "defect", "task_type": "binary", "model_type": "base"},
        {"name": "rank", "task_type": "ordinal", "model_type": "base"}
      ]
    }
  }
}
```

### 19.3 多目的 EHVI

```json
{
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
  "bounds": [[0.0], [1.0]],
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "multi_output_config": {}
  },
  "fit_config": {"maxiter": 64},
  "acquisition_config": {"name": "EHI"},
  "data_context": {
    "multi_objective": {
      "ref_point": [-0.1, -0.1],
      "Y_baseline": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
      "auto_partitioning": true
    }
  },
  "optimize_config": {
    "q": 1,
    "num_restarts": 5,
    "raw_samples": 64
  }
}
```

---

## 20. FastAPI: 制約・fixed features・repair

JSON では線形制約を次の形式で渡せます。

```json
{
  "indices": [0, 1],
  "coefficients": [1.0, 1.0],
  "rhs": 1.0
}
```

`fixed_features` と `fixed_features_list` は JSON object として渡します。key は JSON 上は文字列でも、API 内で `int` に変換されます。

```json
{
  "optimize_config": {
    "optimizer": "optimize_acqf_mixed",
    "q": 1,
    "fixed_features": {"0": 0.5},
    "fixed_features_list": [
      {"2": 0},
      {"2": 1}
    ],
    "inequality_constraints": [
      {"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 1.0}
    ],
    "repair_config": {
      "numeric_indices": [0, 1],
      "steps": [0.1, 0.1],
      "comp_idx": [0, 1],
      "k": 1,
      "final_priority": "grid"
    }
  }
}
```

---

## 21. FastAPI: optimizer 選択

```json
{
  "optimize_config": {
    "optimizer": "torch",
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 512,
    "optimizer_kwargs": {
      "method": "adam",
      "options": {
        "lr": 0.03,
        "num_steps": 200,
        "penalty_factor": 1000.0
      }
    }
  }
}
```

---

## 22. FastAPI: 保存・復元

保存・復元は `bochan.api.fastapi_persistent:app` で使います。

### 22.1 保存済みファイル一覧

```bash
curl http://127.0.0.1:8000/bochan/models
```

### 22.2 セッション保存

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/save \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "demo_model.pt",
    "overwrite": true
  }'
```

### 22.3 セッション復元

`torch.load` は pickle を使うため、復元時は明示的に `trust_pickle=true` が必要です。自分で保存したファイルなど、信頼できるファイルだけ読み込んでください。

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/load \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "demo_model.pt",
    "map_location": "cpu",
    "trust_pickle": true
  }'
```

ロードすると新しい `session_id` が発行されます。その `session_id` は `/predict`, `/candidate`, `/ask`, `/tell` にそのまま使えます。

---

## 23. よくある注意点

### 23.1 `model_type="base"` と `model_cls` は両方必要か

通常は両方は不要です。

```python
ModelConfig(
    task_type="regression",
    model_type="base",
)
```

この場合は標準 registry から `SingleTaskGP` などを解決します。

`model_cls` は registry を使わず直接指定したい場合だけ使います。

```python
ModelConfig(
    task_type="regression",
    model_cls=SingleTaskGP,
)
```

### 23.2 `FitConfig` に `fit_func` は必要か

通常は不要です。

```python
FitConfig(maxiter=128)
```

または、

```python
FitConfig(num_epochs=300, lr=0.01)
```

だけで十分です。

### 23.3 `input_transform` と `input_transform_config` の優先順位

`input_transform` を直接渡した場合は、それが優先されます。`input_transform_config` は無視されます。

```python
ModelConfig(
    input_transform=manual_transform,
    input_transform_config=InputTransformConfig(perturbation=True),
)
```

### 23.4 `EHI` / `EHVI` / `NEHVI` は単目的でも使えるか

基本的には multi-output / multi-objective 用です。単目的では `EI`, `NEI`, `UCB`, `PI`, `KG` などを使います。

### 23.5 binary / ordinal で `KG` は使えるか

短縮 alias としての `KG` は regression / hybrid 専用です。binary / ordinal で `KG` を指定すると、誤って regression 用 BoTorch KG に落ちないようにエラーになります。

### 23.6 FastAPI の stateful session は本番向けか

メモリ上に `BayesianOptimizer` を保持するため、ローカルアプリやプロトタイプ向けです。本番運用では、プロセス再起動でセッションが消える点、ワーカーを複数立てるとメモリストアが共有されない点に注意してください。

### 23.7 保存済みモデルをロードするときの注意

保存・復元対応版では `torch.save` / `torch.load` を使います。pickle を含むため、信頼できないファイルはロードしないでください。

---

## 24. 実装上の注意

- `bochan.api.fastapi` / `bochan.api.fastapi_persistent` は optional module です。FastAPI を使わない通常の Python API には影響しません。
- GPU を使う場合は `tensor_options.device` に `"cuda"` などを指定できます。
- JSON では Python の callable や objective object は直接渡せません。API 経由では、文字列指定・数値・配列・dict で表現できる範囲を基本にしてください。
- 保存・復元対応版では `BOCHAN_API_MODEL_DIR` で保存先を変更できます。
