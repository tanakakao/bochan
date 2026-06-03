# bochan API

`bochan.api` は、BoTorch / GPyTorch ベースのモデル構築・学習・獲得関数生成・候補点最適化を、アプリケーションや外部 API から扱いやすい形にまとめるための高レベル API です。

研究・開発段階では関数単位で細かく扱い、アプリ化・API化するときは `BayesianOptimizer` クラスまたは FastAPI ルーターから文字列中心で操作することを想定しています。

---

## 1. 全体設計

内部処理は、基本的に次の4段階です。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)

acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer` は、この流れを薄く包むクラスです。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

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
| `ModelConfig` | モデルの種類、カテゴリ変数、input transform、outcome transform、モデル固有引数を指定する |
| `FitConfig` | 学習回数、learning rate、maxiter、MLL / fit 関数の上書きなどを指定する |
| `InputTransformConfig` | Normalize / input perturbation を簡易指定する |
| `MultiOutputConfig` | multi-output / hybrid の出力ごとのモデル構成を指定する |
| `OutputConfig` | multi-output / hybrid の各出力を詳細指定する |
| `AcquisitionConfig` | 獲得関数、objective、sampler、獲得関数固有引数を指定する |
| `ObjectiveConfig` | regression / binary / ordinal / hybrid に応じた objective を API 側で自動生成する |
| `DataContext` | `X_baseline`, `best_f`, `ref_point`, `partitioning` など獲得関数側の文脈を渡す |
| `MultiObjectiveConfig` | EHVI / NEHVI / NParEGO などの多目的設定を渡す |
| `OptimizeConfig` | 候補点最適化の q, restart 数, optimizer 種類, 制約を指定する |
| `CandidateRepairConfig` | grid rounding, k-sparse, 制約補修用の post-processing を自動生成する |

`ObjectiveConfig` は現在の API で重要な設定です。ユーザー側で `RegressionScalarObjective` や `BinaryClassificationScoreObjective` を直接選ばなくても、API 側が `task_type` と model 情報から適切な objective を生成します。

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
"multi_objective"
"binary"
"multiclass"
"ordinal"
"hybrid"
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

### 4.4 outcome transform

`outcome_transform` は regression 系モデルに Standardize を適用するかどうかを指定します。

現在の実装では、`outcome_transform=True` の場合、`AutoStandardizeOutcomeTransform` を使って `train_Y` の出力次元から `Standardize(m=...)` を遅延生成します。

```python
# regression: Standardize あり。デフォルト。
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    outcome_transform=True,
)

# regression: Standardize なし。
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    outcome_transform=False,
)
```

`binary` / `multiclass` / `ordinal` では outcome transform は自動的に無効化されます。

`hybrid` の場合、wrapper 自体には outcome transform を渡しません。親の `outcome_transform` は submodel の `ModelConfig` に継承され、submodel が `regression` / `multi_objective` に解決されたときだけ適用されます。

```python
model_config = ModelConfig(
    task_type="hybrid",
    outcome_transform=True,
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),  # Standardize あり
            OutputConfig(task_type="binary", name="defect"),        # outcome_transform 無効
            OutputConfig(task_type="ordinal", name="rank"),         # outcome_transform 無効
        ],
        use_hybrid=True,
    ),
)
```

hybrid の regression 出力も標準化したくない場合は、親で `outcome_transform=False` を指定します。

```python
model_config = ModelConfig(
    task_type="hybrid",
    outcome_transform=False,
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),
            OutputConfig(task_type="binary", name="defect"),
        ],
        use_hybrid=True,
    ),
)
```

### 4.5 mixed model

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

### 4.6 deepkernel

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

### 4.7 deepgp

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

### 4.8 SAAS

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="saas",
    model_kwargs={
        "num_taus": 4,
    },
)
```

### 4.9 PCA / REMBO

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

### 4.10 RRP / hetero

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

### 4.11 model class を直接渡す高度な使い方

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

### 6.5 input perturbation と objective

入力摂動では `q` が内部で `q * n_w` に展開されるため、獲得関数側で `q * n_w -> q` に戻す objective が必要になる場合があります。

従来は objective class を直接指定していました。

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

現在の API では、通常は `ObjectiveConfig` を使います。

```python
from bochan.api import AcquisitionConfig, ObjectiveConfig

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        direction="maximize",
        n_w=8,
        risk_type=None,
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

---

## 7. multi-output / hybrid model

`MultiOutputConfig` を使うと、出力ごとに submodel を作り、それらを wrapper に束ねます。

### 7.1 homogeneous multi-output regression

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)
```

`train_Y.shape[-1]` から出力数を推定し、各出力に同じ `ModelConfig` を複製します。

### 7.2 output ごとに task を変える hybrid

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(task_type="binary", model_type="base", name="defect"),
            OutputConfig(task_type="ordinal", model_type="base", name="rank"),
        ],
        use_hybrid=True,
    ),
)
```

`OutputConfig.name` は hybrid objective で `output="strength"` のように参照できます。

### 7.3 output ごとの fit 設定

```python
model_config = ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),
            OutputConfig(task_type="binary", name="defect"),
        ],
        output_fit_configs=[
            FitConfig(maxiter=128),
            FitConfig(num_epochs=300, lr=0.01),
        ],
        use_hybrid=True,
    ),
)
```

---

## 8. acquisition registry

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

### 8.1 BoTorch 系 alias

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

### 8.2 task context に応じる短縮 alias

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

### 8.3 binary / ordinal の例

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

### 8.4 hybrid の例

`task_type="hybrid"` の場合、獲得関数側では regression 系として解決されます。

```python
AcquisitionConfig(name="EI")   # -> BoTorch qExpectedImprovement
AcquisitionConfig(name="EHI")  # -> BoTorch qExpectedHypervolumeImprovement
AcquisitionConfig(name="KG")   # -> BoTorch qKnowledgeGradient
```

`KG` / `MultiStepLookahead` は regression / hybrid 用です。binary / ordinal で短縮指定した場合は、誤解決を避けるためエラーになります。

---

## 9. `ObjectiveConfig`

`ObjectiveConfig` は、API 側で objective を自動構築するための設定です。

通常の API 利用では、ユーザーは `RegressionScalarObjective` や `BinaryClassificationScoreObjective` などを直接選びません。`ObjectiveConfig` に `mode`, `output`, `outputs`, `direction`, `n_w`, `risk_type` などを指定すると、`factory.py` の `build_objective(...)` が `bundle.task_type` に応じて適切な実 objective を生成します。

### 9.1 objective の優先順位

`build_objective(...)` の優先順位は次です。

```python
if acq_config.objective is not None:
    # 生成済み objective をそのまま使う
    objective = acq_config.objective
elif acq_config.objective_factory is not None:
    # 上級者向け factory で生成する
    objective = acq_config.objective_factory(...)
elif acq_config.objective_config is not None:
    # ObjectiveConfig から API が自動生成する
    objective = build_objective_from_config(...)
else:
    objective = None
```

`acqf_factory` を指定した場合は、`build_acquisition(...)` がその factory に処理を委譲します。そのため、標準の objective 挿入も factory 側で必要に応じて処理してください。

### 9.2 `ObjectiveConfig` の主要引数

| 引数 | 意味 |
|---|---|
| `mode` | `"auto"`, `"none"`, `"scalar"`, `"multi_output"` |
| `output` | scalar objective で使う出力 index または出力名 |
| `outputs` | multi-output objective で使う出力 index / 出力名の列 |
| `specs` | hybrid objective 用の `HybridObjectiveSpec` の列。指定時は `outputs` などより優先 |
| `direction` | scalar objective の方向。`"maximize"` / `"minimize"` |
| `directions` | multi-output objective の各出力方向 |
| `weight` | scalar objective の重み |
| `weights` | multi-output objective の各出力重み |
| `eq_target` | scalar objective で目標値に近いほど良い score にする場合の目標値 |
| `eq_targets` | multi-output objective の各出力目標値 |
| `n_w` | input perturbation で1点あたりに展開される摂動数 |
| `risk_type` | `None`, `"var"`, `"cvar"` |
| `alpha` | VaR / CVaR の risk 集約パラメータ |
| `maximize` | risk 集約時に大きい値を良い方向として扱うか |
| `utility_values` | ordinal class の utility 値 |
| `ordinal_likelihood` | ordinal likelihood の明示指定 |

`mode="auto"` の場合、`outputs` または `specs` が指定されていれば `multi_output`、それ以外は `scalar` として扱われます。

### 9.3 task type ごとの自動生成

| `bundle.task_type` | `mode` | 生成される objective |
|---|---|---|
| `regression` / `multi_objective` | `scalar` | `RegressionScalarObjective` |
| `regression` / `multi_objective` | `multi_output` | `make_hybrid_multi_output_objective(...)` |
| `binary` | `scalar` | `BinaryClassificationScoreObjective` |
| `binary` | `multi_output` | `MultiOutputBinaryClassificationInputPerturbationObjective` |
| `ordinal` | `scalar` | `OrdinalInputPerturbationExpectedUtilityObjective` |
| `ordinal` | `multi_output` | `MultiOutputOrdinalInputPerturbationObjective` |
| `hybrid` | `scalar` | `make_hybrid_scalar_objective(...)` |
| `hybrid` | `multi_output` | `make_hybrid_multi_output_objective(...)` |

`multiclass` の objective 自動生成は現時点では未実装です。必要な場合は `AcquisitionConfig.objective` または `objective_factory` で明示的に渡してください。

---

## 10. regression objective の使い方

### 10.1 single-output regression

single-output regression で、摂動なし・目的値をそのまま最大化する場合は objective なしで構いません。

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
)
```

input perturbation や risk 集約を使う場合は、`ObjectiveConfig` を指定します。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        direction="maximize",
        n_w=8,
        risk_type=None,
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

CVaR を使う場合です。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        direction="maximize",
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

### 10.2 最小化

BoTorch の多くの獲得関数は最大化前提なので、最小化したい場合は `direction="minimize"` を指定します。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        direction="minimize",
    ),
    acqf_kwargs={"best_f": (-train_Y[:, 0]).max()},
)
```

`best_f` も objective の向きに合わせる必要があります。

### 10.3 目標値に近いほど良い目的

`eq_target` を指定すると、`-abs(y - eq_target)` 型の score に変換します。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        eq_target=10.0,
        n_w=8,
        risk_type=None,
    ),
    acqf_kwargs={"best_f": best_score},
)
```

### 10.4 regression multi-output から特定の変数だけ使う

`train_Y.shape = (n, m)` の regression で、特定の目的変数だけを scalar acquisition に使う場合は `output` を指定します。

```python
# y1 だけを EI / UCB などの scalar acquisition に使う
best_f_y1 = train_Y[:, 1].max()

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=1,
        direction="maximize",
        n_w=8,
        risk_type=None,
    ),
    acqf_kwargs={"best_f": best_f_y1},
)
```

通常の non-hybrid multi-output regression では、`output` は整数 index で指定してください。文字列名は `model.output_names` を持つ hybrid model などで使えます。

### 10.5 regression multi-output から一部の出力だけ多目的に使う

`y0` と `y2` だけを EHVI / NEHVI などに使い、`y1` を無視したい場合です。

```python
Y_baseline = train_Y[:, [0, 2]]
ref_point = torch.tensor([0.0, 0.0], dtype=train_Y.dtype, device=train_Y.device)

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=[0, 2],
        directions=["maximize", "minimize"],
        weights=[1.0, 1.0],
        n_w=8,
        risk_type=None,
    ),
)

data_context = DataContext(
    X_baseline=train_X,
    Y_baseline=Y_baseline,
    ref_point=ref_point,
)
```

`outputs` を指定した場合、`Y_baseline`, `ref_point`, `objective_thresholds` も選択後の出力次元に合わせるのが安全です。

---

## 11. binary / ordinal objective

### 11.1 binary classification

binary classification では、獲得関数内部で確率や score が計算済みの場合、objective は主に input perturbation によって展開された `q * n_w` を `q` に戻すために使います。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        n_w=8,
        risk_type=None,
    ),
)
```

API 側では `BinaryClassificationScoreObjective` が選択されます。

multi-output binary classification では次のようにします。

```python
acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
)
```

API 側では `MultiOutputBinaryClassificationInputPerturbationObjective` が選択されます。

### 11.2 ordinal

ordinal では latent `f` を class probability に変換し、`utility_values` で expected utility に変換する必要があります。BO 系では objective を指定するのが基本です。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        utility_values=[0.0, 1.0, 2.0],
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
)
```

API 側では `OrdinalInputPerturbationExpectedUtilityObjective` が選択され、`ordinal_likelihood` は model から推定されます。

`utility_values` を省略した場合、API は `num_classes` や cutpoints / thresholds から `0..K-1` を推定しようとします。推定できない場合は `utility_values` を明示してください。

multi-output ordinal では次のようにします。

```python
acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        utility_values=[0.0, 1.0, 2.0],
        n_w=8,
        risk_type=None,
    ),
)
```

API 側では `MultiOutputOrdinalInputPerturbationObjective` が選択されます。

---

## 12. hybrid objective

hybrid では出力名または index を使って objective を指定します。内部では既存の `HybridObjectiveSpec`, `make_hybrid_scalar_objective`, `make_hybrid_multi_output_objective` が使われます。

### 12.1 single-output hybrid

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output="strength",
        direction="maximize",
        n_w=8,
        risk_type=None,
    ),
    acqf_kwargs={"best_f": best_strength},
)
```

`output` は `OutputConfig(name="...")` で指定した名前、または整数 index を使えます。

### 12.2 multi-output hybrid

```python
acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=["strength", "defect_prob"],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
)
```

`outputs`, `directions`, `weights`, `eq_targets` は同じ長さにしてください。

### 12.3 `HybridObjectiveSpec` を直接使う

より明示したい場合は `HybridObjectiveSpec` を渡せます。

```python
from bochan.acquisition.objective import HybridObjectiveSpec

specs = [
    HybridObjectiveSpec(output="strength", direction="maximize", weight=1.0),
    HybridObjectiveSpec(output="defect_prob", direction="minimize", weight=0.5),
]

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        specs=specs,
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
)
```

---

## 13. `DataContext` / `MultiObjectiveConfig`

`DataContext` は、獲得関数生成時に必要な文脈情報を渡すための設定です。

```python
data_context = DataContext(
    bounds=bounds,
    X_baseline=train_X,
    Y_baseline=train_Y,
    best_f=train_Y.max(),
    ref_point=ref_point,
)
```

主なフィールドは次です。

| フィールド | 用途 |
|---|---|
| `bounds` | 最適化範囲。`optimize_candidates` では別途 `bounds` を渡すことが多い |
| `X_baseline` | NEI / NEHVI / NIPV などの baseline |
| `X_pending` | 既に評価予定の候補点 |
| `Y_baseline` | EHVI / NEHVI / scalarization の基準出力 |
| `best_f` | EI / PI などの既知ベスト値 |
| `ref_point` | EHVI / NEHVI の参照点 |
| `partitioning` | EHVI 用 partitioning |
| `objective_thresholds` | 多目的の threshold |
| `mc_points` | integrated posterior variance 系の積分点 |
| `constraints` | BoTorch acquisition に渡す制約 |
| `extra` | その他の acquisition 固有引数 |

`MultiObjectiveConfig` を使うと、EHVI / NEHVI / NParEGO 用の文脈をまとめられます。

```python
data_context = DataContext(
    X_baseline=train_X,
    multi_objective=MultiObjectiveConfig(
        ref_point=ref_point,
        Y_baseline=train_Y,
        auto_partitioning=True,
    ),
)
```

`MultiObjectiveConfig.auto_scalarization` は、`AcquisitionConfig.objective`, `objective_factory`, `objective_config` がいずれも指定されていない場合だけ自動的に scalarization objective を作ります。明示した objective 設定を上書きしません。

---

## 14. `OptimizeConfig`

候補点最適化は `OptimizeConfig` で制御します。

```python
opt_config = OptimizeConfig(
    q=3,
    num_restarts=20,
    raw_samples=512,
    sequential=True,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

主な引数は次です。

| 引数 | 意味 |
|---|---|
| `q` | 一度に提案する候補点数 |
| `num_restarts` | 多点初期化の restart 数 |
| `raw_samples` | 初期候補生成用の raw sample 数 |
| `sequential` | q-batch を逐次的に最適化するか |
| `optimizer` | `optimize_acqf`, `optimize_acqf_mixed`, `evo`, `torch` など |
| `optimizer_kwargs` | optimizer に渡す追加引数 |
| `post_processing_func` | 候補点の後処理関数 |
| `repair_config` | grid rounding / k-sparse / 制約補修を自動生成する設定 |
| `fixed_features` | 固定する特徴量 |
| `fixed_features_list` | mixed optimizer 用の固定特徴量候補リスト |
| `inequality_constraints` | 線形不等式制約 |
| `equality_constraints` | 線形等式制約 |

### 14.1 optimizer の種類

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="optimize_acqf_mixed")
OptimizeConfig(optimizer="evo")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="evo_mixed")
OptimizeConfig(optimizer="torch_mixed")
```

callable を直接渡すこともできます。

```python
opt_config = OptimizeConfig(
    optimizer=custom_optimizer,
    optimizer_kwargs={"maxiter": 200},
)
```

### 14.2 mixed optimization

`optimize_acqf_mixed` を使う場合は、`fixed_features_list` または `fixed_features` が必要です。

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

`fixed_features` と `fixed_features_list` を両方指定した場合、`fixed_features` は全候補に共通の固定値としてマージされ、各 `fixed_features_list` の値が優先されます。

---

## 15. `CandidateRepairConfig`

`repair_config` を指定すると、grid rounding, k-sparse, 線形制約補修などを post-processing として組み込めます。

```python
opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        numeric_indices=[0, 1, 2],
        steps=[0.1, 0.1, 1.0],
        k=2,
    ),
)
```

### 15.1 丸めのみ

`comp_idx=None` または `comp_idx=[]` の場合は、k-sparse 補修を使わず、数値列の丸めだけに使えます。

```python
opt_config = OptimizeConfig(
    repair_config=CandidateRepairConfig(
        numeric_indices=[0, 1],
        steps=[0.1, 0.01],
        comp_idx=None,
        k=0,
    ),
)
```

### 15.2 k-sparse + 制約補修

```python
opt_config = OptimizeConfig(
    repair_config=CandidateRepairConfig(
        bounds=bounds,
        numeric_indices=[0, 1, 2, 3],
        steps=[0.1, 0.1, 0.1, 0.1],
        comp_idx=[0, 1, 2, 3],
        k=2,
        inequality_constraints=ineq_constraints,
        inequality_sense="le",
        final_priority="constraints",
    ),
)
```

`steps=None` の場合は grid rounding を行いません。

---

## 16. 単目的実行例

### 16.1 regression + UCB

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(maxiter=128),
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

### 16.2 regression + input perturbation + CVaR EI

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        perturbation=True,
        n_w=8,
        std=0.1,
    ),
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

### 16.3 binary + BALD

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={"num_inducing_points": 64},
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(name="BALD")
candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

### 16.4 ordinal + expected utility EI

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    model_kwargs={"num_classes": 3},
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        utility_values=[0.0, 1.0, 2.0],
        n_w=None,
    ),
    acqf_kwargs={"best_f": best_utility},
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

---

## 17. 多目的実行例

### 17.1 regression multi-output + NEHVI

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

Y_baseline = train_Y[:, [0, 2]]
ref_point = torch.tensor([0.0, 0.0], dtype=train_Y.dtype, device=train_Y.device)

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=[0, 2],
        directions=["maximize", "minimize"],
        weights=[1.0, 1.0],
    ),
)

data_context = DataContext(
    X_baseline=train_X,
    Y_baseline=Y_baseline,
    ref_point=ref_point,
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=OptimizeConfig(q=3),
    data_context=data_context,
)
```

### 17.2 hybrid + NEHVI

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),
            OutputConfig(task_type="binary", name="defect_prob"),
        ],
        use_hybrid=True,
    ),
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="NEHVI",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=["strength", "defect_prob"],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
        n_w=8,
        risk_type=None,
    ),
)
```

---

## 18. 注意点

### 18.1 `best_f` は objective 後の値に合わせる

`direction="minimize"`, `output=1`, `eq_target=...` などを使う場合、`best_f` は変換後の objective value に合わせてください。

```python
# y1 を最大化する場合
best_f = train_Y[:, 1].max()

# y1 を最小化する場合
best_f = (-train_Y[:, 1]).max()
```

### 18.2 `ref_point` / `Y_baseline` は選択出力に合わせる

`outputs=[0, 2]` のように一部出力だけを使う場合、`ref_point` と `Y_baseline` も同じ出力次元に合わせます。

```python
Y_baseline = train_Y[:, [0, 2]]
ref_point = torch.tensor([r0, r2], dtype=train_Y.dtype, device=train_Y.device)
```

### 18.3 input perturbation の `n_w` を揃える

`InputTransformConfig(n_w=8)` を使う場合、`ObjectiveConfig(n_w=8)` も同じ値にしてください。

```python
model_config = ModelConfig(
    task_type="regression",
    input_transform_config=InputTransformConfig(perturbation=True, n_w=8),
)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(n_w=8),
)
```

### 18.4 `ObjectiveConfig` が不要な場合もある

single-output regression で、摂動なし、目的値をそのまま最大化するだけなら objective は不要です。

active learning 系の獲得関数でも、獲得関数本体がすでに `batch_shape x q` の score を返す場合は objective が不要なことがあります。

### 18.5 上級者向け override

必要であれば、従来通り objective を直接渡せます。

```python
from bochan.acquisition.objective import RegressionScalarObjective

acq_config = AcquisitionConfig(
    name="EI",
    objective=RegressionScalarObjective(n_w=8, risk_type=None),
)
```

factory で上書きすることもできます。

```python
acq_config = AcquisitionConfig(
    name="EI",
    objective_factory=custom_objective_factory,
    objective_kwargs={"foo": 1},
)
```

この場合、API は `custom_objective_factory(model=..., bundle=..., data_context=..., **objective_kwargs)` を呼び、シグネチャに存在する引数だけを渡します。
