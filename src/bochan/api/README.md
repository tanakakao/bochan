# bochan API

`bochan.api` は、BoTorch / GPyTorch ベースのモデル構築・学習・獲得関数生成・候補点最適化を、アプリケーションや外部 API から扱いやすい形にまとめるための高レベル API です。

研究・開発段階では関数単位で細かく扱い、アプリ化・API 化するときは `BayesianOptimizer`、`BochanStudy`、または FastAPI ルーターから文字列中心で操作することを想定しています。

現在の API は、次を主な対象にします。

- regression / multi-objective regression
- binary classification
- multiclass classification
- ordinal regression
- hybrid multi-output model
- normal input / mixed continuous-categorical input
- BoTorch 標準 optimizer、mixed optimizer、進化計算 optimizer、torch optimizer、NSGA-II optimizer
- grid rounding、k-sparse、線形制約補修などの candidate repair

Distribution-specific regression models are organized under `bochan.models.regression.beta`, `bochan.models.regression.gamma`, and `bochan.models.regression.count`. They are available through the standard `ModelConfig` registry using the family-specific model keys.

---

## 1. 全体設計

内部処理は、基本的に次の 4 段階です。

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

### Binary prediction contract

`task_type="binary"` では、`BayesianOptimizer.predict()` は利用可能なら
`model.probability_posterior()` を優先します。`mean` はクラス1確率
`p(y=1 | x)` です。

`variance` は通常 `p * (1 - p)` の **Bernoulli observation variance** であり、
確率推定値そのものの epistemic uncertainty ではありません。
`return_result=True` の `PredictionResult` には次が入ります。

- `prediction_space="probability"`
- `variance_kind="bernoulli_observation"`
- observation noise を加えた場合は `bernoulli_observation_plus_noise`

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
| `ModelBundle` | モデル、学習データ、設定、MLL、fit 結果、metadata をまとめる内部オブジェクト |
| `CandidateResult` | 候補点生成結果を保持するオブジェクト |
| `PredictionResult` | `posterior`, `mean`, `variance` などの予測結果を保持するオブジェクト |

`ObjectiveConfig` は現在の API で重要な設定です。ユーザー側で `RegressionScalarObjective` や `BinaryClassificationScoreObjective` を直接選ばなくても、API 側が `task_type` と model 情報から適切な objective を生成します。

ただし `multiclass` の objective 自動生成は現時点では未実装です。multiclass BO では、獲得関数固有の `target_class` や `best_f` を渡すか、必要に応じて `objective` / `objective_factory` を明示します。

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

### 4.1 標準 registry

`ModelConfig` は `input_type`, `task_type`, `model_type` からモデルクラスを解決します。`input_type` を省略した場合、`cat_dims` があれば `mixed`、なければ `normal` として扱います。

#### normal input

| `task_type` | 登録済み `model_type` |
|---|---|
| `regression` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multi_objective` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `binary` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `ordinal` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multiclass` | `base`, `deepgp`, `deepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |

#### mixed input

| `task_type` | 登録済み `model_type` |
|---|---|
| `regression` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multi_objective` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `binary` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `ordinal` | `base`, `deepgp`, `deepkernel`, `deepgpdeepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |
| `multiclass` | `base`, `deepgp`, `deepkernel`, `saas`, `pca`, `rembo`, `rrp`, `hetero` |

`deepgpdeepkernel` は regression / multi_objective / binary / ordinal では登録済みですが、multiclass では独立した registry key としては登録していません。

### 4.2 regression base

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
)
```

`task_type="multi_objective"` も regression 系として扱われます。`train_Y` が多次元の場合は、`multi_output_config` を使って ModelList / hybrid wrapper として構築できます。

### 4.3 binary base

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    model_kwargs={
        "num_inducing_points": 64,
    },
)
```

### 4.4 multiclass base

```python
model_config = ModelConfig(
    task_type="multiclass",
    model_type="base",
    model_kwargs={
        "num_classes": 3,
        "num_inducing_points": 64,
    },
)
```

Multiclass は unordered class の確率を扱います。ordered class として扱いたい場合は `task_type="ordinal"` を使います。

### 4.5 ordinal base

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    model_kwargs={
        "num_classes": 3,
    },
)
```

### 4.6 outcome transform

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

### 4.7 mixed model

`cat_dims` がある場合は、`input_type="mixed"` 側のモデルとして解決されます。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[2, 5],
)
```

binary / multiclass / ordinal mixed も同じ指定方法です。

```python
model_config = ModelConfig(
    task_type="multiclass",
    model_type="base",
    cat_dims=[2],
    model_kwargs={"num_classes": 3},
)
```

### 4.8 deepkernel

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

classification / multiclass / ordinal でも family-specific deepkernel wrapper が登録されています。

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

### 4.9 deepgp

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

### 4.10 deepgpdeepkernel

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="deepgpdeepkernel",
    model_kwargs={
        "feature_extractor": feature_extractor,
        "feature_dim": 8,
        "hidden_dims": [16, 8],
        "num_inducing_points": 64,
    },
)
```

`deepgpdeepkernel` は multiclass の標準 registry にはありません。

### 4.11 SAAS

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="saas",
    model_kwargs={
        "num_taus": 4,
    },
)
```

### 4.12 PCA / REMBO

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

### 4.13 RRP / hetero

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

`hetero` は acquisition registry 側でも `qHetero...` 系に自動解決されます。

### 4.14 model class / factory を直接渡す高度な使い方

通常は文字列指定を推奨しますが、デバッグ時や標準 registry にないモデルを使う場合は直接渡せます。

```python
from botorch.models import SingleTaskGP

model_config = ModelConfig(
    model_cls=SingleTaskGP,
    task_type="regression",
    model_type="base",
)
```

```python
model_config = ModelConfig(
    model_factory=custom_model_factory,
    task_type="regression",
    model_type="custom",
    model_kwargs={"foo": 1},
)
```

`model_cls` / `model_factory` を使う場合も、`task_type` と `model_type` は acquisition / fit / metadata の解決に使われるため、実態に近い値を設定してください。

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
| model に `make_mll()` があり `use_model_make_mll=True` | `model.make_mll()` を最優先 |
| `task_type="regression"` かつ exact GP | `ExactMarginalLogLikelihood` + `fit_gpytorch_mll` |
| `task_type="binary"` | `VariationalELBO` + `fit_binary_classifier_mll` |
| `task_type="binary", model_type="rrp"` | `fit_rrp_binary_classifier_mll` |
| `task_type="ordinal"` | `make_ordinal_mll` + `fit_ordinal_mll` |
| `task_type="ordinal", model_type="rrp"` | `fit_rrp_ordinal_mll` |
| `task_type="multiclass"` | `VariationalELBO` + `fit_multiclass_mll` |
| `model_type` に `deepgp` を含む | `fit_deepgp_mll` |
| `model_type` に `deepkernel` を含む | `fit_deepkernel_mll` |
| model に `fit()` がある | `model.fit` |

### 5.4 高度な上書き

```python
fit_config = FitConfig(
    mll_factory=custom_mll_factory,
    fit_func=custom_fit_func,
    fit_kwargs={"num_epochs": 500},
)
```

`fit_kwargs`, `mll_kwargs`, `optimizer_kwargs`, `batch_size`, `shuffle`, `verbose`, `clip_grad_norm` などは、呼び出し先のシグネチャに合わせて渡されます。

### 5.5 fit をスキップする

すでに学習済みのモデルや、外部で fit 済みの model を渡す場合は `skip_fit=True` にできます。

```python
fit_config = FitConfig(skip_fit=True)
```

---

## 6. input transform / input perturbation

`InputTransformConfig` を使うと、`build_input_transform(...)` を API 側で自動生成できます。

`InputTransformConfig` の主な引数は次です。

| 引数 | 既定値 | 意味 |
|---|---:|---|
| `normalize` | `True` | Normalize を使うか |
| `perturbation` | `False` | 入力摂動を使うか |
| `n_w` | `16` | 1点あたりの摂動サンプル数 |
| `std` | `0.1` | 摂動の標準偏差 |
| `bounds` | `None` | 明示的な bounds。省略時は `train_X` の min/max から生成 |
| `categorical_idx` | `None` | Normalize / perturbation から除外するカテゴリ列。省略時は `cat_dims` を使用 |

### 6.1 Normalize のみ

```python
from bochan.api import InputTransformConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=True,
        perturbation=False,
    ),
)
```

### 6.2 Normalize も input perturbation も使わない

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=False,
        perturbation=False,
    ),
)
```

### 6.3 入力摂動あり

```python
N_W = 8
STD_DEV = 0.1

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=True,
        perturbation=True,
        n_w=N_W,
        std=STD_DEV,
    ),
)
```

これは内部的に以下のような transform 構築へ変換されます。

```python
from bochan.models.transforms.input import build_input_transform

intf = build_input_transform(
    train_X=train_X,
    bounds=bounds,
    perturbation=True,
    categorical_idx=None,
    n_w=N_W,
    std=STD_DEV,
)
```

### 6.4 mixed model での input perturbation

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

### 6.5 明示的に bounds を渡す

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=True,
        perturbation=True,
        n_w=8,
        std=0.1,
        bounds=bounds,
    ),
)
```

### 6.6 input perturbation と objective

入力摂動では `q` が内部で `q * n_w` に展開されるため、獲得関数側で `q * n_w -> q` に戻す objective が必要になる場合があります。

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

`InputTransformConfig(n_w=8)` と `ObjectiveConfig(n_w=8)` は揃えてください。

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

`train_Y.shape[-1]` から出力数を推定し、各出力に同じ `ModelConfig` を複製します。homogeneous regression / multi_objective では、標準では `ModelListGP` が使われます。

### 7.2 homogeneous binary / ordinal multi-output

```python
model_config = ModelConfig(
    task_type="binary",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)
```

binary では `MultiOutputBinaryClassificationModel`、ordinal では `MultiOutputOrdinalModel` が使われます。

### 7.3 multiclass multi-output

Multiclass は homogeneous でも `HybridMultiOutputModel` 側に寄せて扱われます。これは class-probability posterior と objective-space 変換を output spec で明示しやすくするためです。

```python
model_config = ModelConfig(
    task_type="multiclass",
    model_type="base",
    model_kwargs={"num_classes": 3},
    multi_output_config=MultiOutputConfig(
        output_names=["defect_type_a", "defect_type_b"],
        use_hybrid=True,
    ),
)
```

### 7.4 output ごとに task を変える hybrid

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(task_type="binary", model_type="base", name="defect"),
            OutputConfig(task_type="multiclass", model_type="base", name="defect_type", model_kwargs={"num_classes": 3}),
            OutputConfig(task_type="ordinal", model_type="base", name="rank", model_kwargs={"num_classes": 4}),
        ],
        use_hybrid=True,
    ),
)
```

`OutputConfig.name` は hybrid objective で `output="strength"` のように参照できます。

### 7.5 output ごとの fit 設定

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

`OutputConfig` 自体に `fit_config` を埋め込むこともできます。`output_fit_configs` と embedded fit config が両方ある場合は、embedded fit config が優先されます。

### 7.6 wrapper の上書き

標準 wrapper では足りない場合は、`wrapper_cls` または `wrapper_factory` を指定できます。

```python
model_config = ModelConfig(
    task_type="regression",
    multi_output_config=MultiOutputConfig(
        wrapper_cls=CustomWrapper,
        wrapper_kwargs={"foo": 1},
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

`BayesianOptimizer` 経由では、`bundle.task_type`, `bundle.model_type`, `multi_output` を見て短縮名を自動解決します。

```python
AcquisitionConfig(name="BALD")
```

| task_type | `BALD` の解決先 |
|---|---|
| `regression` | `qRegressionBALD` |
| `binary` | `qBinaryBALD` |
| `multiclass` | `qMulticlassBALD` |
| `ordinal` | `qOrdinalBALD` |

同様に、以下も task context に応じて解決されます。

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

### 8.3 binary / multiclass / ordinal の例

```python
model_config = ModelConfig(task_type="binary", model_type="base")
AcquisitionConfig(name="BALD")  # -> qBinaryBALD
AcquisitionConfig(name="EI")    # -> qBinaryExpectedImprovement
```

```python
model_config = ModelConfig(task_type="multiclass", model_type="base", model_kwargs={"num_classes": 3})
AcquisitionConfig(name="BALD")  # -> qMulticlassBALD
AcquisitionConfig(name="EI")    # -> qMulticlassExpectedImprovement
```

```python
model_config = ModelConfig(task_type="ordinal", model_type="base")
AcquisitionConfig(name="BALD")  # -> qOrdinalBALD
AcquisitionConfig(name="EI")    # -> qOrdinalExpectedImprovement
```

### 8.4 hetero の例

`model_type="hetero"` の場合、短縮 alias は `qHetero...` 系へ解決されます。

```python
model_config = ModelConfig(task_type="multiclass", model_type="hetero", model_kwargs={"num_classes": 3})
AcquisitionConfig(name="Entropy")   # -> qHeteroMulticlassPredictiveEntropy
AcquisitionConfig(name="EI")        # -> qHeteroMulticlassExpectedImprovement
```

### 8.5 multi-output の例

`multi_output_config` がある場合、短縮 alias は `qMultiOutput...` または `qHeteroMultiOutput...` 系へ解決されます。

```python
model_config = ModelConfig(
    task_type="ordinal",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)

AcquisitionConfig(name="Entropy")  # -> qMultiOutputOrdinalPredictiveEntropy
AcquisitionConfig(name="NEHVI")    # -> qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement
```

### 8.6 hybrid の例

`task_type="hybrid"` の場合、獲得関数側では regression 系として解決されます。hybrid model の posterior が objective-space 出力を返せるため、BoTorch の regression / multi-objective acquisitions と組み合わせやすい設計です。

```python
AcquisitionConfig(name="EI")   # -> BoTorch qExpectedImprovement
AcquisitionConfig(name="EHI")  # -> BoTorch qExpectedHypervolumeImprovement
AcquisitionConfig(name="KG")   # -> BoTorch qKnowledgeGradient
```

`KG` / `MultiStepLookahead` は regression / hybrid 用です。binary / multiclass / ordinal で短縮指定した場合は、誤解決を避けるためエラーになります。

`EHI` / `EHVI` / `NEHVI` / `NParEGO` は binary / multiclass / ordinal または hetero multi-output 系に解決される場合、multi-output model が必要です。

---

## 9. `ObjectiveConfig`

`ObjectiveConfig` は、API 側で objective を自動構築するための設定です。

通常の API 利用では、ユーザーは `RegressionScalarObjective` や `BinaryClassificationScoreObjective` などを直接選びません。`ObjectiveConfig` に `mode`, `output`, `outputs`, `direction`, `n_w`, `risk_type` などを指定すると、`factory.py` の `build_objective(...)` が `bundle.task_type` に応じて適切な実 objective を生成します。

### 9.1 objective の優先順位

`build_objective(...)` の優先順位は次です。

```python
if acq_config.objective is not None:
    objective = acq_config.objective
elif acq_config.objective_factory is not None:
    objective = acq_config.objective_factory(...)
elif acq_config.objective_config is not None:
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
| `n_w` | input perturbation で 1 点あたりに展開される摂動数 |
| `risk_type` | `None`, `"var"`, `"cvar"` |
| `alpha` | VaR / CVaR の risk 集約パラメータ |
| `maximize` | risk 集約時に大きい値を良い方向として扱うか |
| `aggregate_mean_when_no_risk` | risk 指定なしのときに摂動方向を平均集約するか |
| `allow_unexpanded` | `n_w` 未展開の入力を許容するか |
| `utility_values` | ordinal class の utility 値 |
| `ordinal_likelihood` | ordinal likelihood の明示指定 |
| `objective_kwargs` | objective 生成時に追加で渡す引数 |

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

## 11. binary / multiclass / ordinal objective

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

### 11.2 multiclass classification

multiclass では、現在 `ObjectiveConfig` からの自動 objective 生成は行いません。多くの multiclass acquisition は `target_class` を直接受け取り、内部で `p(target_class | x)` を score として扱います。

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "target_class": 2,
        "best_f": 0.70,
    },
)
```

level-set / feasibility 系でも `target_class` と `threshold` を指定します。

```python
acq_config = AcquisitionConfig(
    name="Straddle",
    acqf_kwargs={
        "target_class": 1,
        "threshold": 0.5,
    },
)
```

### 11.3 ordinal

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
| `multi_objective` | `MultiObjectiveConfig` |
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
| `optimizer` | `optimize_acqf`, `optimize_acqf_mixed`, `evo`, `torch`, `nsgaii` など |
| `optimizer_kwargs` | optimizer に渡す追加引数 |
| `post_processing_func` | 候補点の後処理関数 |
| `repair_config` | grid rounding / k-sparse / 制約補修を自動生成する設定 |
| `fixed_features` | 固定する特徴量 |
| `fixed_features_list` | mixed optimizer 用の固定特徴量候補リスト |
| `inequality_constraints` | 線形不等式制約 |
| `equality_constraints` | 線形等式制約 |
| `return_best_only` | 最良候補のみ返すか |

### 14.1 optimizer の種類

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="optimize_acqf_mixed")
OptimizeConfig(optimizer="evo")
OptimizeConfig(optimizer="optimize_acqf_evo")
OptimizeConfig(optimizer="evo_mixed")
OptimizeConfig(optimizer="optimize_acqf_evo_mixed")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="optimize_acqf_torch")
OptimizeConfig(optimizer="torch_mixed")
OptimizeConfig(optimizer="optimize_acqf_torch_mixed")
OptimizeConfig(optimizer="nsgaii")
OptimizeConfig(optimizer="optimize_acqf_nsgaii")
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

### 14.3 evo / torch / NSGA-II optimizer

進化計算 optimizer や torch optimizer を使う場合は、`optimizer_kwargs` にそれぞれの backend 用の設定を渡します。

```python
opt_config = OptimizeConfig(
    optimizer="evo",
    q=3,
    optimizer_kwargs={
        "method": "cmaes",
        "maxiter": 200,
    },
)
```

```python
opt_config = OptimizeConfig(
    optimizer="torch",
    q=3,
    optimizer_kwargs={
        "num_steps": 200,
        "lr": 0.05,
    },
)
```

```python
opt_config = OptimizeConfig(
    optimizer="nsgaii",
    q=10,
    optimizer_kwargs={
        "population_size": 128,
        "num_generations": 100,
    },
)
```

NSGA-II は多目的・制約付き・非滑らかな探索空間で、勾配ベース optimizer より扱いやすい場合があります。一方で BoTorch の標準 `optimize_acqf` と同じ API 互換性を前提にしない backend 固有設定が必要になることがあります。

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

主な引数は次です。

| 引数 | 意味 |
|---|---|
| `bounds` | 補修後に使う bounds |
| `numeric_indices` | 丸め対象の数値列 |
| `steps` | 各数値列の刻み幅。`None` なら grid rounding しない |
| `comp_idx` | k-sparse 対象列。`None` / `[]` なら k-sparse を使わない |
| `k` | 非ゼロまたは選択する成分数 |
| `equality_constraints` | 線形等式制約 |
| `inequality_constraints` | 線形不等式制約 |
| `inequality_sense` | `"le"` または `"ge"` |
| `fixed_features` | 固定値 |
| `final_sum_constraint` | 最終的な和制約 |
| `diversify` | 補修後候補の多様化を行うか |
| `diversify_kwargs` | 多様化処理への追加引数 |
| `score` | support selection の score。`"abs"` または `"value"` |
| `support_selection` | `"topk"` または `"sample"` |
| `sample_tau` | sample support selection の温度 |
| `sample_eps` | sample support selection の epsilon |
| `generator` | sampling 用 generator |
| `max_iters` | 制約補修の最大反復数 |
| `num_alternations` | 交互補修の回数 |
| `final_priority` | `"grid"` または `"constraints"` |
| `support_eps` | support 判定の epsilon |

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

### 15.3 support selection

`support_selection="topk"` は score 上位の成分を選びます。`support_selection="sample"` は score に基づいて確率的に support を選びます。

```python
repair_config = CandidateRepairConfig(
    numeric_indices=[0, 1, 2, 3],
    steps=[0.1, 0.1, 0.1, 0.1],
    comp_idx=[0, 1, 2, 3],
    k=2,
    support_selection="sample",
    sample_tau=0.2,
    sample_eps=0.05,
)
```

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

### 16.4 multiclass + predictive entropy

```python
model_config = ModelConfig(
    task_type="multiclass",
    model_type="base",
    model_kwargs={
        "num_classes": 3,
        "num_inducing_points": 64,
    },
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(name="Entropy")
candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

### 16.5 multiclass + target-class EI

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "target_class": 2,
        "best_f": 0.7,
    },
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=1))
```

### 16.6 ordinal + expected utility EI

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

## 17. level-set / active learning 実行例

### 17.1 regression level-set

```python
acq_config = AcquisitionConfig(
    name="Straddle",
    acqf_kwargs={
        "threshold": 0.5,
    },
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

### 17.2 multiclass level-set

Class 2 の確率が 0.5 付近になる境界を探索する例です。

```python
acq_config = AcquisitionConfig(
    name="Straddle",
    acqf_kwargs={
        "target_class": 2,
        "threshold": 0.5,
    },
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

### 17.3 ordinal boundary

```python
acq_config = AcquisitionConfig(
    name="BoundaryVariance",
    acqf_kwargs={
        "target_boundary_idx": 1,
    },
)

candidates, acq_value = bo.candidate(acq_config, OptimizeConfig(q=3))
```

---

## 18. 多目的実行例

### 18.1 regression multi-output + NEHVI

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

### 18.2 multiclass multi-output + NEHVI

```python
model_config = ModelConfig(
    task_type="multiclass",
    model_type="base",
    model_kwargs={"num_classes": 3},
    multi_output_config=MultiOutputConfig(
        output_names=["defect_a", "defect_b"],
        use_hybrid=True,
    ),
)

bo = BayesianOptimizer(
    model_config=model_config,
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)

bo.fit(train_X, train_Y_multi)

acq_config = AcquisitionConfig(
    name="NEHVI",
    acqf_kwargs={
        "target_classes": [1, 2],
    },
)
```

Multiclass multi-output の詳細な `target_class` / output reduction 引数は、使用する `qMultiOutputMulticlass...` クラスの実装に合わせて指定します。

### 18.3 hybrid + NEHVI

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

## 19. `BochanStudy`

`BochanStudy` は `BayesianOptimizer` を使った Optuna / Ax 風の最適化ループ API です。候補生成と評価が別プロセスになる場合、`ask()` / `tell()` を使います。

```python
from bochan.api import BochanStudy

study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
)

study.optimize(
    objective_func=lambda X: X.sum(dim=-1),
    n_trials=20,
    q=2,
    save_path="study.json",
)
```

Human-in-the-loop の例です。

```python
batch = study.ask(q=3, mark_running=True, return_batch=True)

# batch.candidates を実験・Web UI・外部 simulator に渡す。
# 測定値が得られたら tell で登録する。

study.tell(batch, measured_values)
study.save("study.json")
```

`BochanStudy` の詳細は `src/bochan/api/STUDY_README.md` を参照してください。

---

## 20. 注意点

### 20.1 `best_f` は objective 後の値に合わせる

`direction="minimize"`, `output=1`, `eq_target=...` などを使う場合、`best_f` は変換後の objective value に合わせてください。

```python
# y1 を最大化する場合
best_f = train_Y[:, 1].max()

# y1 を最小化する場合
best_f = (-train_Y[:, 1]).max()
```

### 20.2 `ref_point` / `Y_baseline` は選択出力に合わせる

`outputs=[0, 2]` のように一部出力だけを使う場合、`ref_point` と `Y_baseline` も同じ出力次元に合わせます。

```python
Y_baseline = train_Y[:, [0, 2]]
ref_point = torch.tensor([r0, r2], dtype=train_Y.dtype, device=train_Y.device)
```

### 20.3 input perturbation の `n_w` を揃える

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

### 20.4 `ObjectiveConfig` が不要な場合もある

single-output regression で、摂動なし、目的値をそのまま最大化するだけなら objective は不要です。

active learning 系の獲得関数でも、獲得関数本体がすでに `batch_shape x q` の score を返す場合は objective が不要なことがあります。

### 20.5 multiclass objective は明示する

multiclass は `ObjectiveConfig` の自動生成対象外です。`target_class`, `threshold`, `best_f` など、使用する multiclass acquisition class が要求する引数を `acqf_kwargs` で渡してください。

### 20.6 上級者向け override

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
# Cross-validation

The regular API can evaluate any model built by `BayesianOptimizer` without
changing the optimizer's fitted state:

```python
from bochan.api import BayesianOptimizer, CrossValidationConfig

result = optimizer.cross_validate(
    train_X,
    train_Y,
    cv_config=CrossValidationConfig(n_splits=5, shuffle=True, random_state=0),
)
print(result.test_metric_summary["rmse"].mean)  # fold mean
print(result.output.oof_metrics["rmse"])        # metric over all OOF rows
```

Regression outputs report RMSE, MAE, MAPE, and R2; classification outputs
report accuracy, precision, recall, and F1. Multi-output and hybrid results are
kept separate (for example, `result.outputs["strength"]` and
`result.outputs["phase"]`) so metrics on different scales are never silently
averaged. OOF rows are restored to input order. `aggregated_train_predictions`
contains each row's mean prediction, `fold_prediction_std`, and
`prediction_count` across the fold models that trained on it.

`predictive_std` is derived from each GP posterior. It is distinct from
`fold_prediction_std`, which measures disagreement between fitted fold models.
For ordinary K-fold validation every OOF row is predicted once, so its latter
value is NaN. Binary `variance_kind` is retained because Bernoulli observation
variance is not necessarily epistemic uncertainty.

Use `splitter="loo"` for leave-one-out validation or pass any sklearn-compatible
splitter object. LOO validation-fold R2 is NaN (one row cannot define R2), while
the combined OOF R2 remains available. The safe default MAPE policy returns NaN
and records a warning when targets are zero; choose `ignore` or `clip` and set
`mape_epsilon` when another policy is appropriate.

Every fold receives a deep-copied model/fit configuration and a fresh optimizer.
Models are retained only with `return_models=True`. Cross-validation therefore
does not alter `model`, `bundle`, `mll`, training data, bounds, or history on the
calling optimizer. It does, however, fit one new model per fold and consequently
costs approximately the number of folds times a single fit.
