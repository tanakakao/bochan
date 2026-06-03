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
from bochan.api import DEFAULT_MODEL_REGISTRY, MODEL_REGISTRY
```

---

## 2. 設定クラス

| 設定クラス | 役割 |
|---|---|
| `ModelConfig` | モデルの種類、カテゴリ変数、input transform、outcome transform、モデル固有引数を指定する |
| `FitConfig` | 学習回数、learning rate、maxiter などを指定する |
| `InputTransformConfig` | Normalize / input perturbation を簡易指定する |
| `MultiOutputConfig` | multi-output / hybrid の出力ごとのモデル構成を指定する |
| `OutputConfig` | multi-output / hybrid の各出力を詳細指定する |
| `AcquisitionConfig` | 獲得関数、objective、sampler、獲得関数固有引数を指定する |
| `ObjectiveConfig` | regression / binary / ordinal / hybrid に応じた objective を API 側で自動生成する |
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

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)

bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
)

opt_config = OptimizeConfig(q=1, num_restarts=10, raw_samples=256)

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

### 4.1 outcome transform

`outcome_transform` は regression 系モデルの Standardize を使うかどうかを指定します。

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

`hybrid` では wrapper 自体には outcome transform を渡しません。親の `outcome_transform` は submodel 側へ継承され、regression / multi_objective 出力だけに適用されます。

```python
model_config = ModelConfig(
    task_type="hybrid",
    outcome_transform=True,
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),  # Standardize あり
            OutputConfig(task_type="binary", name="defect"),        # 無効
            OutputConfig(task_type="ordinal", name="rank"),         # 無効
        ],
        use_hybrid=True,
    ),
)
```

### 4.2 mixed model

`cat_dims` がある場合は、`mixed` 側のモデルとして解決されます。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[2, 5],
)
```

### 4.3 deepkernel / deepgp / SAAS / PCA / REMBO

モデル固有の引数は `model_kwargs` に渡します。

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

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="saas",
    model_kwargs={"num_taus": 4},
)
```

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="pca",
    model_kwargs={"latent_dim": 5, "standardize_X": True},
)
```

---

## 5. `FitConfig`

通常利用では `fit_func`, `mll_factory`, `mll_cls` を指定する必要はありません。`task_type` / `model_type` / model の `make_mll()` から内部で自動選択します。

```python
fit_config = FitConfig(maxiter=128)
```

classification / ordinal / deep 系では epoch ベースの学習設定を使います。

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.01,
)
```

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

高度な上書きもできます。

```python
fit_config = FitConfig(
    mll_factory=custom_mll_factory,
    fit_func=custom_fit_func,
    fit_kwargs={"num_epochs": 500},
)
```

---

## 6. input transform / input perturbation

`InputTransformConfig` を使うと、`build_input_transform(...)` を API 側で自動生成できます。

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

入力摂動ありの場合です。

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

入力摂動では `q` が内部で `q * n_w` に展開されるため、獲得関数側で `q * n_w -> q` に戻す objective が必要になる場合があります。通常は `ObjectiveConfig` に `n_w` を指定すると、API 側で task に応じた objective を自動生成します。

---

## 7. objective の自動生成

通常の API 利用では、`RegressionScalarObjective` や `BinaryClassificationScoreObjective` などをユーザーが直接選ぶ必要はありません。`ObjectiveConfig` に出力・方向・risk 集約を指定すると、API 側が `bundle.task_type` と model 情報から適切な objective を選びます。

優先順位は次です。

```python
if acq_config.objective is not None:
    # 完全手動。生成済み objective をそのまま使う。
    objective = acq_config.objective
elif acq_config.objective_factory is not None:
    # 上級者向け。factory で上書き。
    objective = acq_config.objective_factory(...)
elif acq_config.objective_config is not None:
    # 通常ルート。API が task_type に応じて自動生成。
    objective = build_objective_from_config(...)
else:
    objective = None
```

### 7.1 regression: single-output

single-output regression で摂動なし、目的値をそのまま最大化する場合は objective なしで構いません。

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
)
```

入力摂動や risk 集約を使う場合は、`ObjectiveConfig` を使います。

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
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

API 側では `RegressionScalarObjective` が選択されます。

### 7.2 regression: multi-output から特定出力だけ使う

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

この場合、API 側では `RegressionScalarObjective(output_index=1, ...)` を生成します。`best_f` は必ず対象出力に合わせてください。

### 7.3 regression: multi-output から一部の出力だけ多目的に使う

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

`outputs` を指定した場合、`Y_baseline` と `ref_point` も選択後の出力に合わせるのが安全です。

### 7.4 binary classification

binary classification では、入力摂動で展開された score を `q * n_w -> q` に戻す目的で objective を使います。

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

### 7.5 ordinal

ordinal では latent `f` を class probability に変換し、`utility_values` で expected utility に変換するため、BO 系では objective を指定するのが基本です。

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

API 側では `OrdinalInputPerturbationExpectedUtilityObjective` が選択され、`ordinal_likelihood` は model から取得されます。

### 7.6 hybrid

hybrid では出力名または index を使って objective を指定します。内部では既存の `HybridObjectiveSpec`, `make_hybrid_scalar_objective`, `make_hybrid_multi_output_objective` が使われます。

single-output hybrid の例です。

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

multi-output hybrid の例です。

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

直接 `HybridObjectiveSpec` を渡すこともできます。

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

### 7.7 上級者向け: objective を直接渡す

既存の objective を直接渡すこともできます。

```python
from bochan.acquisition.objective import RegressionScalarObjective

objective = RegressionScalarObjective(n_w=8, risk_type=None)

acq_config = AcquisitionConfig(
    name="EI",
    objective=objective,
    acqf_kwargs={"best_f": train_Y.max()},
)
```

`objective_factory` による上書きも可能ですが、通常は `ObjectiveConfig` を推奨します。

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

`task_type="hybrid"` の場合、獲得関数側では regression として解決されます。

```python
AcquisitionConfig(name="EI")   # -> BoTorch qExpectedImprovement
AcquisitionConfig(name="EHI")  # -> BoTorch qExpectedHypervolumeImprovement
AcquisitionConfig(name="KG")   # -> BoTorch qKnowledgeGradient
```

`KG` / `MultiStepLookahead` は regression / hybrid 用です。binary / ordinal で短縮指定した場合は、誤解決を避けるためエラーになります。

---

## 9. 候補点最適化

```python
opt_config = OptimizeConfig(
    q=3,
    num_restarts=20,
    raw_samples=512,
    sequential=True,
)

candidates, acq_value = bo.candidate(acq_config, opt_config)
```

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
