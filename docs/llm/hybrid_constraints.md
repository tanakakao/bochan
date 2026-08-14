# LLMによるHybrid・制約付きベイズ最適化

このREADMEでは、`bochan` のLLM Plannerを使って次の設定を提案させる方法を説明します。

- 回帰・二値分類・ordinalを組み合わせたHybridモデル
- 最大化・最小化する目的変数と重み
- 数値目的変数の上下限制約
- 分類出力のクラス確率制約
- ordinal出力のrank確率制約
- 原料合計、線形制約、固定変数、刻み幅、候補修復
- `suggest_*()` と `AcquisitionConfig(name="llm_selected")` の使い分け

LLMは設定案を生成しますが、モデル学習、制約評価、獲得関数計算、候補点の再順位付けは `bochan` が実行します。

---

## 1. 3種類の設定を区別する

制約付きHybrid最適化では、次の3層を混同しないことが重要です。

| 層 | Config | 例 |
|---|---|---|
| 最適化目的 | `ObjectiveConfig` | `property`を最大化、`cost`を最小化 |
| 目的変数側の実行可能条件 | `OutcomeConstraintConfig` | 合格確率が0.8以上、収縮率が0.3以下 |
| 説明変数・候補点側の条件 | `OptimizeConfig` / `CandidateRepairConfig` | 原料合計1.0、0.05刻み、雰囲気固定 |

Hybridモデルの出力構成は `ModelConfig.multi_output_config` に設定します。

---

## 2. 想定するデータ

説明変数:

```text
raw material 1
raw material 2
raw material 3
temperature
time
atmosphere
```

目的変数:

```text
property          : 連続値。高いほど望ましい
pass_probability  : 二値分類。class 1が合格
quality_rank      : ordinal。0 < 1 < 2 < 3
```

`train_Y` の列順は次とします。

```python
# train_Y[:, 0] = property
# train_Y[:, 1] = pass_probability
# train_Y[:, 2] = quality_rank
```

Hybridの `output_configs` は、この列順と必ず一致させます。

---

## 3. 共通LLM設定

```python
from bochan.llm import LLMConfig, LLMContextConfig, LLMSettings

llm_context = LLMContextConfig(
    variable_names=[
        "raw material 1",
        "raw material 2",
        "raw material 3",
        "temperature",
        "time",
        "atmosphere",
    ],
    target_names=[
        "property",
        "pass_probability",
        "quality_rank",
    ],
    variable_descriptions={
        "raw material 1": "主原料1。組成比。",
        "raw material 2": "主原料2。組成比。",
        "raw material 3": "添加成分。組成比。",
        "temperature": "処理温度。degC。",
        "time": "保持時間。hour。",
        "atmosphere": "0=air、1=N2、2=Arのカテゴリ変数。",
    },
    target_descriptions={
        "property": "連続特性。高いほど望ましい。",
        "pass_probability": "二値の合否。class 1が合格。",
        "quality_rank": "ordinal品質ランク。0から3で高いほど望ましい。",
    },
    domain_notes=[
        "raw material 1から3の合計は1.0にする。",
        "各原料は0.05刻み。",
        "pass_probabilityはclass 1の確率が0.8以上必要。",
        "quality_rankは2以上となる確率が0.8以上必要。",
    ],
)

llm_settings = LLMSettings(
    goal=(
        "propertyを最大化する。"
        "pass_probabilityの合格確率とquality_rankを実行可能条件にする。"
    ),
    llm_config=LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    llm_context=llm_context,
    n_llm_candidates=50,
)
```

APIキーはコードに直接書かず、環境変数へ保存してください。

---

## 4. `suggest_all()`で全設定を提案させる

重要な実験では、まずreview-first APIを使います。

```python
from bochan.api import BayesianOptimizer, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(),
    bounds=bounds,
    llm_settings=llm_settings,
)

suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
    model_prompt=(
        "propertyはregression、pass_probabilityはbinary、"
        "quality_rankは4クラスordinalとしてHybridモデルを構成する。"
    ),
    acquisition_prompt=(
        "propertyを最大化する。"
        "class 1の確率が0.8以上、quality_rankが2以上となる確率が"
        "0.8以上という制約を使う。観測ノイズも考慮する。"
    ),
    optimizer_prompt=(
        "raw material 1から3の合計を1.0にする。"
        "各原料は0.05刻み。atmosphereはカテゴリとして扱う。"
        "候補はq=3とする。"
    ),
)
```

返却された設定を確認します。

```python
print(suggestion.model_config)
print(suggestion.fit_config)
print(suggestion.acq_config)
print(suggestion.opt_config)
print(suggestion.reasoning_summary)
print(suggestion.warnings)
```

問題がなければ適用します。

```python
bo.apply_suggestion(suggestion)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate()
```

モデル構成を適用した後は、必ず `fit()` または `refit()` を実行してください。

---

## 5. Plannerが返すHybrid構造

LLM Plannerには次の専用schemaが渡されます。

```python
{
    "model_config": {
        "task_type": "hybrid",
        "model_type": "base",
        "multi_output_config": {
            "output_names": [
                "property",
                "pass_probability",
                "quality_rank",
            ],
            "output_configs": [
                {
                    "name": "property",
                    "task_type": "regression",
                    "model_type": "base",
                    "model_kwargs": {},
                    "output_spec_kwargs": {},
                },
                {
                    "name": "pass_probability",
                    "task_type": "binary",
                    "model_type": "base",
                    "model_kwargs": {},
                    "output_spec_kwargs": {},
                },
                {
                    "name": "quality_rank",
                    "task_type": "ordinal",
                    "model_type": "base",
                    "model_kwargs": {"num_classes": 4},
                    "output_spec_kwargs": {},
                },
            ],
            "use_hybrid": True,
            "fit_submodels": True,
            "fit_wrapper": False,
        },
    }
}
```

異なる `task_type` が含まれる場合、Plannerは親 `task_type` を `hybrid` に正規化します。

`LLMContextConfig.target_names` が指定されていれば、欠けている `output_names` と各 `output_configs[].name` も補完されます。

---

## 6. 数値目的変数の制約

例えば `shrinkage <= 0.3` は次の構造です。

```python
{
    "kind": "feasibility",
    "output": "shrinkage",
    "threshold": 0.3,
    "sense": "le",
    "margin": 0.0,
    "scale": 1.0,
}
```

`output` は出力indexまたは `target_names` の名前を使えます。

利用可能な `sense`:

```text
ge: 出力 >= threshold
le: 出力 <= threshold
eq: |出力 - threshold| <= margin
```

---

## 7. 分類出力のクラス確率制約

`pass_probability` のclass 1確率が0.8以上という条件は次です。

```python
{
    "kind": "feasibility",
    "output": "pass_probability",
    "threshold": 0.8,
    "sense": "ge",
    "target_class": 1,
}
```

複数クラスをまとめて実行可能とする場合は `target_classes` を使います。

```python
{
    "kind": "feasibility",
    "output": "material_class",
    "threshold": 0.9,
    "sense": "ge",
    "target_classes": [1, 2],
}
```

この場合は次を評価します。

```text
P(class in {1, 2}) >= 0.9
```

`target_class` と `target_classes` は同時に指定しません。

---

## 8. ordinal rank確率制約

`quality_rank >= 2` となる確率が0.8以上という条件は次です。

```python
{
    "kind": "ordinal_rank",
    "output": "quality_rank",
    "rank": 2,
    "sense": "ge",
    "probability_threshold": 0.8,
}
```

意味は次のとおりです。

```text
P(quality_rank >= 2) >= 0.8
```

`rank` はordinal classのindexです。クラス定義が曖昧な場合、LLMに推測させず、`target_descriptions` で明示してください。

---

## 9. `OutcomeConstraintConfig`全体

```python
{
    "acquisition_config": {
        "name": "NEHVI",
        "objective_config": {
            "mode": "multi_output",
            "outputs": ["property"],
            "directions": ["maximize"],
            "weights": [1.0],
        },
        "outcome_constraint_config": {
            "constraints": [
                {
                    "kind": "feasibility",
                    "output": "pass_probability",
                    "threshold": 0.8,
                    "sense": "ge",
                    "target_class": 1,
                },
                {
                    "kind": "ordinal_rank",
                    "output": "quality_rank",
                    "rank": 2,
                    "sense": "ge",
                    "probability_threshold": 0.8,
                },
            ],
            "eta": 0.001,
            "reduce_constraints": "prod",
            "reduce_q": "mean",
            "posterior_mode": "objective",
            "min_feasibility": 0.0,
            "detach_feasibility": False,
        },
    }
}
```

複数制約の結合方法は `reduce_constraints` で指定します。通常は `prod` を使います。

---

## 10. 説明変数側の線形制約

原料1から3の合計を1.0にする等式制約は次です。

```python
{
    "indices": [
        "raw material 1",
        "raw material 2",
        "raw material 3",
    ],
    "coefficients": [1.0, 1.0, 1.0],
    "rhs": 1.0,
}
```

Plannerは `LLMContextConfig.variable_names` を使い、既知の変数名を列indexへ変換します。

上の構造は内部的に次へ正規化されます。

```python
([0, 1, 2], [1.0, 1.0, 1.0], 1.0)
```

`OptimizeConfig` の例:

```python
{
    "optimize_config": {
        "optimizer": "optimize_acqf",
        "q": 3,
        "raw_samples": 256,
        "num_restarts": 10,
        "equality_constraints": [
            {
                "indices": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                ],
                "coefficients": [1.0, 1.0, 1.0],
                "rhs": 1.0,
            }
        ],
        "inequality_constraints": [],
    }
}
```

---

## 11. `CandidateRepairConfig`

獲得関数最適化後の丸め、固定、合計補修などは `repair_config` へ設定します。

```python
{
    "optimize_config": {
        "optimizer": "optimize_acqf",
        "q": 3,
        "repair_config": {
            "numeric_indices": [
                "raw material 1",
                "raw material 2",
                "raw material 3",
                "temperature",
                "time",
            ],
            "steps": {
                "raw material 1": 0.05,
                "raw material 2": 0.05,
                "raw material 3": 0.05,
                "temperature": 10.0,
            },
            "comp_idx": [
                "raw material 1",
                "raw material 2",
                "raw material 3",
            ],
            "fixed_features": {
                "atmosphere": 1.0,
            },
            "final_sum_constraint": {
                "indices": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                ],
                "rhs": 1.0,
            },
            "diversify": True,
            "max_iters": 12,
            "num_alternations": 2,
            "final_priority": "grid",
        },
    }
}
```

変数名は次の項目でindexへ正規化されます。

- `numeric_indices`
- `comp_idx`
- `steps` のキー
- `fixed_features` のキー
- `equality_constraints[].indices`
- `inequality_constraints[].indices`
- `final_sum_constraint.indices`

直接Pythonで `CandidateRepairConfig` を作る場合は、列indexを使うのが最も確実です。

```python
from bochan.api import CandidateRepairConfig, OptimizeConfig

opt_config = OptimizeConfig(
    optimizer="optimize_acqf",
    q=3,
    repair_config=CandidateRepairConfig(
        numeric_indices=[0, 1, 2, 3, 4],
        steps={0: 0.05, 1: 0.05, 2: 0.05, 3: 10.0},
        comp_idx=[0, 1, 2],
        fixed_features={5: 1.0},
        final_sum_constraint=([0, 1, 2], 1.0),
        diversify=True,
    ),
)
```

---

## 12. 獲得関数だけを実行時に自動選択する

モデル、目的方向、制約は人が固定し、獲得関数の種類だけをLLMに選ばせる方法です。

```python
from bochan.api import (
    AcquisitionConfig,
    ObjectiveConfig,
    OptimizeConfig,
    OutcomeConstraintConfig,
)

acq_config = AcquisitionConfig(
    name="llm_selected",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=["property"],
        directions=["maximize"],
        weights=[1.0],
    ),
    outcome_constraint_config=OutcomeConstraintConfig(
        constraints=[
            {
                "kind": "feasibility",
                "output": "pass_probability",
                "threshold": 0.8,
                "sense": "ge",
                "target_class": 1,
            },
            {
                "kind": "ordinal_rank",
                "output": "quality_rank",
                "rank": 2,
                "sense": "ge",
                "probability_threshold": 0.8,
            },
        ],
    ),
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=OptimizeConfig(
        optimizer="llm_candidate_set",
        q=3,
    ),
)
```

この経路では、明示した `objective_config` と `outcome_constraint_config` がLLM提案より優先されます。

解決後の設定:

```python
print(bo.acq_config)
print(bo.last_acquisition_suggestion.reasoning_summary)
print(bo.last_acquisition_suggestion.warnings)
```

---

## 13. `suggest_*()`との使い分け

### 事前確認が必要

```python
suggestion = bo.suggest_all(...)
# または
suggestion = bo.suggest_model(...)
suggestion = bo.suggest_acquisition(...)
suggestion = bo.suggest_optimizer(...)
```

適した用途:

- Hybridの出力構成を確認したい
- 制約の意味としきい値を確認したい
- LLMの理由と警告を保存したい
- 実験前に人が承認したい
- 一部のConfigだけ適用したい

### 実行時に自動選択

```python
AcquisitionConfig(name="llm_selected")
```

適した用途:

- モデルは学習済み
- 目的方向と制約は固定済み
- 現在の問題に合う獲得関数だけ自動選択したい
- バッチ処理や反復運用でコードを簡潔にしたい

Hybridモデルの構成変更は `fit()` 前に必要なため、`AcquisitionConfig(name="llm_selected")` では変更されません。

---

## 14. オフラインテスト

`planner_response` を使うと、providerを呼ばずにschema変換をテストできます。

```python
suggestion = bo.suggest_all(
    train_X=train_X,
    train_Y=train_Y,
    llm_context=llm_context,
    planner_response={
        "model_config": {
            "task_type": "hybrid",
            "model_type": "base",
            "multi_output_config": {
                "output_configs": [
                    {"task_type": "regression", "model_type": "base"},
                    {"task_type": "binary", "model_type": "base"},
                    {
                        "task_type": "ordinal",
                        "model_type": "base",
                        "model_kwargs": {"num_classes": 4},
                    },
                ]
            },
        },
        "fit_config": {
            "method": "auto",
            "skip_fit": True,
        },
        "acquisition_config": {
            "name": "NEHVI",
            "objective_config": {
                "mode": "multi_output",
                "outputs": ["property"],
                "directions": ["maximize"],
                "weights": [1.0],
            },
            "outcome_constraint_config": {
                "constraints": [
                    {
                        "kind": "feasibility",
                        "output": "pass_probability",
                        "threshold": 0.8,
                        "sense": "ge",
                        "target_class": 1,
                    },
                    {
                        "kind": "ordinal_rank",
                        "output": "quality_rank",
                        "rank": 2,
                        "sense": "ge",
                        "probability_threshold": 0.8,
                    },
                ]
            },
        },
        "optimize_config": {
            "optimizer": "optimize_acqf",
            "q": 3,
            "repair_config": {
                "comp_idx": [
                    "raw material 1",
                    "raw material 2",
                    "raw material 3",
                ],
                "fixed_features": {"atmosphere": 1.0},
                "final_sum_constraint": {
                    "indices": [
                        "raw material 1",
                        "raw material 2",
                        "raw material 3",
                    ],
                    "rhs": 1.0,
                },
            },
        },
    },
)
```

確認:

```python
assert suggestion.model_config.task_type == "hybrid"
assert suggestion.model_config.multi_output_config.use_hybrid is True
assert suggestion.acq_config.outcome_constraint_config is not None
assert suggestion.opt_config.repair_config.comp_idx == [0, 1, 2]
assert suggestion.opt_config.repair_config.fixed_features == {5: 1.0}
```

---

## 15. 安全な運用方針

LLMに任せやすい項目:

- `model_type` の候補比較
- 獲得関数の種類
- `q`、`raw_samples`、`num_restarts` の初期案
- mixed変数に適したoptimizer backend
- Hybrid構成の下書き

人が明示すべき項目:

- `train_Y` の列順と各列の意味
- 最大化・最小化方向
- objective weight
- 合否のtarget class
- ordinal rankの意味
- feasibility threshold
- 組成合計値
- 法規・安全・製造上のハード制約

Plannerはこれらが曖昧な場合、値を推測せず `warnings` に記録するよう指示されています。

---

## 16. トラブルシューティング

### Hybridにならない

`target_names` と各目的変数の型を明示します。

```python
model_prompt=(
    "train_Y列0はregression、列1はbinary、列2はordinal。"
    "親task_typeをhybrid、use_hybrid=Trueにする。"
)
```

### 出力名がずれる

`LLMContextConfig.target_names` の順序を `train_Y` の列順と一致させます。

### 制約が目的関数として扱われる

プロンプトで次を分けます。

```text
propertyは最適化目的
pass_probabilityとquality_rankはOutcomeConstraintConfig
原料合計と刻み幅はCandidateRepairConfig
```

### 変数名がindexへ変換されない

`variable_names` に完全一致する名前を登録します。大文字小文字や空白も一致させてください。

### LLMがしきい値を推測する

しきい値を `goal`、`target_descriptions`、`domain_notes`、または明示Configで固定します。

---

## 17. 関連API

```python
from bochan.llm import build_config_planner_prompt, plan_configs
```

Plannerへ渡される専用schemaを確認する場合:

```python
prompt = build_config_planner_prompt(
    goal=llm_settings.goal,
    llm_context=llm_context,
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="full",
)
print(prompt)
```

設定辞書だけを取得する場合:

```python
plan = plan_configs(
    goal=llm_settings.goal,
    llm_config=llm_settings.llm_config,
    llm_context=llm_context,
    train_X=train_X,
    train_Y=train_Y,
    bounds=bounds,
    mode="full",
)
```

返却値は `model_config`、`fit_config`、`acquisition_config`、`optimize_config`、`warnings`、`reasoning_summary` を含む辞書です。
