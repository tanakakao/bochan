# BochanStudy: ask/tell と optimize ループ

`BochanStudy` は、`BayesianOptimizer` を候補点生成エンジンとして利用し、その上に Optuna / Ax 風の最適化ループを提供する高レベル API です。

主な用途は次のとおりです。

- Python 関数を自動評価するベイズ最適化
- `ask()` / `tell()` による実験・シミュレーション連携
- 初期ランダム探索からモデルベース探索への自動切り替え
- trial 履歴の保存・再開
- early stopping
- generation schedule による探索戦略の切り替え

`BochanStudy` は config dataclass を直接渡す方法に加えて、`TabularBayesianOptimizer` と同様に辞書形式の設定を受け取ります。通常の単目的回帰では、config を省略しても内部デフォルトで実行できます。

---

## 1. 最小例

```python
import torch

from bochan.api import BochanStudy

bounds = torch.tensor(
    [[0.0, 0.0], [1.0, 1.0]],
    dtype=torch.double,
)

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

train_X, train_Y = study.completed_data()
```

この例では、最初の10 trialをランダム生成し、その後は学習済みモデルと獲得関数を使って候補点を生成します。

configを省略した場合、内部では次の設定が使われます。

```python
ModelConfig(
    task_type="regression",
    model_type="base",
)

FitConfig()

AcquisitionConfig(
    name="EI",
)

OptimizeConfig()

DataContext(
    bounds=bounds,
)
```

`EI` に必要な `best_f` は、完了済みの観測データから高レベル API が自動計算します。

> [!NOTE]
> `n_initial_random=0` かつ既存観測データがない場合、最初のモデルを学習できません。`n_initial_random` を1以上にするか、`add_observations()` で既存データを登録してください。

---

## 2. 辞書形式で設定する

configクラスをimportせず、JSONに近い辞書形式で設定できます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
    model_config={
        "task_type": "regression",
        "model_type": "base",
        "input_transform_config": {
            "normalize": True,
            "perturbation": False,
            "n_w": 4,
            "std": 0.1,
        },
    },
    fit_config={
        "maxiter": 128,
    },
    acq_config={
        "name": "EI",
        "objective_config": {
            "direction": "maximize",
        },
    },
    opt_config={
        "q": 2,
        "num_restarts": 10,
        "raw_samples": 256,
    },
)
```

次の設定は辞書から対応する dataclass へ再帰的に変換されます。

| 入力 | 変換先 |
|---|---|
| `model_config` | `ModelConfig` |
| `input_transform_config` | `InputTransformConfig` |
| `multi_output_config` | `MultiOutputConfig` |
| `output_configs` | `OutputConfig` |
| `output_fit_configs` | `FitConfig` |
| `fit_config` | `FitConfig` |
| `acq_config` | `AcquisitionConfig` |
| `objective_config` | `ObjectiveConfig` |
| `opt_config` | `OptimizeConfig` |
| `repair_config` | `CandidateRepairConfig` |
| `data_context` | `DataContext` |
| `multi_objective` | `MultiObjectiveConfig` |
| `early_stopping_config` | `EarlyStoppingConfig` |
| `generation_schedule` | `GenerationSchedule` / `GenerationStep` |

### 2.1 獲得関数名だけ指定する

`acq_config` には文字列も使用できます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
    acq_config="UCB",
)
```

または、辞書内で `acq_name` を使用できます。

```python
acq_config = {
    "acq_name": "UCB",
    "acqf_kwargs": {
        "beta": 2.0,
    },
}
```

### 2.2 objective の簡易フィールド

`objective_*` 形式のフィールドは、内部で `ObjectiveConfig` にまとめられます。

```python
acq_config = {
    "name": "EI",
    "objective_direction": "maximize",
    "objective_output": 0,
    "objective_weight": 1.0,
}
```

主な対応フィールドは次のとおりです。

```text
objective_mode
objective_output
objective_outputs
objective_specs
objective_directions
objective_weights
objective_eq_targets
objective_direction
objective_weight
objective_eq_target
objective_n_w
objective_risk_type
objective_alpha
objective_maximize
objective_aggregate_mean_when_no_risk
objective_allow_unexpanded
objective_utility_values
objective_ordinal_likelihood
```

### 2.3 FitConfig の別名

Tabular APIと同様に、次の別名を利用できます。

```python
fit_config = {
    "fit_method": "auto",
    "fit_optimizer_kwargs": {
        "options": {
            "maxiter": 128,
        },
    },
    "fit_beta": 0.5,
}
```

| 別名 | FitConfigフィールド |
|---|---|
| `fit_method` | `method` |
| `fit_optimizer_kwargs` | `optimizer_kwargs` |
| `fit_beta` | `beta` |

---

## 3. config dataclassも引き続き使用可能

既存の dataclass API もそのまま利用できます。

```python
from bochan.api import (
    AcquisitionConfig,
    BochanStudy,
    DataContext,
    FitConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizeConfig,
)

study = BochanStudy(
    model_config=ModelConfig(
        task_type="regression",
        model_type="base",
    ),
    fit_config=FitConfig(maxiter=128),
    acq_config=AcquisitionConfig(
        name="EI",
        objective_config=ObjectiveConfig(
            direction="maximize",
        ),
    ),
    opt_config=OptimizeConfig(
        q=2,
        num_restarts=10,
        raw_samples=256,
    ),
    data_context=DataContext(bounds=bounds),
    bounds=bounds,
    n_initial_random=10,
)
```

辞書と dataclass は config ごとに混在させることもできます。

---

## 4. タスク別のデフォルト

### 4.1 単目的回帰

`model_config` を省略した場合は、単目的回帰として扱われます。

```python
model_config = {
    "task_type": "regression",
    "model_type": "base",
}
```

獲得関数の既定値は `EI` です。

### 4.2 多目的最適化

`task_type="multi_objective"` を明示し、`acq_config` を省略すると、獲得関数には `NEHVI` が選択されます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
    model_config={
        "task_type": "multi_objective",
        "model_type": "base",
    },
)

study.optimize(
    objective_func=lambda X: torch.stack(
        [X.sum(dim=-1), -((X - 0.5) ** 2).sum(dim=-1)],
        dim=-1,
    ),
    n_trials=30,
    q=3,
)
```

必要に応じて多目的設定を明示できます。

```python
data_context = {
    "multi_objective": {
        "ref_point": torch.tensor(
            [0.0, -1.0],
            dtype=torch.double,
        ),
    },
}
```

`ref_point`、`Y_baseline`、`partitioning`、`objective_thresholds`、`constraints` などは `DataContext` または `MultiObjectiveConfig` から渡せます。

### 4.3 分類・ordinal・hybrid

目的関数の戻り値だけから task type を安全に推定することはできないため、回帰以外では `model_config` を明示してください。

```python
model_config = {
    "task_type": "binary",
    "model_type": "base",
    "model_kwargs": {
        "num_inducing_points": 64,
    },
}
```

```python
model_config = {
    "task_type": "ordinal",
    "model_type": "base",
    "model_kwargs": {
        "num_classes": 3,
    },
}
```

---

## 5. mixed input

カテゴリ列がある場合は `cat_dims` に列indexを指定します。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
    model_config={
        "task_type": "regression",
        "model_type": "base",
        "cat_dims": [2],
    },
    opt_config={
        "q": 3,
        "num_restarts": 10,
        "raw_samples": 256,
        "fixed_features_list": [
            {2: 0.0},
            {2: 1.0},
            {2: 2.0},
        ],
    },
)
```

`cat_dims` がある場合、`BayesianOptimizer` 側で mixed 対応の optimizer に解決されます。

---

## 6. ask / tell

Python関数を `optimize()` に渡す代わりに、候補生成と結果登録を分離できます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
)

batch = study.ask(
    q=3,
    return_batch=True,
)

print(batch.candidates)
print(batch.trial_ids)

# 実験、外部シミュレーション、Web画面などで評価する
measured_values = batch.candidates.sum(dim=-1)

study.tell(batch, measured_values)
```

非同期実験では、候補生成時に `mark_running=True` を使用できます。

```python
batch = study.ask(
    q=5,
    mark_running=True,
    return_batch=True,
)
```

失敗した trial は次のように記録できます。

```python
study.mark_failed(
    batch.trial_ids[:1],
    reason="simulation crashed",
)
```

---

## 7. 呼び出しごとに設定を上書きする

`ask()` と `optimize()` の `acq_config` / `opt_config` も辞書形式を受け取ります。

```python
batch = study.ask(
    q=3,
    acq_config={
        "name": "UCB",
        "acqf_kwargs": {
            "beta": 4.0,
        },
    },
    opt_config={
        "q": 3,
        "num_restarts": 20,
        "raw_samples": 512,
    },
    return_batch=True,
)
```

呼び出し時の明示設定は、study作成時の設定や generation step の設定より優先されます。

---

## 8. Early stopping

`early_stopping_config` も辞書形式で指定できます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=5,
    early_stopping_config={
        "output_index": 0,
        "direction": "maximize",
        "target": 0.90,
        "target_mode": "ge",
        "target_patience": 2,
        "min_completed_trials": 5,
    },
)

study.optimize(
    objective_func,
    n_trials=100,
    q=3,
)

if study.stop_decision and study.stop_decision.should_stop:
    print(study.stop_decision.reason)
    print(study.stop_decision.details)
```

### 8.1 目標値以上で停止

```python
early_stopping_config = {
    "direction": "maximize",
    "target": 0.90,
    "target_mode": "ge",
    "target_patience": 2,
}
```

### 8.2 目標値以下で停止

```python
early_stopping_config = {
    "direction": "minimize",
    "target": 0.05,
    "target_mode": "le",
    "target_patience": 1,
}
```

### 8.3 目標値との差で停止

```python
early_stopping_config = {
    "direction": "minimize",
    "target": 1.50,
    "target_mode": "abs_diff_le",
    "target_tolerance": 0.02,
    "target_patience": 3,
}
```

### 8.4 改善が見られない場合に停止

```python
early_stopping_config = {
    "direction": "maximize",
    "no_improvement_patience": 5,
    "min_delta": 0.01,
    "min_completed_trials": 10,
}
```

---

## 9. Generation schedule

進行状況に応じて `q`、獲得関数、optimizer設定を切り替えられます。schedule全体と各stepを辞書で指定できます。

```python
study = BochanStudy(
    bounds=bounds,
    n_initial_random=5,
    generation_schedule={
        "steps": [
            {
                "name": "explore",
                "num_trials": 20,
                "q": 5,
                "acq_config": {
                    "name": "UCB",
                    "acqf_kwargs": {
                        "beta": 4.0,
                    },
                },
                "opt_config": {
                    "q": 5,
                    "num_restarts": 20,
                    "raw_samples": 512,
                },
            },
            {
                "name": "exploit",
                "q": 1,
                "acq_config": "EI",
                "opt_config": {
                    "q": 1,
                    "num_restarts": 10,
                    "raw_samples": 256,
                },
            },
        ],
    },
)

study.optimize(
    objective_func,
    n_trials=50,
)
```

`num_trials` はstep内で使用するtrial数、`until_completed` は累積完了trial数による切り替え条件です。

```python
generation_schedule = {
    "steps": [
        {
            "name": "initial",
            "until_completed": 10,
            "q": 5,
        },
        {
            "name": "middle",
            "until_completed": 30,
            "q": 3,
        },
        {
            "name": "final",
            "q": 1,
        },
    ],
}
```

使用されたstep名はtrial metadataに保存されます。

```python
history = study.trials_dataframe()
```

---

## 10. 保存と再開

```python
study.save("study.json")
```

trial履歴はJSONへ保存されます。

```python
study = BochanStudy.load(
    "study.json",
    bounds=bounds,
    n_initial_random=10,
)
```

標準の単目的回帰設定であれば、再開時にも内部デフォルトを使用できます。

カスタム設定を使用していた場合は、同じ設定を辞書または dataclass で再注入してください。

```python
study = BochanStudy.load(
    "study.json",
    bounds=bounds,
    model_config={
        "task_type": "regression",
        "model_type": "deepkernel",
        "model_kwargs": {
            "feature_extractor": feature_extractor,
            "feature_dim": 8,
        },
    },
    fit_config={
        "num_epochs": 300,
        "lr": 0.01,
    },
    acq_config={
        "name": "UCB",
        "acqf_kwargs": {
            "beta": 2.0,
        },
    },
    opt_config={
        "q": 3,
        "num_restarts": 10,
        "raw_samples": 256,
    },
)
```

`ModelConfig` などには callable や実行時オブジェクトが含まれる場合があるため、`save()` はtrial履歴を中心に保存します。モデルクラス、feature extractor、カスタムfactoryなどは再開時に再注入してください。

---

## 11. Trial の状態

| 状態 | 意味 |
|---|---|
| `CANDIDATE` | 候補として生成済み |
| `RUNNING` | 実験中・シミュレーション中 |
| `COMPLETED` | 結果登録済み |
| `FAILED` | 実験失敗・計算失敗 |

---

## 12. 主なメソッド

| メソッド | 役割 |
|---|---|
| `add_observations(X, Y)` | 既存データを `COMPLETED` trial として登録 |
| `ask(q)` | 次の候補点を生成 |
| `tell(batch, values)` | 候補点の評価結果を登録 |
| `optimize(objective_func, n_trials, q)` | Python関数を使って自動ループ |
| `check_early_stop()` | early stopping 条件を手動で判定 |
| `reset_early_stopping_state()` | early stopping の内部状態を初期化 |
| `current_generation_step()` | 現在のschedule stepを確認 |
| `completed_data()` | `COMPLETED` trialから `train_X`, `train_Y` を作成 |
| `pending_data()` | `CANDIDATE` / `RUNNING` から `X_pending` を作成 |
| `trials_dataframe()` | trial履歴を表形式へ変換 |
| `save(path)` | trial履歴をJSON保存 |
| `load(path, ...)` | trial履歴をJSONから復元 |

---

## 13. 既存 API との関係

`BochanStudy` は、次の低レベル API を置き換えるものではありません。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(
    acqf,
    bounds,
    opt_config,
)
```

この処理を `BayesianOptimizer` 経由で候補生成の1 stepとして利用し、trial管理、ask/tell、保存、early stopping、generation scheduleを上位レイヤーとして追加しています。
