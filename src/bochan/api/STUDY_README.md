# BochanStudy: ask/tell と optimize ループ

`BochanStudy` は、既存の `BayesianOptimizer` を 1 step の候補点生成エンジンとして使い、その上に Optuna / Ax 風の最適化ループを載せるための上位 API です。

主な目的は次の2つです。

1. Python 関数を自動で評価する
2. 実験・Web アプリ・外部シミュレーションのように、人間や外部処理を挟んで続きを回す

内部設計としては、`ask/tell` を中核にしています。`optimize()` は `ask/tell` を Python 関数評価用に包んだ便利関数です。

---

## 1. Python 関数を自動で回す

```python
import torch

from bochan.api import BochanStudy

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

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

この例では、初期候補は `bounds` からランダム生成されます。`model_config`, `acq_config`, `opt_config` を渡すと、初期点の後に既存 API の `BayesianOptimizer` を使った BO 候補生成へ切り替わります。

---

## 2. config の最小設定例

`BochanStudy` は既存の `ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig`, `DataContext` をそのまま受け取ります。

### 2.1 単目的 regression + EI

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BochanStudy,
    DataContext,
    FitConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizeConfig,
)

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    outcome_transform=True,
)

fit_config = FitConfig(
    maxiter=128,
)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        direction="maximize",
    ),
)

opt_config = OptimizeConfig(
    q=1,
    num_restarts=10,
    raw_samples=256,
)

data_context = DataContext(
    bounds=bounds,
)

study = BochanStudy(
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
    bounds=bounds,
    n_initial_random=5,
)
```

`EI` の `best_f` などは `DataContext` や `acqf_kwargs` で明示的に渡せます。

```python
acq_config = AcquisitionConfig(
    name="EI",
    acqf_kwargs={"best_f": train_Y.max()},
)
```

### 2.2 batch 候補 + qUCB

```python
acq_config = AcquisitionConfig(
    name="UCB",
    acqf_kwargs={"beta": 2.0},
)

opt_config = OptimizeConfig(
    q=3,
    num_restarts=20,
    raw_samples=512,
    sequential=False,
)

study = BochanStudy(
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    bounds=bounds,
    n_initial_random=10,
)
```

### 2.3 mixed model

カテゴリ列がある場合は、`ModelConfig.cat_dims` に列 index を渡します。

```python
model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    cat_dims=[2],
)

opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
)
```

`cat_dims` がある場合、`BayesianOptimizer` 側で mixed optimizer へ解決されます。

### 2.4 多目的 EHVI / NEHVI の雛形

```python
from bochan.api import MultiObjectiveConfig

model_config = ModelConfig(
    task_type="multi_objective",
    model_type="base",
    outcome_transform=True,
)

data_context = DataContext(
    bounds=bounds,
    multi_objective=MultiObjectiveConfig(
        ref_point=torch.tensor([0.0, 0.0], dtype=torch.double),
    ),
)

acq_config = AcquisitionConfig(
    name="NEHVI",
)

opt_config = OptimizeConfig(
    q=3,
    num_restarts=10,
    raw_samples=256,
)

study = BochanStudy(
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
    bounds=bounds,
    n_initial_random=10,
)
```

`ref_point`, `Y_baseline`, `partitioning`, `constraints` などは `MultiObjectiveConfig` または `DataContext` から渡します。

### 2.5 保存・再開時の config 再注入

`save()` は trial 履歴を中心に JSON 保存します。`ModelConfig` などには callable や実行時オブジェクトが含まれる場合があるため、再開時は config を再注入します。

```python
study.save("study.json")

study = BochanStudy.load(
    "study.json",
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
    bounds=bounds,
)
```

---

## 3. 実験・Web アプリ・シミュレーションで使う

```python
batch = study.ask(q=3, return_batch=True)

# batch.candidates を実験、Web 画面、外部シミュレーションなどへ渡す
# 後で結果が返ってきたら登録する

study.tell(batch, measured_values)
study.save("study.json")
```

後日、続きを行う場合は次のように復元します。

```python
study = BochanStudy.load(
    "study.json",
    model_config=model_config,
    acq_config=acq_config,
    opt_config=opt_config,
    fit_config=fit_config,
    bounds=bounds,
)

next_batch = study.ask(q=3, return_batch=True)
```

`ModelConfig` などには callable や実行時オブジェクトが含まれることがあるため、`save()` は trial 履歴を中心に保存します。再開時は必要な設定を `load()` に再注入してください。

---

## 4. Trial の状態管理

`BochanStudy` は trial ごとに状態を持ちます。

| 状態 | 意味 |
|---|---|
| `CANDIDATE` | 候補として生成済み |
| `RUNNING` | 実験中・シミュレーション中 |
| `COMPLETED` | 結果登録済み |
| `FAILED` | 実験失敗・計算失敗 |

非同期実験では、候補生成時に `mark_running=True` を使うと便利です。

```python
batch = study.ask(q=5, mark_running=True, return_batch=True)
```

失敗した trial は次のように記録できます。

```python
study.mark_failed(batch.trial_ids[:1], reason="simulation crashed")
```

---

## 5. 既存 API との関係

`BochanStudy` は既存の4段階 API を置き換えません。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

この流れを内部で `BayesianOptimizer` 経由で呼び出します。

そのため、既存の `ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig`, `DataContext` をそのまま利用できます。

---

## 6. 主なメソッド

| メソッド | 役割 |
|---|---|
| `add_observations(X, Y)` | 既存データを `COMPLETED` trial として登録 |
| `ask(q)` | 次の候補点を生成 |
| `tell(batch, values)` | 候補点の評価結果を登録 |
| `optimize(objective_func, n_trials, q)` | Python 関数を使って自動ループ |
| `completed_data()` | `COMPLETED` trial から `train_X`, `train_Y` を作る |
| `pending_data()` | `CANDIDATE` / `RUNNING` から `X_pending` を作る |
| `save(path)` | trial 履歴を JSON 保存 |
| `load(path, ...)` | trial 履歴を JSON から復元 |
