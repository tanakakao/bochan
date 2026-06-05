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

## 2. 実験・Web アプリ・シミュレーションで使う

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

## 3. Trial の状態管理

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

## 4. 既存 API との関係

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

## 5. 主なメソッド

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
