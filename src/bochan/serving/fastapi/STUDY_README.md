# BochanStudy FastAPI

`BochanStudy` は `/api/v1/studies` 以下の stateful API から利用できます。HTTP では任意の Python callable を安全に受け取れないため、`optimize(objective_func=...)` を直接公開するのではなく、実験・シミュレーション連携に適した `create -> ask -> tell` を公開します。

## Endpoint

| method | path | 内容 |
|---|---|---|
| `POST` | `/api/v1/studies` | Study作成。初期観測も任意で登録 |
| `POST` | `/api/v1/studies/restore` | JSON snapshotから新しいStudyを復元 |
| `GET` | `/api/v1/studies` | Study ID一覧 |
| `GET` | `/api/v1/studies/{study_id}` | 件数、設定種別、停止判定などの概要 |
| `POST` | `/api/v1/studies/{study_id}/observations` | 既存観測を追加 |
| `POST` | `/api/v1/studies/{study_id}/ask` | 候補点とtrial IDを生成 |
| `POST` | `/api/v1/studies/{study_id}/tell` | trial IDに評価値を登録 |
| `POST` | `/api/v1/studies/{study_id}/trials/running` | trialを実行中へ変更 |
| `POST` | `/api/v1/studies/{study_id}/trials/failed` | trialを失敗へ変更 |
| `GET` | `/api/v1/studies/{study_id}/trials` | trial履歴 |
| `GET` | `/api/v1/studies/{study_id}/best` | 指定出力のbest情報 |
| `GET` | `/api/v1/studies/{study_id}/history` | 観測値とbest-so-farの可視化用レコード |
| `POST` | `/api/v1/studies/{study_id}/pareto` | Pareto判定済みレコード |
| `GET` | `/api/v1/studies/{study_id}/snapshot` | JSON保存用snapshot |
| `DELETE` | `/api/v1/studies/{study_id}` | インメモリStudyを削除 |

## 1. Studyを作成

configはPython APIと同じく辞書形式で指定できます。単目的回帰ではconfigを省略できます。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/studies \
  -H "Content-Type: application/json" \
  -d '{
    "bounds": [[0.0, 0.0], [1.0, 1.0]],
    "n_initial_random": 10,
    "metadata": {
      "feature_names": ["temperature", "pressure"]
    }
  }'
```

既存データを同時に登録する場合です。

```json
{
  "bounds": [[0.0], [1.0]],
  "initial_X": [[0.0], [0.5], [1.0]],
  "initial_Y": [[0.0], [0.25], [1.0]],
  "n_initial_random": 3
}
```

## 2. ask

```bash
curl -X POST http://127.0.0.1:8000/api/v1/studies/STUDY_ID/ask \
  -H "Content-Type: application/json" \
  -d '{
    "q": 3,
    "mark_running": true
  }'
```

レスポンスには候補点と対応する`trial_ids`が含まれます。

```json
{
  "study_id": "...",
  "trial_ids": [0, 1, 2],
  "candidates": [[0.1, 0.8], [0.4, 0.2], [0.7, 0.5]],
  "acq_value": null
}
```

獲得関数と最適化設定は呼び出しごとに上書きできます。

```json
{
  "q": 2,
  "acquisition_config": {
    "name": "UCB",
    "acqf_kwargs": {"beta": 4.0}
  },
  "optimize_config": {
    "q": 2,
    "num_restarts": 20,
    "raw_samples": 512
  }
}
```

## 3. tell

```bash
curl -X POST http://127.0.0.1:8000/api/v1/studies/STUDY_ID/tell \
  -H "Content-Type: application/json" \
  -d '{
    "trial_ids": [0, 1, 2],
    "values": [0.40, 0.65, 0.52],
    "metadata": [
      {"cycle": 0},
      {"cycle": 0},
      {"cycle": 0}
    ]
  }'
```

`tell`後は設定済みのearly stoppingも更新されます。判定結果はStudy概要の`stop_decision`に含まれます。

## 4. bestと履歴

```text
GET /api/v1/studies/STUDY_ID/best
GET /api/v1/studies/STUDY_ID/best?output_index=1&direction=minimize
GET /api/v1/studies/STUDY_ID/history?output_index=0&direction=maximize
```

`best`には`trial_id`、`value`、`values`、`x`、`params`、`direction`、`metadata`が含まれます。`history`はフロントエンドで折れ線グラフを作れるよう、次のレコードを返します。

```json
{
  "trial_id": 4,
  "order": 4,
  "cycle": 2,
  "value": 0.81,
  "best_value": 0.81,
  "is_best": true
}
```

## 5. 多目的とPareto

```bash
curl -X POST http://127.0.0.1:8000/api/v1/studies/STUDY_ID/pareto \
  -H "Content-Type: application/json" \
  -d '{
    "output_indices": [0, 1],
    "directions": ["maximize", "minimize"]
  }'
```

レスポンスの`pareto_trials`が非劣解です。`trials`には全ての有限な完了trialと`is_pareto`が含まれるため、散布図とPareto frontの描画に利用できます。

## 6. snapshotと復元

サーバー上の任意ファイルパスは受け取らず、JSON snapshotとして保存・復元します。

```text
GET /api/v1/studies/STUDY_ID/snapshot
POST /api/v1/studies/restore
```

復元時にはcallableやカスタムmodel factoryをsnapshotだけから再構築できないため、必要な`model_config`、`fit_config`、`acquisition_config`、`optimize_config`を再注入してください。

```json
{
  "snapshot": {"schema_version": 1, "next_trial_id": 10, "trials": [], "metadata": {}},
  "bounds": [[0.0], [1.0]],
  "model_config": {
    "task_type": "regression",
    "model_type": "deepkernel"
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.01
  }
}
```

## Storeについて

既定の`InMemoryStudyStore`はプロセスローカルです。Study単位のlockで`ask`、`tell`、状態変更を直列化しますが、サーバー再起動で内容は失われます。実運用では`get_study_store()`をDBやobject storageを利用する実装へdependency overrideしてください。
