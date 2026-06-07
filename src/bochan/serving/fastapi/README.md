# bochan FastAPI serving

`bochan.serving.fastapi` は、`bochan.api` の Python API を HTTP / JSON 経由で利用するための serving 層です。

`bochan.api` はモデル構築・学習・獲得関数生成・候補点最適化の中核ロジックを持ちます。一方、FastAPI 層は Pydantic schema、JSON 変換、endpoint、インメモリ store などの transport 層を担当します。

---

## 1. 設計方針

依存方向は次のようにします。

```text
bochan.serving.fastapi
  -> bochan.api
  -> bochan.models / bochan.acquisition / bochan.optim
```

`bochan.api` 側は FastAPI / Pydantic に依存しません。FastAPI を使わない Python API 利用者が余計な依存を import しないようにするためです。

---

## 2. インストールと起動

FastAPI 関連は optional dependency です。

```bash
pip install -e ".[api]"
```

```bash
uvicorn bochan.serving.fastapi.app:app --reload
```

Python から app を作成する場合です。

```python
from bochan.serving.fastapi import create_app

app = create_app(title="bochan Optimization API")
```

---

## 3. endpoint 一覧

| method | path | 内容 |
|---|---|---|
| `GET` | `/health` | ヘルスチェック |
| `POST` | `/models` | モデル作成・学習 |
| `GET` | `/models` | インメモリ store 内の model id 一覧 |
| `DELETE` | `/models/{model_id}` | model id を削除 |
| `POST` | `/models/{model_id}/predict` | 予測 |
| `POST` | `/models/{model_id}/candidates` | 獲得関数生成 + 候補点最適化 |
| `POST` | `/models/{model_id}/ask` | ask-and-tell 用の候補点生成 |
| `POST` | `/models/{model_id}/tell` | 新規観測追加と任意の再学習 |
| `POST` | `/models/{model_id}/refit` | 既存データで再学習 |
| `POST` | `/models/{model_id}/candidates/compare` | 複数獲得関数を比較 |
| `GET` | `/acquisitions/names` | 利用可能な acquisition alias 一覧 |

現在の store はプロセス内インメモリです。サーバー再起動で fitted model は失われます。実運用では `dependencies.py` の `get_optimizer_store()` を差し替え、モデル artifact や metadata を DB / object storage / model registry に保存してください。

---

## 4. 全体フロー

```text
POST /models
  -> JSON request
  -> Pydantic schema
  -> converters.py
  -> ModelConfig / FitConfig / DataContext / torch.Tensor
  -> BayesianOptimizer.fit(...)
  -> model_id を返す

POST /models/{model_id}/candidates または /ask
  -> model_id から fitted optimizer を取得
  -> AcquisitionConfig / ObjectiveConfig / OptimizeConfig へ変換
  -> BayesianOptimizer.candidate(...) または ask(...)
  -> candidates / acq_value を JSON で返す

POST /models/{model_id}/tell
  -> new_X / new_Y を追加
  -> refit=true なら BayesianOptimizer.tell(..., refit=True)
  -> 更新後 metadata を返す
```

---

## 5. モデル学習リクエスト

### 5.1 single-output regression

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "outcome_transform": true
  },
  "fit_config": {
    "maxiter": 128
  },
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0], [0.25], [1.0]],
  "bounds": [[0.0], [1.0]]
}
```

### 5.2 single-output multiclass

`task_type="multiclass"`、`model_type="base"` を指定します。`model_kwargs.num_classes` は明示するのが安全です。

```json
{
  "model_config": {
    "task_type": "multiclass",
    "model_type": "base",
    "model_kwargs": {
      "num_classes": 3,
      "num_inducing_points": 32
    }
  },
  "fit_config": {
    "num_epochs": 250,
    "lr": 0.03
  },
  "train_X": [[0.10, 0.20], [0.80, 0.25], [0.50, 0.85], [0.30, 0.70]],
  "train_Y": [0, 1, 2, 2],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```

レスポンス例です。

```json
{
  "model_id": "e2f1...",
  "task_type": "multiclass",
  "model_type": "base",
  "n_train": 4,
  "metadata": {
    "model_cls": "MulticlassClassificationGPModel"
  }
}
```

### 5.3 mixed-input multiclass

`cat_dims` を指定すると Python API 側で `input_type="mixed"` に解決されます。

```json
{
  "model_config": {
    "task_type": "multiclass",
    "model_type": "base",
    "cat_dims": [2],
    "model_kwargs": {
      "num_classes": 3,
      "num_inducing_points": 32
    }
  },
  "fit_config": {
    "num_epochs": 250,
    "lr": 0.03
  },
  "train_X": [[0.10, 0.20, 0], [0.80, 0.25, 1], [0.50, 0.85, 2], [0.30, 0.70, 1]],
  "train_Y": [0, 1, 2, 2],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]
}
```

### 5.4 heteroscedastic multiclass

```json
{
  "model_config": {
    "task_type": "multiclass",
    "model_type": "hetero",
    "model_kwargs": {
      "num_classes": 3,
      "num_inducing_points": 32
    }
  },
  "fit_config": {
    "num_epochs": 250,
    "lr": 0.03
  },
  "train_X": [[0.10, 0.20], [0.80, 0.25], [0.50, 0.85], [0.30, 0.70]],
  "train_Y": [0, 1, 2, 2],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```

### 5.5 multi-output multiclass

`multi_output_config` を使います。多クラス multi-output は hybrid wrapper 経由で扱うため、`use_hybrid=true` を指定してください。

```json
{
  "model_config": {
    "task_type": "multiclass",
    "model_type": "base",
    "model_kwargs": {
      "num_classes": 3,
      "num_inducing_points": 32
    },
    "multi_output_config": {
      "use_hybrid": true,
      "output_task_types": ["multiclass", "multiclass"],
      "output_names": ["defect_a", "defect_b"]
    }
  },
  "fit_config": {
    "num_epochs": 200,
    "lr": 0.03
  },
  "train_X": [[0.10, 0.20], [0.80, 0.25], [0.50, 0.85], [0.30, 0.70]],
  "train_Y": [[0, 1], [1, 1], [2, 0], [2, 2]],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```

---

## 6. 予測リクエスト

```json
{
  "X": [[0.25], [0.75]],
  "return_type": "mean_variance"
}
```

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[0.25], [0.75]], "return_type": "mean_variance"}'
```

レスポンス例です。

```json
{
  "model_id": "e2f1...",
  "mean": [[0.12], [0.72]],
  "variance": [[0.01], [0.02]],
  "value": null
}
```

Multiclass の `mean` は class probability として扱います。shape は概ね `n x C` または `n x output x C` 系になります。

---

## 7. 候補点生成リクエスト

### 7.1 regression EI

```json
{
  "acq_config": {
    "name": "EI",
    "acqf_kwargs": {
      "best_f": 1.0
    }
  },
  "opt_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 256
  }
}
```

### 7.2 multiclass active learning: entropy

`task_type="multiclass"` の fitted model に対して `name="entropy"` を指定すると、single / multi / hetero の状態に応じて対応する multiclass acquisition に解決されます。

```json
{
  "acq_config": {
    "name": "entropy"
  },
  "opt_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128,
    "sequential": true
  }
}
```

他に次の alias が使えます。

```json
{"name": "BALD"}
{"name": "variance"}
{"name": "margin"}
{"name": "NIPV"}
```

### 7.3 multiclass level-set estimation

クラス2の確率が 0.5 付近の境界を探索する例です。

```json
{
  "acq_config": {
    "name": "straddle",
    "acqf_kwargs": {
      "target_class": 2,
      "threshold": 0.5
    }
  },
  "opt_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

利用できる alias 例です。

```json
{"name": "straddle"}
{"name": "ICU"}
{"name": "boundaryvariance"}
{"name": "classentropy"}
{"name": "poe"}
{"name": "levelset"}
```

### 7.4 multiclass Bayesian optimization: target class EI

多クラス BO では `target_class` が必須です。目的は `p(target_class | x)` の最大化です。

```json
{
  "acq_config": {
    "name": "EI",
    "acqf_kwargs": {
      "target_class": 2,
      "best_f": 0.70,
      "num_samples": 128
    }
  },
  "opt_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

UCB の例です。

```json
{
  "acq_config": {
    "name": "UCB",
    "acqf_kwargs": {
      "target_class": 2,
      "beta": 2.0,
      "num_samples": 128
    }
  },
  "opt_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

### 7.5 hetero multiclass acquisition

`model_type="hetero"` の multiclass model では、同じ alias が hetero 版に解決されます。ノイズ重み付けを変えたい場合は `acqf_kwargs` で指定します。

```json
{
  "acq_config": {
    "name": "entropy",
    "acqf_kwargs": {
      "noise_mode": "inverse_linear",
      "noise_combine": "multiply",
      "noise_penalty_lambda": 1.0
    }
  },
  "opt_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

### 7.6 multi-output multiclass acquisition

multi-output multiclass acquisition では `output_reduction` を指定できます。

```json
{
  "acq_config": {
    "name": "entropy",
    "acqf_kwargs": {
      "output_reduction": "weighted_mean",
      "output_weights": [0.7, 0.3]
    }
  },
  "opt_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

---

## 8. ask / tell / refit

### 8.1 ask

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "acq_config": {"name": "entropy"},
    "opt_config": {"q": 1, "num_restarts": 10, "raw_samples": 128}
  }'
```

### 8.2 tell

```json
{
  "new_X": [[0.42, 0.80]],
  "new_Y": [2],
  "refit": true,
  "fit_config": {
    "num_epochs": 150,
    "lr": 0.03
  }
}
```

### 8.3 refit

```json
{
  "fit_config": {
    "num_epochs": 150,
    "lr": 0.03
  }
}
```

---

## 9. acquisition comparison

```json
{
  "acq_configs": [
    {"name": "entropy"},
    {"name": "margin"},
    {"name": "EI", "acqf_kwargs": {"target_class": 2, "best_f": 0.70}}
  ],
  "opt_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/candidates/compare \
  -H "Content-Type: application/json" \
  -d '{
    "acq_configs": [
      {"name": "entropy"},
      {"name": "margin"},
      {"name": "EI", "acqf_kwargs": {"target_class": 2, "best_f": 0.70}}
    ],
    "opt_config": {"q": 1, "num_restarts": 10, "raw_samples": 128}
  }'
```

---

## 10. schema と converter の考え方

FastAPI 側では `ModelConfigSchema`, `ObjectiveConfigSchema`, `OptimizeConfigSchema` などの Pydantic schema を受け取ります。

`converters.py` が次の変換を担当します。

```text
ModelConfigSchema       -> bochan.api.ModelConfig
FitConfigSchema         -> bochan.api.FitConfig
ObjectiveConfigSchema   -> bochan.api.ObjectiveConfig
AcquisitionConfigSchema -> bochan.api.AcquisitionConfig
OptimizeConfigSchema    -> bochan.api.OptimizeConfig
DataContextSchema       -> bochan.api.DataContext
JSON numeric arrays     -> torch.Tensor
```

Multiclass でよく使う `target_class`, `threshold`, `num_samples`, `output_reduction`, `output_weights`, `noise_mode` などは `acqf_kwargs` にそのまま渡します。`output_weights`, `best_f`, `ref_point`, `X_baseline` など tensor 的に扱う値は converter 側で必要に応じて tensor 化されます。
