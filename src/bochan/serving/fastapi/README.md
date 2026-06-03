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

## 2. ディレクトリ構成

```text
src/bochan/serving/
  __init__.py
  fastapi/
    __init__.py
    app.py
    dependencies.py
    converters.py
    README.md
    schemas/
      __init__.py
      configs.py
      requests.py
      responses.py
    routers/
      __init__.py
      health.py
      models.py
      predictions.py
      candidates.py
      acquisitions.py
```

| ファイル | 役割 |
|---|---|
| `app.py` | FastAPI app factory。router を束ねる |
| `dependencies.py` | FastAPI dependency。現在はインメモリ optimizer store を提供 |
| `converters.py` | Pydantic schema / JSON-like object から `bochan.api` dataclass や torch tensor へ変換 |
| `schemas/configs.py` | `ModelConfig`, `ObjectiveConfig`, `OptimizeConfig` などに対応する Pydantic schema |
| `schemas/requests.py` | HTTP request body schema |
| `schemas/responses.py` | HTTP response schema |
| `routers/models.py` | model fit / list / delete endpoint |
| `routers/predictions.py` | prediction endpoint |
| `routers/candidates.py` | candidate generation endpoint |
| `routers/acquisitions.py` | acquisition registry endpoint |

---

## 3. インストール

FastAPI 関連は optional dependency です。

```bash
pip install -e ".[api]"
```

`pyproject.toml` の `api` extra には次が含まれます。

```toml
[project.optional-dependencies]
api = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "pydantic>=2",
]
```

---

## 4. 起動方法

```bash
uvicorn bochan.serving.fastapi.app:app --reload
```

Python から app を作成する場合です。

```python
from bochan.serving.fastapi import create_app

app = create_app(title="bochan Optimization API")
```

---

## 5. endpoint 一覧

| method | path | 内容 |
|---|---|---|
| `GET` | `/health` | ヘルスチェック |
| `POST` | `/models` | モデル作成・学習 |
| `GET` | `/models` | インメモリ store 内の model id 一覧 |
| `DELETE` | `/models/{model_id}` | model id を削除 |
| `POST` | `/models/{model_id}/predict` | 予測 |
| `POST` | `/models/{model_id}/candidates` | 獲得関数生成 + 候補点最適化 |
| `GET` | `/acquisitions/names` | 利用可能な acquisition alias 一覧 |

現在の store はプロセス内インメモリです。サーバー再起動で fitted model は失われます。実運用では `dependencies.py` の `get_optimizer_store()` を差し替え、モデル artifact や metadata を DB / object storage / model registry に保存してください。

---

## 6. 全体フロー

```text
POST /models
  -> JSON request
  -> Pydantic schema
  -> converters.py
  -> ModelConfig / FitConfig / DataContext / torch.Tensor
  -> BayesianOptimizer.fit(...)
  -> model_id を返す

POST /models/{model_id}/candidates
  -> model_id から fitted optimizer を取得
  -> AcquisitionConfig / ObjectiveConfig / OptimizeConfig へ変換
  -> BayesianOptimizer.candidate(...)
  -> candidates / acq_value を JSON で返す
```

---

## 7. モデル学習リクエスト

### 7.1 single-output regression

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

レスポンス例です。

```json
{
  "model_id": "e2f1...",
  "task_type": "regression",
  "model_type": "base",
  "n_train": 3,
  "metadata": {
    "model_cls": "SingleTaskGP"
  }
}
```

### 7.2 input perturbation あり

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "input_transform_config": {
      "perturbation": true,
      "n_w": 8,
      "std": 0.1
    }
  },
  "fit_config": {
    "maxiter": 128
  },
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0], [0.25], [1.0]],
  "bounds": [[0.0], [1.0]]
}
```

`InputTransformConfig(n_w=8)` を使う場合、候補点生成時の `ObjectiveConfig(n_w=8)` も同じ値にしてください。

---

## 8. 予測リクエスト

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

`return_type` は HTTP API では `mean`, `variance`, `mean_variance` を想定しています。`posterior` はそのまま JSON 化できないため、HTTP API では標準レスポンスに含めていません。

---

## 9. 候補点生成リクエスト

### 9.1 EI

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

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/candidates \
  -H "Content-Type: application/json" \
  -d '{
    "acq_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
    "opt_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
  }'
```

### 9.2 input perturbation + CVaR EI

```json
{
  "acq_config": {
    "name": "EI",
    "objective_config": {
      "mode": "scalar",
      "output": 0,
      "direction": "maximize",
      "n_w": 8,
      "risk_type": "cvar",
      "alpha": 0.8
    },
    "acqf_kwargs": {
      "best_f": 1.0
    }
  },
  "opt_config": {
    "q": 3,
    "num_restarts": 20,
    "raw_samples": 512
  }
}
```

---

## 10. `ObjectiveConfig` の JSON 指定

FastAPI では Python class を直接指定できないため、通常は `objective_config` を JSON で渡します。サーバー側で `bochan.api.ObjectiveConfig` に変換され、`factory.py` の `build_objective(...)` が task type に応じて実 objective を生成します。

HTTP API の標準 schema では、JSON 化できない `objective` / `objective_factory` は扱いません。通常は `objective_config` を使ってください。

### 10.1 regression multi-output で特定出力だけ使う

```json
{
  "acq_config": {
    "name": "EI",
    "objective_config": {
      "mode": "scalar",
      "output": 1,
      "direction": "maximize",
      "n_w": 8,
      "risk_type": null
    },
    "acqf_kwargs": {
      "best_f": 1.23
    }
  },
  "opt_config": {
    "q": 1
  }
}
```

`best_f` は対象出力、ここでは `train_Y[:, 1]` に合わせてください。

### 10.2 regression multi-output で一部出力だけ多目的に使う

```json
{
  "acq_config": {
    "name": "NEHVI",
    "objective_config": {
      "mode": "multi_output",
      "outputs": [0, 2],
      "directions": ["maximize", "minimize"],
      "weights": [1.0, 1.0]
    }
  },
  "data_context": {
    "X_baseline": [[0.0], [0.5], [1.0]],
    "Y_baseline": [[0.0, 1.0], [0.25, 0.8], [1.0, 0.2]],
    "ref_point": [0.0, 0.0]
  },
  "opt_config": {
    "q": 3
  }
}
```

`outputs=[0, 2]` のように一部出力を選ぶ場合、`Y_baseline` と `ref_point` も選択後の出力次元に合わせてください。

### 10.3 ordinal expected utility

```json
{
  "acq_config": {
    "name": "EI",
    "objective_config": {
      "mode": "scalar",
      "utility_values": [0.0, 1.0, 2.0],
      "n_w": 8,
      "risk_type": "cvar",
      "alpha": 0.8
    },
    "acqf_kwargs": {
      "best_f": 2.0
    }
  },
  "opt_config": {
    "q": 3
  }
}
```

---

## 11. hybrid model の JSON 指定

### 11.1 hybrid model config

```json
{
  "model_config": {
    "task_type": "hybrid",
    "model_type": "base",
    "outcome_transform": true,
    "multi_output_config": {
      "use_hybrid": true,
      "output_configs": [
        {"task_type": "regression", "model_type": "base", "name": "strength"},
        {"task_type": "binary", "model_type": "base", "name": "defect_prob"}
      ]
    }
  },
  "fit_config": {
    "maxiter": 128
  },
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0, 0.0], [0.25, 1.0], [1.0, 1.0]],
  "bounds": [[0.0], [1.0]]
}
```

`outcome_transform=true` は hybrid wrapper には渡されず、regression submodel にだけ適用されます。binary / ordinal submodel では無効化されます。

### 11.2 hybrid scalar objective

```json
{
  "acq_config": {
    "name": "EI",
    "objective_config": {
      "mode": "scalar",
      "output": "strength",
      "direction": "maximize",
      "n_w": 8,
      "risk_type": null
    },
    "acqf_kwargs": {
      "best_f": 1.0
    }
  },
  "opt_config": {
    "q": 1
  }
}
```

### 11.3 hybrid multi-output objective

```json
{
  "acq_config": {
    "name": "NEHVI",
    "objective_config": {
      "mode": "multi_output",
      "outputs": ["strength", "defect_prob"],
      "directions": ["maximize", "minimize"],
      "weights": [1.0, 0.5],
      "n_w": 8,
      "risk_type": "cvar",
      "alpha": 0.8
    }
  },
  "data_context": {
    "X_baseline": [[0.0], [0.5], [1.0]],
    "Y_baseline": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
    "ref_point": [0.0, 0.0]
  },
  "opt_config": {
    "q": 3
  }
}
```

---

## 12. schema と converter の考え方

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

Python callable は JSON では表現できないため、HTTP schema では `model_cls`, `model_factory`, `acqf_cls`, `acqf_factory`, `objective`, `objective_factory` は標準扱いしません。これらが必要な場合は Python API を使うか、サーバー側に安全な registry / allow-list を実装してください。

---

## 13. store の差し替え

現在の `InMemoryOptimizerStore` は次の制約があります。

- サーバー再起動で fitted model が失われる
- 複数 worker / 複数 replica で共有されない
- 長期運用ではメモリ管理が必要

本番利用では `dependencies.py` の `get_optimizer_store()` を差し替えてください。

---

## 14. 今後の拡張候補

- model artifact の保存 / 読み込み endpoint
- job queue による非同期 fit / candidate generation
- registry 方式による安全な custom model / acquisition / objective の指定
- 認証・認可
- request id / experiment id / run id の導入
- model metadata と training data の永続化
- OpenAPI schema の分割と versioning
