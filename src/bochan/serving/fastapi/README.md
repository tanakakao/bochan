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

FastAPI 側の request schema は、公開 API の dataclass にできるだけ近い名前・構造を受け取ります。例えば、HTTP payload の `model_config` は `bochan.api.ModelConfig` に、`fit_config` は `bochan.api.FitConfig` に、`acquisition_config` は `bochan.api.AcquisitionConfig` に変換されます。

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

`main.py` などに上記を置くと、次のように module path を指定して起動できます。

```bash
uvicorn main:app --reload
```

`create_app()` は FastAPI の application factory なので、Python コードから直接 import して設定を変えたり、テスト用 client に渡したりできます。`version` は OpenAPI に表示される API version です。

```python
from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app


def build_client() -> TestClient:
    """Create a test client for the bochan FastAPI app.

    Returns:
        TestClient connected to an in-process bochan FastAPI application.
    """
    app = create_app(title="bochan Optimization API", version="0.2.0")
    return TestClient(app)


client = build_client()
response = client.get("/api/v1/health")
response.raise_for_status()
print(response.json())
```

学習から予測までを Python だけで smoke test したい場合は、同じ `TestClient` で HTTP endpoint を呼び出せます。`create_app()` の既定では router が `/api/v1` に mount されるため、path には `/api/v1/models` のように prefix を付けます。

```python
from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app


def fit_regression_model(client: TestClient) -> str:
    """Fit a small regression model through the FastAPI layer.

    Args:
        client: Test client connected to a bochan FastAPI application.

    Returns:
        Model identifier returned by the in-memory optimizer store.
    """
    payload = {
        "model_config": {"task_type": "regression", "model_type": "base"},
        "fit_config": {"maxiter": 32},
        "train_X": [[0.0], [0.5], [1.0]],
        "train_Y": [[0.0], [0.25], [1.0]],
        "bounds": [[0.0], [1.0]],
    }
    response = client.post("/api/v1/models", json=payload)
    response.raise_for_status()
    return response.json()["model_id"]


def predict_mean_variance(client: TestClient, model_id: str) -> dict[str, object]:
    """Predict posterior mean and variance through the FastAPI layer.

    Args:
        client: Test client connected to a bochan FastAPI application.
        model_id: Identifier returned by the model fitting endpoint.

    Returns:
        JSON response containing mean and variance summaries.
    """
    response = client.post(
        f"/api/v1/models/{model_id}/predict",
        json={"X": [[0.25], [0.75]], "return_type": "mean_variance"},
    )
    response.raise_for_status()
    return response.json()


client = TestClient(create_app(title="bochan Optimization API"))
model_id = fit_regression_model(client)
prediction = predict_mean_variance(client, model_id)
print(prediction["mean"], prediction["variance"])
```

外部サーバーとして起動済みの app に対して Python から呼び出す場合は、通常の HTTP client を使います。

```python
import httpx


def check_health(base_url: str = "http://127.0.0.1:8000") -> dict[str, object]:
    """Call the bochan FastAPI health endpoint.

    Args:
        base_url: Base URL where the FastAPI server is running.

    Returns:
        Parsed JSON health response.
    """
    response = httpx.get(f"{base_url}/api/v1/health", timeout=10.0)
    response.raise_for_status()
    return response.json()
```

OpenAPI / Swagger UI は、通常 FastAPI の既定通り次で確認できます。

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

---

## 3. endpoint 一覧

| method | path | 内容 |
|---|---|---|
| `GET` | `/api/v1/health` | ヘルスチェック |
| `POST` | `/api/v1/models` | モデル作成・学習 |
| `GET` | `/api/v1/models` | インメモリ store 内の model id 一覧 |
| `DELETE` | `/api/v1/models/{model_id}` | model id を削除 |
| `POST` | `/api/v1/models/{model_id}/predict` | 予測 |
| `POST` | `/api/v1/models/{model_id}/candidates` | 獲得関数生成 + 候補点最適化 |
| `POST` | `/api/v1/models/{model_id}/ask` | ask-and-tell 用の候補点生成 |
| `POST` | `/api/v1/models/{model_id}/tell` | 新規観測追加と任意の再学習 |
| `POST` | `/api/v1/models/{model_id}/refit` | 既存データで再学習 |
| `POST` | `/api/v1/models/{model_id}/candidates/compare` | 複数獲得関数を比較 |
| `GET` | `/api/v1/acquisitions/names` | 利用可能な acquisition alias 一覧 |
| `POST` | `/api/v1/models/{model_id}/save` | インメモリモデルを保存 |
| `POST` | `/api/v1/models/load` | 保存モデルを読み込み新しい model id として登録 |
| `GET` | `/api/v1/artifacts` | 保存済みモデル一覧 |
| `POST` | `/api/v1/suggest` | ステートレス候補生成 |

現在の store はプロセス内インメモリです。サーバー再起動で fitted model は失われます。実運用では `dependencies.py` の `get_optimizer_store()` を差し替え、モデル artifact や metadata を DB / object storage / model registry に保存してください。

---

## 4. payload の基本ルール

### 4.1 API 風の名前と互換 alias

候補点生成 payload では、推奨名として `acquisition_config` / `optimize_config` を使います。

```json
{
  "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "optimize_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
}
```

既存コードとの互換のため、短い名前 `acq_config` / `opt_config` も受け取れます。

```json
{
  "acq_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "opt_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
}
```

複数 acquisition の比較では、推奨名は `acquisition_configs` です。互換名として `acq_configs` も使えます。

### 4.2 tensor_options

JSON の数値配列は converter で `torch.Tensor` に変換されます。dtype や device を明示したい場合は、各 request に `tensor_options` を指定します。

```json
{
  "tensor_options": {
    "dtype": "float64",
    "device": "cpu"
  }
}
```

対応する dtype 名の例です。

```text
float64 / double / torch.float64
float32 / float / torch.float32
int64 / long / torch.int64
```

通常の `train_X`, `train_Y`, `bounds`, `X`, `new_X`, `new_Y`, `X_baseline`, `best_f`, `ref_point`, `objective_thresholds`, `steps` などは、この設定に従って tensor 化されます。線形制約の `indices` だけは long tensor に変換されます。

---

## 5. 全体フロー

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

## 6. モデル学習リクエスト

### 6.1 single-output regression

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

curl 例です。

```bash
curl -X POST http://127.0.0.1:8000/models \
  -H "Content-Type: application/json" \
  -d '{
    "model_config": {"task_type": "regression", "model_type": "base"},
    "fit_config": {"maxiter": 128},
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]]
  }'
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

### 6.2 FitConfig.beta

`FitConfig.beta` は `mll_kwargs["beta"]` への便利 alias です。DeepGP / DeepKernel classifier などで ELBO の beta を変えたい場合に使います。

```json
{
  "model_config": {
    "task_type": "multiclass",
    "model_type": "deepgp",
    "model_kwargs": {
      "num_classes": 3,
      "num_inducing_points": 32
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.03,
    "beta": 0.5
  },
  "train_X": [[0.10, 0.20], [0.80, 0.25], [0.50, 0.85], [0.30, 0.70]],
  "train_Y": [0, 1, 2, 2],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```

`mll_kwargs` に明示的に `beta` を入れた場合は、そちらが優先されます。

```json
{
  "fit_config": {
    "beta": 0.5,
    "mll_kwargs": {"beta": 0.8}
  }
}
```

### 6.3 single-output multiclass

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

### 6.4 mixed-input multiclass

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

### 6.5 input_transform_config

Normalize と input perturbation は `model_config.input_transform_config` で指定します。

```json
{
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "input_transform_config": {
      "normalize": true,
      "perturbation": true,
      "n_w": 8,
      "std": 0.05,
      "categorical_idx": [2]
    }
  },
  "train_X": [[0.10, 0.20, 0], [0.80, 0.25, 1], [0.50, 0.85, 2]],
  "train_Y": [[0.1], [0.5], [0.9]],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]
}
```

`bounds` を `input_transform_config` の中に書くこともできます。その場合、Normalize / perturbation 用の bounds として使われます。

### 6.6 multi-output / hybrid

`multi_output_config` を使います。異なる task family を混ぜる場合や multiclass multi-output では `use_hybrid=true` を指定してください。

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

出力ごとに設定を変える場合は `output_configs` を使います。

```json
{
  "model_config": {
    "task_type": "hybrid",
    "model_type": "base",
    "multi_output_config": {
      "use_hybrid": true,
      "output_configs": [
        {"name": "yield", "task_type": "regression", "model_type": "base"},
        {"name": "defect", "task_type": "binary", "model_type": "base"}
      ]
    }
  }
}
```

---

## 7. 予測リクエスト

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

`return_type` は次を指定できます。

| return_type | 内容 |
|---|---|
| `posterior` | posterior summary を返す |
| `mean` | mean のみ返す |
| `variance` | variance のみ返す |
| `mean_variance` | mean と variance を返す |

binary prediction response は `prediction_space="probability"` を返します。`mean` はクラス1確率です。`variance_kind="bernoulli_observation"` の `variance` は `p * (1 - p)` であり、確率推定値の epistemic variance ではありません。

Multiclass の `mean` は class probability として扱います。shape は概ね `n x C` または `n x output x C` 系になります。

---

## 8. 候補点生成リクエスト

### 8.1 regression EI

```json
{
  "acquisition_config": {
    "name": "EI",
    "acqf_kwargs": {
      "best_f": 1.0
    }
  },
  "optimize_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 256
  }
}
```

### 8.2 regression UCB

`bochan.api.AcquisitionConfig` では UCB に `beta=3.0` の既定値があります。変えたい場合は `acqf_kwargs.beta` を指定します。

```json
{
  "acquisition_config": {
    "name": "UCB",
    "acqf_kwargs": {
      "beta": 2.0
    }
  },
  "optimize_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 256,
    "sequential": true
  }
}
```

### 8.3 optimizer と evo_method

`OptimizeConfig.optimizer` には backend family 名を指定します。

```json
{
  "optimize_config": {
    "optimizer": "optimize_acqf",
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 256
  }
}
```

Evolutionary backend を使う場合です。

```json
{
  "acquisition_config": {"name": "NIPV"},
  "optimize_config": {
    "optimizer": "evo",
    "evo_method": "ga",
    "q": 5,
    "optimizer_kwargs": {
      "population_size": 128,
      "num_generations": 80
    }
  }
}
```

`evo_method` は次を指定できます。

```text
ga / pso / sa / cmaes
```

`optimizer` に直接 `"ga"`, `"pso"`, `"sa"`, `"cmaes"` を指定した場合も `optimizer="evo"` に正規化されます。`cmaes` は `q > 1` のとき sequential に解決されます。

### 8.4 fixed_features / fixed_features_list

特定列を固定したい場合は `fixed_features` を使います。JSON object の key は文字列になりますが、converter で int に変換されます。

```json
{
  "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "optimize_config": {
    "q": 3,
    "fixed_features": {
      "2": 1.0
    }
  }
}
```

Mixed optimizer のカテゴリ列展開では `fixed_features_list` を使います。

```json
{
  "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "optimize_config": {
    "q": 3,
    "fixed_features_list": [
      {"2": 0.0},
      {"2": 1.0},
      {"2": 2.0}
    ]
  }
}
```

### 8.5 linear constraints

BoTorch 互換の線形制約は、dict 形式または list 形式で指定できます。

```json
{
  "optimize_config": {
    "q": 1,
    "inequality_constraints": [
      {"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 1.0}
    ]
  }
}
```

同じ内容を list で書く例です。

```json
{
  "optimize_config": {
    "q": 1,
    "inequality_constraints": [
      [[0, 1], [1.0, 1.0], 1.0]
    ]
  }
}
```

### 8.6 candidate repair: step 丸め・k-sparse

`repair_config` は `CandidateRepairConfig` に変換されます。候補点の丸め、k-sparse、制約補修に使います。

```json
{
  "acquisition_config": {"name": "NIPV"},
  "optimize_config": {
    "q": 5,
    "repair_config": {
      "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
      "numeric_indices": [0, 1, 2],
      "steps": [0.1, 0.1, 0.1],
      "comp_idx": [0, 1, 2],
      "k": 2,
      "final_priority": "constraints"
    }
  }
}
```

`steps=null` なら丸めません。`comp_idx=[]` または `null` の場合は k-sparse ではなく、丸め・制約補修だけを行う用途になります。

---

## 9. active learning / level-set examples

### 9.1 multiclass active learning: entropy

`task_type="multiclass"` の fitted model に対して `name="entropy"` を指定すると、single / multi / hetero の状態に応じて対応する multiclass acquisition に解決されます。

```json
{
  "acquisition_config": {
    "name": "entropy"
  },
  "optimize_config": {
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

### 9.2 multiclass level-set estimation

クラス2の確率が 0.5 付近の境界を探索する例です。

```json
{
  "acquisition_config": {
    "name": "straddle",
    "acqf_kwargs": {
      "target_class": 2,
      "threshold": 0.5
    }
  },
  "optimize_config": {
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

### 9.3 multiclass Bayesian optimization: target class EI

多クラス BO では `target_class` が必須です。目的は `p(target_class | x)` の最大化です。

```json
{
  "acquisition_config": {
    "name": "EI",
    "acqf_kwargs": {
      "target_class": 2,
      "best_f": 0.70,
      "num_samples": 128
    }
  },
  "optimize_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

### 9.4 hetero multiclass acquisition

`model_type="hetero"` の multiclass model では、同じ alias が hetero 版に解決されます。ノイズ重み付けを変えたい場合は `acqf_kwargs` で指定します。

```json
{
  "acquisition_config": {
    "name": "entropy",
    "acqf_kwargs": {
      "noise_mode": "inverse_linear",
      "noise_combine": "multiply",
      "noise_penalty_lambda": 1.0
    }
  },
  "optimize_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

### 9.5 multi-output multiclass acquisition

multi-output multiclass acquisition では `output_reduction` を指定できます。

```json
{
  "acquisition_config": {
    "name": "entropy",
    "acqf_kwargs": {
      "output_reduction": "weighted_mean",
      "output_weights": [0.7, 0.3]
    }
  },
  "optimize_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

---

## 10. outcome constraints

`AcquisitionConfig` は、低レベルの BoTorch sample constraint である `constraints` と、user-facing な `outcome_constraint_config` を受け取れます。通常は JSON 化しやすい `outcome_constraint_config` を推奨します。

### 10.1 numeric outcome constraints

出力0が 0.5 以上、出力1が 1.2 以下、のような sample 上の制約です。

```json
{
  "acquisition_config": {
    "name": "NEHVI",
    "outcome_constraint_config": {
      "output_indices": [0, 1],
      "operators": ["ge", "le"],
      "thresholds": [0.5, 1.2]
    }
  },
  "optimize_config": {
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 256
  },
  "data_context": {
    "X_baseline": [[0.0, 0.0], [1.0, 1.0]],
    "ref_point": [0.0, 0.0]
  }
}
```

### 10.2 feasibility / ordinal rank constraints

`constraints` フィールドには、`kind` 付きの dict を渡せます。内部で feasibility constraint spec に変換されます。

```json
{
  "acquisition_config": {
    "name": "EI",
    "outcome_constraint_config": {
      "constraints": [
        {
          "kind": "feasibility",
          "output": "defect",
          "operator": "le",
          "threshold": 0.2
        }
      ],
      "eta": 0.001,
      "reduce_constraints": "prod",
      "reduce_q": "mean",
      "posterior_mode": "objective"
    },
    "acqf_kwargs": {
      "best_f": 1.0
    }
  },
  "optimize_config": {
    "q": 1
  }
}
```

低レベルの `constraints` と `outcome_constraint_config` は同時指定できません。

---

## 11. ask / tell / refit

### 11.1 ask

```bash
curl -X POST http://127.0.0.1:8000/models/<model_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "acquisition_config": {"name": "entropy"},
    "optimize_config": {"q": 1, "num_restarts": 10, "raw_samples": 128}
  }'
```

### 11.2 tell

```json
{
  "new_X": [[0.42, 0.80]],
  "new_Y": [2],
  "refit": true,
  "fit_config": {
    "num_epochs": 150,
    "lr": 0.03,
    "beta": 0.5
  }
}
```

### 11.3 refit

```json
{
  "fit_config": {
    "num_epochs": 150,
    "lr": 0.03
  }
}
```

---

## 12. acquisition comparison

推奨 payload 名は `acquisition_configs` です。

```json
{
  "acquisition_configs": [
    {"name": "entropy"},
    {"name": "margin"},
    {"name": "EI", "acqf_kwargs": {"target_class": 2, "best_f": 0.70}}
  ],
  "optimize_config": {
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 128
  }
}
```

互換名として `acq_configs` も使えます。

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

## 13. schema と converter の考え方

FastAPI 側では `ModelConfigSchema`, `FitConfigSchema`, `ObjectiveConfigSchema`, `OutcomeConstraintConfigSchema`, `OptimizeConfigSchema` などの Pydantic schema を受け取ります。

`converters.py` が次の変換を担当します。

```text
ModelConfigSchema            -> bochan.api.ModelConfig
FitConfigSchema              -> bochan.api.FitConfig
ObjectiveConfigSchema        -> bochan.api.ObjectiveConfig
OutcomeConstraintConfigSchema -> bochan.api.OutcomeConstraintConfig
AcquisitionConfigSchema      -> bochan.api.AcquisitionConfig
OptimizeConfigSchema         -> bochan.api.OptimizeConfig
DataContextSchema            -> bochan.api.DataContext
JSON numeric arrays          -> torch.Tensor
```

Multiclass でよく使う `target_class`, `threshold`, `num_samples`, `output_reduction`, `output_weights`, `noise_mode` などは `acqf_kwargs` にそのまま渡します。`output_weights`, `best_f`, `ref_point`, `X_baseline` など tensor 的に扱う値は converter 側で必要に応じて tensor 化されます。

---

## 14. 旧 FastAPI 実装からの移行

正式な HTTP API は `bochan.serving.fastapi` に統一しました。旧 `/bochan/sessions` 系 API は削除され、同等のモデル作成・予測・候補生成・ask/tell・永続化は `/api/v1/models` 系エンドポイントで提供します。ステートレス候補生成は `/api/v1/suggest` を使用してください。

## Tabular feature importance

`POST /tabular/models/{model_id}/feature-importance` accepts optional evaluation records, a JSON feature-importance config, and presentation settings. With no records it evaluates stored training data and returns a warning. Responses contain the JSON-safe Core result, a long summary, diagnostics, and Plotly payloads; per-repeat values default to disabled. Cross-validation fit requests accept `cv_config.feature_importance_config`.
