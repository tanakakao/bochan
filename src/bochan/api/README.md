# bochan API

`bochan.api` は、BoTorch / GPyTorch ベースのモデル構築・学習・獲得関数生成・候補点最適化を、アプリケーションや外部 API から扱いやすい形にまとめるための高レベル API です。

研究・開発段階では関数単位で細かく扱い、アプリ化・API化するときは `BayesianOptimizer` クラスまたは FastAPI ルーターから文字列中心で操作することを想定しています。

---

## 1. Python API の基本

内部処理は、基本的に次の4段階に分かれます。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)

acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer` は、この流れを薄く包むクラスです。

```python
from bochan.api import BayesianOptimizer, ModelConfig, FitConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)

bo.fit(train_X, train_Y)
posterior = bo.predict(test_X)
```

`model_registry` は通常不要です。省略時は API 標準の `DEFAULT_MODEL_REGISTRY` を内部で参照します。

---

## 2. 設定クラス

| 設定クラス | 役割 |
|---|---|
| `ModelConfig` | モデルの種類、カテゴリ変数、input transform、モデル固有引数を指定する |
| `FitConfig` | 学習回数、learning rate、maxiter などを指定する |
| `InputTransformConfig` | Normalize / input perturbation を簡易指定する |
| `MultiOutputConfig` | multi-output / hybrid の出力ごとのモデル構成を指定する |
| `AcquisitionConfig` | 獲得関数を文字列またはクラスで指定する |
| `DataContext` | `X_baseline`, `best_f`, `ref_point`, `partitioning` など獲得関数側の文脈を渡す |
| `MultiObjectiveConfig` | EHVI / NEHVI / NParEGO などの多目的設定を渡す |
| `OptimizeConfig` | 候補点最適化の q, restart 数, optimizer 種類, 制約を指定する |
| `CandidateRepairConfig` | grid rounding, k-sparse, 制約補修用の post-processing を自動生成する |

---

## 3. FastAPI で使う

FastAPI 統合は optional dependency です。通常の `bochan.api` import では FastAPI を要求しません。

### 3.1 インストール

```bash
pip install 'botorch_ext[api]'
```

開発環境で editable install する場合は次です。

```bash
pip install -e '.[api]'
```

### 3.2 起動

```bash
uvicorn bochan.api.fastapi:app --reload
```

OpenAPI UI は通常、以下で確認できます。

```text
http://127.0.0.1:8000/docs
```

### 3.3 提供エンドポイント

すべて `/bochan` prefix 配下です。

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/bochan/health` | ヘルスチェック |
| `GET` | `/bochan/acquisitions` | 登録済み acquisition 名の一覧 |
| `GET` | `/bochan/sessions` | セッション ID 一覧 |
| `POST` | `/bochan/sessions` | 学習済み `BayesianOptimizer` セッションを作成 |
| `DELETE` | `/bochan/sessions/{session_id}` | セッション削除 |
| `POST` | `/bochan/sessions/{session_id}/predict` | セッションで予測 |
| `POST` | `/bochan/sessions/{session_id}/candidate` | セッションで候補点生成 |
| `POST` | `/bochan/sessions/{session_id}/ask` | `/candidate` の alias |
| `POST` | `/bochan/sessions/{session_id}/tell` | 観測追加と任意の再学習 |
| `POST` | `/bochan/suggest` | stateless に fit → candidate まで実行 |

---

## 4. FastAPI: stateful session 例

### 4.1 セッション作成

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]],
    "model_config": {
      "task_type": "regression",
      "model_type": "base"
    },
    "fit_config": {
      "maxiter": 64
    }
  }'
```

レスポンス例:

```json
{
  "session_id": "...",
  "task_type": "regression",
  "model_type": "base",
  "input_type": "normal",
  "metadata": {
    "model_cls": "SingleTaskGP"
  }
}
```

### 4.2 候補点生成

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/candidate \
  -H 'Content-Type: application/json' \
  -d '{
    "acquisition_config": {
      "name": "EI",
      "acqf_kwargs": {
        "best_f": 1.0
      }
    },
    "optimize_config": {
      "q": 1,
      "num_restarts": 5,
      "raw_samples": 64
    }
  }'
```

### 4.3 予測

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "X": [[0.25], [0.75]],
    "return_type": "mean_variance"
  }'
```

### 4.4 ask / tell

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "acquisition_config": {"name": "UCB", "acqf_kwargs": {"beta": 0.2}},
    "optimize_config": {"q": 1, "num_restarts": 5, "raw_samples": 64}
  }'
```

```bash
curl -X POST http://127.0.0.1:8000/bochan/sessions/<session_id>/tell \
  -H 'Content-Type: application/json' \
  -d '{
    "new_X": [[0.3]],
    "new_Y": [[0.09]],
    "refit": true,
    "fit_config": {"maxiter": 64}
  }'
```

---

## 5. FastAPI: stateless suggest 例

`/bochan/suggest` は、1リクエスト内で `fit -> acquisition -> optimize` まで実行します。セッションを残したくない単発 API に向いています。

```bash
curl -X POST http://127.0.0.1:8000/bochan/suggest \
  -H 'Content-Type: application/json' \
  -d '{
    "train_X": [[0.0], [0.5], [1.0]],
    "train_Y": [[0.0], [0.25], [1.0]],
    "bounds": [[0.0], [1.0]],
    "model_config": {
      "task_type": "regression",
      "model_type": "base"
    },
    "fit_config": {
      "maxiter": 64
    },
    "acquisition_config": {
      "name": "EI",
      "acqf_kwargs": {"best_f": 1.0}
    },
    "optimize_config": {
      "q": 1,
      "num_restarts": 5,
      "raw_samples": 64
    }
  }'
```

---

## 6. FastAPI: multi-output 例

`MultiOutputConfig()` は複数出力モデルを作る合図です。`output_configs` を省略した場合、`train_Y.shape[-1]` の数だけ親の `task_type` / `model_type` を複製します。

```json
{
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
  "bounds": [[0.0], [1.0]],
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "multi_output_config": {}
  },
  "fit_config": {
    "maxiter": 64
  }
}
```

出力ごとに task を変えたい場合は、`output_configs` を指定します。

```json
{
  "model_config": {
    "task_type": "hybrid",
    "multi_output_config": {
      "use_hybrid": true,
      "output_configs": [
        {"name": "strength", "task_type": "regression", "model_type": "base"},
        {"name": "defect", "task_type": "binary", "model_type": "base"},
        {"name": "rank", "task_type": "ordinal", "model_type": "base"}
      ]
    }
  }
}
```

---

## 7. FastAPI: 多目的 EHVI 例

多目的最適化では、モデル側で multi-output を作り、獲得関数側で `EHI` / `EHVI` / `NEHVI` / `NParEGO` を指定します。

```json
{
  "train_X": [[0.0], [0.5], [1.0]],
  "train_Y": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
  "bounds": [[0.0], [1.0]],
  "model_config": {
    "task_type": "regression",
    "model_type": "base",
    "multi_output_config": {}
  },
  "fit_config": {
    "maxiter": 64
  },
  "acquisition_config": {
    "name": "EHI"
  },
  "data_context": {
    "multi_objective": {
      "ref_point": [-0.1, -0.1],
      "Y_baseline": [[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]],
      "auto_partitioning": true
    }
  },
  "optimize_config": {
    "q": 1,
    "num_restarts": 5,
    "raw_samples": 64
  }
}
```

---

## 8. FastAPI: 制約・fixed features・repair

JSON では線形制約を次の形式で渡せます。

```json
{
  "indices": [0, 1],
  "coefficients": [1.0, 1.0],
  "rhs": 1.0
}
```

`fixed_features` と `fixed_features_list` は JSON object として渡します。key は JSON 上は文字列でも、API 内で `int` に変換されます。

```json
{
  "optimize_config": {
    "optimizer": "optimize_acqf_mixed",
    "q": 1,
    "fixed_features": {"0": 0.5},
    "fixed_features_list": [
      {"2": 0},
      {"2": 1}
    ],
    "inequality_constraints": [
      {"indices": [0, 1], "coefficients": [1.0, 1.0], "rhs": 1.0}
    ],
    "repair_config": {
      "numeric_indices": [0, 1],
      "steps": [0.1, 0.1],
      "comp_idx": [0, 1],
      "k": 1,
      "final_priority": "grid"
    }
  }
}
```

`fixed_features` と `fixed_features_list` が両方ある場合は、API 側で各 `fixed_features_list` に `fixed_features` をマージします。

---

## 9. FastAPI: optimizer 選択

`OptimizeConfig.optimizer` は以下を指定できます。

```text
optimize_acqf
optimize_acqf_mixed
evo
optimize_acqf_evo
torch
optimize_acqf_torch
evo_mixed
optimize_acqf_evo_mixed
torch_mixed
optimize_acqf_torch_mixed
```

例:

```json
{
  "optimize_config": {
    "optimizer": "torch",
    "q": 3,
    "num_restarts": 10,
    "raw_samples": 512,
    "optimizer_kwargs": {
      "method": "adam",
      "options": {
        "lr": 0.03,
        "num_steps": 200,
        "penalty_factor": 1000.0
      }
    }
  }
}
```

---

## 10. 実装上の注意

- `bochan.api.fastapi` は optional module です。FastAPI を使わない通常の Python API には影響しません。
- stateful session はメモリ上に `BayesianOptimizer` を保持します。ローカルアプリやプロトタイプ向けです。
- 本番運用では、プロセス再起動でセッションが消える点、ワーカーを複数立てるとメモリストアが共有されない点に注意してください。
- GPU を使う場合は `tensor_options.device` に `"cuda"` などを指定できます。
- JSON では Python の callable や objective object は直接渡せません。API 経由では、文字列指定・数値・配列・dict で表現できる範囲を基本にしてください。
