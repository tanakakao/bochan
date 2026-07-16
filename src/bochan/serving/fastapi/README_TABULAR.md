# Tabular FastAPI

`/api/v1/tabular/models` は `TabularBayesianOptimizer` を直接利用するHTTP APIです。通常の `/api/v1/models` と異なり、DataFrame相当のJSON records、列名、文字列カテゴリを受け取ります。

利用時はFastAPIとtabularのoptional dependencyを両方インストールします。

```bash
pip install -e ".[api,tabular]"
```

## Endpoints

| method | path | purpose |
|---|---|---|
| `POST` | `/api/v1/tabular/models` | recordsからモデルを作成・学習 |
| `GET` | `/api/v1/tabular/models` | tabular model id一覧 |
| `POST` | `/api/v1/tabular/models/{model_id}/predict` | 列名付きデータを予測 |
| `POST` | `/api/v1/tabular/models/{model_id}/candidates` | 列名・カテゴリラベル付き候補を生成 |
| `POST` | `/api/v1/tabular/models/{model_id}/ask` | candidatesのask alias |
| `DELETE` | `/api/v1/tabular/models/{model_id}` | インメモリモデルを削除 |

Tensor APIとtabular APIは別のインメモリstoreを使います。サーバー再起動時には両方のstoreが消去されます。

## Jupyter / TestClient example

```python
import pandas as pd
from fastapi.testclient import TestClient

from bochan.serving.fastapi import create_app


df = pd.read_csv("resin.csv")
client = TestClient(create_app())

fit_response = client.post(
    "/api/v1/tabular/models",
    json={
        "data": df.to_dict(orient="records"),
        "input_cols": [
            "raw material 1",
            "raw material 2",
            "raw material 3",
            "temperature",
            "time",
        ],
        "target_cols": ["property", "property2"],
        "categorical_cols": ["raw material 1"],
        "model_config": {
            "task_type": "regression",
            "model_type": "base",
            "input_transform_config": {
                "perturbation": False,
                "n_w": 4,
                "std": 0.1,
            },
        },
        "fit_config": {"maxiter": 128},
    },
)
fit_response.raise_for_status()
model_id = fit_response.json()["model_id"]

candidate_response = client.post(
    f"/api/v1/tabular/models/{model_id}/candidates",
    json={
        "acquisition_config": {"name": "ehvi"},
        "optimize_config": {
            "q": 2,
            "optimizer": "optimize_acqf",
            "num_restarts": 2,
            "raw_samples": 4,
        },
    },
)
candidate_response.raise_for_status()

candidates_df = pd.DataFrame(candidate_response.json()["candidates"])
display(candidates_df)
```

`categorical_cols` に指定した文字列カテゴリは学習時に自動エンコードされ、候補応答では `return_original_categories=True` の既定値により元のラベルへ戻されます。`fixed_features`、`repair_config.numeric_indices`、`comp_idx` などには、tabular Python APIと同様に列名を使用できます。
