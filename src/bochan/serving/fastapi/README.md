# bochan FastAPI serving

`bochan.serving.fastapi` は、`bochan.api` を HTTP / JSON から利用する serving 層です。モデル構築、学習、予測、候補点生成、ask / tell を同じ設定語彙で扱います。

## 1. インストールと起動

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

Pythonからappを作る場合:

```python
from bochan.serving.fastapi import create_app

app = create_app(title="bochan Optimization API")
```

## 2. endpoint

| method | path | 内容 |
|---|---|---|
| `GET` | `/health` | ヘルスチェック |
| `POST` | `/models` | model作成・学習 |
| `GET` | `/models` | model id一覧 |
| `DELETE` | `/models/{model_id}` | model削除 |
| `POST` | `/models/{model_id}/predict` | 予測 |
| `POST` | `/models/{model_id}/candidates` | 候補点生成 |
| `POST` | `/models/{model_id}/ask` | ask-and-tell候補生成 |
| `POST` | `/models/{model_id}/tell` | 新規観測追加 |
| `POST` | `/models/{model_id}/refit` | 再学習 |
| `POST` | `/models/{model_id}/candidates/compare` | acquisition比較 |
| `GET` | `/acquisitions/names` | alias一覧 |

storeはプロセス内インメモリです。サーバー再起動でfitted modelは失われます。

## 3. 基本フロー

```text
POST /models
  -> ModelConfig / FitConfig / train_X / train_Y
  -> BayesianOptimizer.fit(...)
  -> model_id

POST /models/{model_id}/predict
  -> BayesianOptimizer.predict(...)

POST /models/{model_id}/candidates
  -> AcquisitionConfig / OptimizeConfig
  -> BayesianOptimizer.candidate(...)
```

## 4. single-output regression

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

## 5. mixed task-feature multi-task

### 5.1 データ契約

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

例の列構成:

```text
continuous | task_id | category
```

`model_type="multitask"` は、タスクごとに入力位置や観測数が異なるlong formatを扱います。`model_type="kronecker"` は全タスクが同じ入力点で観測されるblock designです。

### 5.2 binary task-feature model

```json
{
  "model_config": {
    "task_type": "binary",
    "input_type": "mixed",
    "model_type": "multitask",
    "cat_dims": [2],
    "model_kwargs": {
      "num_tasks": 2,
      "task_feature": 1,
      "rank": 2,
      "num_inducing_points": 32
    },
    "input_transform_config": {
      "normalize": true,
      "perturbation": false,
      "categorical_idx": [1, 2]
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.01
  },
  "train_X": [
    [0.05, 0, 0],
    [0.20, 0, 1],
    [0.60, 0, 0],
    [0.10, 1, 1],
    [0.45, 1, 0],
    [0.90, 1, 1]
  ],
  "train_Y": [0, 0, 1, 0, 1, 1],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
}
```

`categorical_idx`には通常カテゴリ列とtask-id列の両方を指定し、Normalize / perturbationから保護します。task-id列を`cat_dims`にも含めることはできません。

### 5.3 multiclass task-feature model

```json
{
  "model_config": {
    "task_type": "multiclass",
    "input_type": "mixed",
    "model_type": "multitask",
    "cat_dims": [2],
    "model_kwargs": {
      "num_classes": 3,
      "num_tasks": 2,
      "task_feature": 1,
      "rank": 2,
      "num_inducing_points": 32
    },
    "input_transform_config": {
      "normalize": true,
      "categorical_idx": [1, 2]
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.01
  },
  "train_X": [
    [0.05, 0, 0],
    [0.20, 0, 1],
    [0.60, 0, 0],
    [0.10, 1, 1],
    [0.45, 1, 0],
    [0.90, 1, 1]
  ],
  "train_Y": [0, 1, 2, 0, 2, 1],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
}
```

multiclass task-feature modelはクラスlogitごとにtask covarianceを持ちます。

### 5.4 ordinal task-feature model

```json
{
  "model_config": {
    "task_type": "ordinal",
    "input_type": "mixed",
    "model_type": "multitask",
    "cat_dims": [2],
    "model_kwargs": {
      "num_classes": 4,
      "num_tasks": 2,
      "task_feature": 1,
      "rank": 2
    },
    "input_transform_config": {
      "normalize": true,
      "categorical_idx": [1, 2]
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.03
  },
  "train_X": [
    [0.05, 0, 0],
    [0.20, 0, 1],
    [0.60, 0, 0],
    [0.10, 1, 1],
    [0.45, 1, 0],
    [0.90, 1, 1]
  ],
  "train_Y": [0, 1, 3, 0, 2, 3],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
}
```

ordinal multi-taskは全タスクでクラス定義とcutpointを共有します。

### 5.5 Gaussian task-feature model

```json
{
  "model_config": {
    "task_type": "regression",
    "input_type": "mixed",
    "model_type": "multitask",
    "cat_dims": [2],
    "model_kwargs": {
      "task_feature": 1,
      "rank": 2
    },
    "input_transform_config": {
      "normalize": true,
      "categorical_idx": [1, 2]
    }
  },
  "fit_config": {
    "maxiter": 128
  },
  "train_X": [
    [0.05, 0, 0],
    [0.20, 0, 1],
    [0.60, 0, 0],
    [0.10, 1, 1],
    [0.45, 1, 0],
    [0.90, 1, 1]
  ],
  "train_Y": [[0.1], [0.4], [0.8], [0.3], [0.7], [1.0]],
  "bounds": [[0.0, 0.0], [1.0, 1.0]]
}
```

Gaussian `MixedMultiTaskGP`ではtraining `train_X`にtask-id列を含めますが、candidate / prediction boundsはtask列を除いたdata featureに対応させます。

## 6. mixed Kronecker model

block-design binaryの例です。

```json
{
  "model_config": {
    "task_type": "binary",
    "input_type": "mixed",
    "model_type": "kronecker",
    "cat_dims": [2],
    "model_kwargs": {
      "rank": 2,
      "num_inducing_points": 32
    },
    "input_transform_config": {
      "normalize": true,
      "categorical_idx": [2]
    }
  },
  "fit_config": {
    "num_epochs": 300,
    "lr": 0.01,
    "batch_size": null
  },
  "train_X": [
    [0.05, 0.20, 0],
    [0.30, 0.45, 1],
    [0.80, 0.60, 0]
  ],
  "train_Y": [
    [0, 0],
    [0, 1],
    [1, 1]
  ],
  "bounds": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
}
```

Kronecker modelは`train_Y: [n, m]`のcomplete block design専用です。

## 7. 予測

一般的な予測:

```json
{
  "X": [[0.25], [0.75]],
  "return_type": "mean_variance",
  "posterior_kwargs": {}
}
```

### classification / ordinal task-feature

予測`X`にtask-id列を含めます。

```json
{
  "X": [
    [0.25, 0, 1],
    [0.25, 1, 1]
  ],
  "return_type": "mean_variance"
}
```

### Gaussian task-feature

予測`X`からtask-id列を除き、`posterior_kwargs.output_indices`で出力タスクを選択します。

```json
{
  "X": [
    [0.25, 1],
    [0.75, 0]
  ],
  "return_type": "mean_variance",
  "posterior_kwargs": {
    "output_indices": [0, 1]
  }
}
```

binaryの`mean`はクラス1確率です。通常の`variance`は`p * (1-p)`であり、epistemic varianceそのものではありません。

## 8. taskを固定した候補点生成

binary / multiclass / ordinal task-feature modelでは、探索対象task-idを固定し、通常カテゴリを列挙します。

```json
{
  "acq_config": {
    "name": "Entropy"
  },
  "opt_config": {
    "optimizer": "optimize_acqf_mixed",
    "q": 1,
    "num_restarts": 10,
    "raw_samples": 256,
    "fixed_features": {
      "1": 1.0
    },
    "fixed_features_list": [
      {"2": 0.0},
      {"2": 1.0}
    ]
  }
}
```

JSON objectのkeyは文字列ですが、schema側で整数feature indexへ変換されます。

multiclass BOでは`target_class`などを指定します。

```json
{
  "acq_config": {
    "name": "EI",
    "acqf_kwargs": {
      "target_class": 2,
      "best_f": 0.7
    }
  },
  "opt_config": {
    "optimizer": "optimize_acqf_mixed",
    "q": 1,
    "fixed_features": {"1": 1.0},
    "fixed_features_list": [{"2": 0.0}, {"2": 1.0}]
  }
}
```

Gaussian task-feature modelではcandidate tensorにtask-id列を含めず、選択したoutput taskに対応するacquisition構成を使用します。

## 9. ask / tell

候補生成:

```json
POST /models/{model_id}/ask
{
  "acq_config": {"name": "Entropy"},
  "opt_config": {
    "q": 1,
    "optimizer": "optimize_acqf_mixed",
    "fixed_features": {"1": 1.0},
    "fixed_features_list": [{"2": 0.0}, {"2": 1.0}]
  }
}
```

観測追加:

```json
POST /models/{model_id}/tell
{
  "new_X": [[0.55, 1, 0]],
  "new_Y": [1],
  "refit": true,
  "fit_config": {
    "num_epochs": 100,
    "lr": 0.01
  }
}
```

- task-feature model: `new_X`にtask-id列を含める
- Kronecker model: `new_Y`は全タスクを含む`[n_new, m]`

## 10. candidate repair

```json
{
  "acq_config": {"name": "UCB", "acqf_kwargs": {"beta": 0.2}},
  "opt_config": {
    "q": 3,
    "repair_config": {
      "numeric_indices": [0, 1],
      "steps": [0.1, 0.01],
      "comp_idx": null,
      "k": 0
    }
  }
}
```

- `steps=null`: grid roundingなし
- `comp_idx=null`または`[]`: k-sparseなし
- `inequality_sense`: `le`または`ge`

## 11. 注意点

- task-idを`cat_dims`に含めない
- categoryとtask-idをNormalize / perturbationしない
- classification / ordinal candidateではtask-idを固定する
- multiclass task covarianceは`[C, m, m]`
- ordinal multi-taskは全タスクでcutpointを共有する
- Kronecker modelはcomplete block design専用
- fitted modelの永続化は現在のin-memory storeでは行わない

## 12. 関連ドキュメント

- `src/bochan/api/README.md`
- `docs/mixed_task_feature_multitask_models.md`
- `src/bochan/models/regression/gaussian/README.md`
- `src/bochan/models/classification/binary/README.md`
- `src/bochan/models/classification/multiclass/README.md`
- `src/bochan/models/ordinal/README.md`
