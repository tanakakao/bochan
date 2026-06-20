from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

api_path = ROOT / "src/bochan/api/README.md"
text = api_path.read_text(encoding="utf-8")
marker = '''bo.fit(train_X, train_Y)
posterior = bo.predict(test_X)
```
'''
addition = '''bo.fit(train_X, train_Y)
posterior = bo.predict(test_X)
```

### Binary prediction contract

`task_type="binary"` では、`BayesianOptimizer.predict()` は利用可能なら
`model.probability_posterior()` を優先します。`mean` はクラス1確率
`p(y=1 | x)` です。

`variance` は通常 `p * (1 - p)` の **Bernoulli observation variance** であり、
確率推定値そのものの epistemic uncertainty ではありません。
`return_result=True` の `PredictionResult` には次が入ります。

- `prediction_space="probability"`
- `variance_kind="bernoulli_observation"`
- observation noise を加えた場合は `bernoulli_observation_plus_noise`
'''
if text.count(marker) != 1:
    raise RuntimeError("API README prediction example was not found.")
api_path.write_text(text.replace(marker, addition, 1), encoding="utf-8")

serving_path = ROOT / "src/bochan/serving/fastapi/README.md"
text = serving_path.read_text(encoding="utf-8")
anchor = '''現在の store はプロセス内インメモリです。サーバー再起動で fitted model は失われます。実運用では `dependencies.py` の `get_optimizer_store()` を差し替え、モデル artifact や metadata を DB / object storage / model registry に保存してください。
'''
note = anchor + '''
### Binary prediction response

binary prediction response は `prediction_space="probability"` を返します。
`mean` はクラス1確率です。`variance_kind="bernoulli_observation"` の
`variance` は `p * (1 - p)` であり、確率推定値の epistemic variance ではありません。

`return_type="posterior"` でも Python posterior オブジェクトの文字列ではなく、
`type`・`mean`・`variance` を持つ JSON summary を返します。
'''
if text.count(anchor) != 1:
    raise RuntimeError("Serving README store paragraph was not found.")
serving_path.write_text(text.replace(anchor, note, 1), encoding="utf-8")
