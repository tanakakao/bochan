# `AcquisitionConfig(name="llm_selected")`

`BayesianOptimizer.suggest_acquisition()` は、LLMの提案内容・理由・警告を確認してから適用する review-first API です。

一方、次の形式は `candidate()` または `acquisition()` の実行時に獲得関数設定を自動解決します。

```python
from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    ModelConfig,
    ObjectiveConfig,
    OptimizeConfig,
)

bo = BayesianOptimizer(
    model_config=ModelConfig(model_type="llm_selected"),
    bounds=bounds,
    llm_settings=llm_settings,
)
bo.fit(train_X, train_Y)

acq_config = AcquisitionConfig(
    name="llm_selected",
    objective_config=ObjectiveConfig(
        mode="multi_output",
        outputs=[0, 1],
        directions=["maximize", "minimize"],
        weights=[1.0, 0.5],
    ),
)

candidates, acq_value = bo.candidate(
    acq_config=acq_config,
    opt_config=OptimizeConfig(
        optimizer="llm_candidate_set",
        q=3,
    ),
)
```

## 解決時の優先順位

LLMは `NEHVI`、`EHVI`、`NParEGO`、`UCB` などの具体的な獲得関数を選択します。

利用者が `AcquisitionConfig` に明示した次の設定は、LLM提案より優先されます。

- `objective` / `objective_config`
- `constraints` / `outcome_constraint_config`
- `sampler`
- `objective_kwargs`
- `acqf_kwargs`

したがって、目的変数の方向や重みは人が固定し、獲得関数の種類だけをLLMに選ばせる運用ができます。

解決後の具体的な設定は `bo.acq_config` に保存され、LLMの提案情報は `bo.last_acquisition_suggestion` から確認できます。

## `suggest_acquisition()`との使い分け

```python
suggestion = bo.suggest_acquisition(
    "観測ノイズを重視して選択する",
)
print(suggestion.acq_config)
print(suggestion.reasoning_summary)
bo.apply_suggestion(suggestion)
```

こちらは事前確認、比較、監査記録、一部適用に向きます。

```python
AcquisitionConfig(name="llm_selected")
```

こちらは、通常のConfig記法を維持したまま実行時に自動選択したい場合に向きます。

`OptimizeConfig(optimizer="llm_candidate_set")` は別の役割です。これは獲得関数を選ぶ機能ではなく、LLMが候補集合を生成し、具体化された獲得関数で再順位付けする候補最適化backendです。
