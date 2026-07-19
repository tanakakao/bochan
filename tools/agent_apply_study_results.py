from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find patch marker for {label}")
    return text.replace(old, new, 1)


# Install the result API on the public Study subclass.
path = Path("src/bochan/api/study_controls.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    ")\n\nDirection = Literal[\"maximize\", \"minimize\"]",
    ")\nfrom .study_results import install_study_result_api\n\nDirection = Literal[\"maximize\", \"minimize\"]",
    label="study_results import",
)
if "install_study_result_api(BochanStudy)" not in text:
    text = text.rstrip() + "\n\n\ninstall_study_result_api(BochanStudy)\n"
path.write_text(text, encoding="utf-8")


# Export Study visualization helpers from bochan.visualization.
path = Path("src/bochan/visualization/__init__.py")
text = path.read_text(encoding="utf-8")
study_import = '''from .study import (
    show_optimization_history_study,
    show_pareto_front_study,
    study_history_dataframe,
    study_pareto_dataframe,
)
'''
text = replace_once(
    text,
    "from .plots import (\n",
    study_import + "from .plots import (\n",
    label="visualization study import",
)
text = replace_once(
    text,
    '    "show_ordinal_triscatter_from_optimizer",\n',
    '    "show_ordinal_triscatter_from_optimizer",\n'
    '    "show_optimization_history_study",\n'
    '    "show_pareto_front_study",\n',
    label="visualization show exports",
)
text = replace_once(
    text,
    '    "training_dataframe",\n',
    '    "training_dataframe",\n'
    '    "study_history_dataframe",\n'
    '    "study_pareto_dataframe",\n',
    label="visualization dataframe exports",
)
path.write_text(text, encoding="utf-8")


# Remove an unused import from the newly added module.
path = Path("src/bochan/visualization/study.py")
text = path.read_text(encoding="utf-8").replace("import numpy as np\n", "")
path.write_text(text, encoding="utf-8")


# Document best results and visualizations in the Study guide.
path = Path("src/bochan/api/STUDY_README.md")
text = path.read_text(encoding="utf-8")n
method_marker = "| `trials_dataframe()` | trial履歴を表形式へ変換 |"
method_replacement = '''| `trials_dataframe()` | trial履歴を表形式へ変換 |
| `best_trial` | 単目的studyの最良Trialを取得 |
| `best_value` | 単目的studyの最良目的値を取得 |
| `best_x` | 単目的studyの最良入力を取得 |
| `best_params` | 最良入力をパラメータ名付きdictで取得 |
| `get_best_trial(...)` | 出力indexと方向を指定して最良Trialを取得 |
| `best_trials(top_k=...)` | 単一出力について上位Trialを取得 |
| `best_result(...)` | 最良Trialの情報をまとめたdictを取得 |
| `pareto_trials(...)` | 多目的studyの非劣解Trialを取得 |'''
text = replace_once(
    text,
    method_marker,
    method_replacement,
    label="Study method table",
)
section = r'''

---

## 14. best結果と可視化

### 14.1 単目的のbest情報

単目的studyでは、Optunaに近いプロパティで最良結果を取得できます。

```python
print(study.best_trial)   # Trialオブジェクト
print(study.best_value)   # 最良目的値
print(study.best_x)       # 最良入力
print(study.best_params)  # {"x0": ..., "x1": ...}
```

入力名を付けたい場合は、studyのmetadataに`feature_names`または`param_names`を登録します。

```python
study = BochanStudy(
    bounds=bounds,
    metadata={
        "feature_names": ["temperature", "pressure"],
    },
)

print(study.best_params)
# {"temperature": ..., "pressure": ...}
```

明示的な名前をその場で指定することもできます。

```python
params = study.get_best_params(
    param_names=["temperature", "pressure"],
)
```

最良trialの情報をまとめて取得する場合は`best_result()`を使用します。

```python
result = study.best_result()

print(result["trial_id"])
print(result["value"])
print(result["values"])
print(result["x"])
print(result["params"])
print(result["direction"])
print(result["metadata"])
```

上位複数件は既存の`best_trials()`で取得できます。非有限値を持つtrialは自動的に除外されます。

```python
top_trials = study.best_trials(
    top_k=5,
    output_index=0,
    direction="maximize",
)
```

`direction`を省略すると、`acq_config.objective_config.direction`または`directions`から推定し、設定がなければ`maximize`を使用します。

### 14.2 multi-output・多目的の結果

複数の目的値がある場合、単一の`best_trial`は一意に決まりません。そのため、`best_trial`、`best_value`、`best_x`、`best_params`はエラーにし、目的を明示するAPIを使用します。

```python
best_strength = study.get_best_trial(
    output_index=0,
    direction="maximize",
)

lowest_cost = study.get_best_trial(
    output_index=1,
    direction="minimize",
)
```

Pareto非劣解は`pareto_trials()`で取得できます。

```python
pareto_trials = study.pareto_trials(
    output_indices=[0, 1],
    directions=["maximize", "minimize"],
)

for trial in pareto_trials:
    print(trial.trial_id, trial.x, trial.y)
```

### 14.3 最適化履歴

`show_optimization_history_study()`は、各trialの観測値とbest-so-farを表示します。

```python
from bochan.visualization import show_optimization_history_study

fig = show_optimization_history_study(
    study,
    output_index=0,
    direction="maximize",
    target_name="strength",
)
fig.show()
```

描画前のDataFrameだけが必要な場合は`study_history_dataframe()`を使用します。

```python
from bochan.visualization import study_history_dataframe

history = study_history_dataframe(
    study,
    output_index=0,
    target_name="strength",
)
```

履歴には次の列が含まれます。

- `trial_id`
- `order`
- `cycle`
- 目的値列
- `best_value`
- `is_best`

既存の`show_target_over_cycle_study()`は、trial metadataの`cycle`ごとの生データや平均・中央値・最大値などを表示する用途で引き続き利用できます。

```python
from bochan.visualization import show_target_over_cycle_study

fig = show_target_over_cycle_study(
    study,
    target="strength",
    target_cols=["strength"],
    agg="mean",
)
fig.show()
```

### 14.4 Pareto front

2目的のPareto frontは`show_pareto_front_study()`で表示できます。

```python
from bochan.visualization import show_pareto_front_study

fig = show_pareto_front_study(
    study,
    output_indices=[0, 1],
    directions=["maximize", "minimize"],
    target_cols=["strength", "cost"],
)
fig.show()
```

表形式の結果だけが必要な場合は`study_pareto_dataframe()`を使用します。`is_pareto=True`が非劣解です。

```python
from bochan.visualization import study_pareto_dataframe

pareto_df = study_pareto_dataframe(
    study,
    directions=["maximize", "minimize"],
    target_cols=["strength", "cost"],
)
```

可視化関数には`plotly`、データ作成関数には`pandas`が必要です。
'''
if "## 14. best結果と可視化" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")


# Add the same discoverability to the visualization guide.
path = Path("src/bochan/visualization/README.md")
text = path.read_text(encoding="utf-8")
section = r'''

## BochanStudy の結果と最適化履歴

単目的studyのbest-so-far履歴を表示できます。

```python
from bochan.visualization import show_optimization_history_study

fig = show_optimization_history_study(
    study,
    output_index=0,
    direction="maximize",
    target_name="strength",
)
fig.show()
```

描画に使うDataFrameは直接取得できます。

```python
from bochan.visualization import study_history_dataframe

history = study_history_dataframe(study, target_name="strength")
```

trial metadataに`cycle`がある場合、既存の`show_target_over_cycle_study()`でcycle単位の推移を表示できます。

```python
from bochan.visualization import show_target_over_cycle_study

fig = show_target_over_cycle_study(
    study,
    target="strength",
    target_cols=["strength"],
    agg="mean",
)
fig.show()
```

2目的studyでは、完了trialとPareto frontを表示できます。

```python
from bochan.visualization import show_pareto_front_study

fig = show_pareto_front_study(
    study,
    output_indices=[0, 1],
    directions=["maximize", "minimize"],
    target_cols=["strength", "cost"],
)
fig.show()
```

Pareto判定済みの表は`study_pareto_dataframe()`で取得できます。
'''
if "## BochanStudy の結果と最適化履歴" not in text:
    text = text.rstrip() + section + "\n"
path.write_text(text, encoding="utf-8")


# Run the new focused tests and lint the new integration files in PR CI.
path = Path(".github/workflows/wide-multitask-smoke.yml")
text = path.read_text(encoding="utf-8")
text = text.replace(
    'python -m pip install -e ".[test,tabular,api]"',
    'python -m pip install -e ".[test,tabular,api,visualization]"',
)
text = replace_once(
    text,
    "            tests/test_fastapi_tabular_endpoints.py \\\n            2>&1 | tee pytest.log",
    "            tests/test_fastapi_tabular_endpoints.py \\\n"
    "            tests/api/test_study.py \\\n"
    "            tests/api/test_study_results.py \\\n"
    "            tests/test_study_visualization.py \\\n"
    "            2>&1 | tee pytest.log",
    label="focused Study tests",
)
text = replace_once(
    text,
    "            src/bochan/api/optimizer_api.py \\\n",
    "            src/bochan/api/optimizer_api.py \\\n"
    "            src/bochan/api/study_controls.py \\\n"
    "            src/bochan/api/study_results.py \\\n",
    label="Study lint sources",
)
text = replace_once(
    text,
    "            src/bochan/visualization/README.md \\\n" if "            src/bochan/visualization/README.md \\\n" in text else "            src/bochan/tabular/prediction_labels.py \\\n",
    (
        "            src/bochan/visualization/README.md \\\n"
        "            src/bochan/visualization/__init__.py \\\n"
        "            src/bochan/visualization/study.py \\\n"
        if "            src/bochan/visualization/README.md \\\n" in text
        else "            src/bochan/tabular/prediction_labels.py \\\n"
        "            src/bochan/visualization/__init__.py \\\n"
        "            src/bochan/visualization/study.py \\\n"
    ),
    label="Study visualization lint sources",
)
text = replace_once(
    text,
    "            tests/test_fastapi_tabular_endpoints.py \\\n            2>&1 | tee ruff.log",
    "            tests/test_fastapi_tabular_endpoints.py \\\n"
    "            tests/api/test_study.py \\\n"
    "            tests/api/test_study_results.py \\\n"
    "            tests/test_study_visualization.py \\\n"
    "            2>&1 | tee ruff.log",
    label="Study lint tests",
)
path.write_text(text, encoding="utf-8")
