# bochan.tabular 多目的最適化ガイド

この文書は、`bochan.tabular.TabularBayesianOptimizer` で複数の数値目的変数を扱う場合の実践的な設定指針です。
既存の `src/bochan/tabular/README.md` の補足として、特に `ModelListGP` 相当、`kronecker`、`multitask`、`hybrid` の使い分けを整理します。

---

## 1. まず決めること

多目的最適化では、次の 4 点を先に決めます。

1. 目的変数を何列使うか
2. 各目的を最大化するか、最小化するか
3. 目的変数間の相関をモデルに学習させたいか
4. 目的変数がすべて同じ種類か、回帰・分類・順序などが混在するか

`bochan.tabular` では、複数目的は基本的に `target_cols=[...]` で指定します。

```python
bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="base",
    input_cols=["temperature", "pressure", "machine"],
    target_cols=["yield", "strength"],
    categorical_cols=["machine"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
        "machine": [0, 2],
    },
)
```

ただし、`model_type="base"` だけでは「複数目的をどのようなモデル構造で扱うか」が曖昧になります。独立モデルとして扱いたい場合は、次のように `MultiOutputConfig` を明示することを推奨します。

```python
from bochan.api import MultiOutputConfig
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_names=["yield", "strength"],
    ),
    input_cols=["temperature", "pressure", "machine"],
    target_cols=["yield", "strength"],
    categorical_cols=["machine"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
        "machine": [0, 2],
    },
    maxiter=128,
)

bo.fit(df)
```

この設定では、各目的に対して submodel を作り、内部では BoTorch の `ModelListGP` に束ねます。

---

## 2. モデル構造の使い分け

| 設定 | 内部の考え方 | 推奨する場面 | 注意点 |
| --- | --- | --- | --- |
| `model_type="base"` + `MultiOutputConfig()` | 目的ごとに独立した GP を作り、`ModelListGP` として束ねる | まず試す標準設定。目的変数間の相関を強く仮定したくない場合 | 目的間相関は直接は学習しない |
| `model_type="kronecker"` | Kronecker 型の multi-task GP として目的間相関を学習する | すべての目的が同じ実験点 `X` で観測され、目的間に物理的・統計的な相関がありそうな場合 | 目的間相関の仮定が強い。データ数が少ないと不安定な場合がある |
| `model_type="multitask"` | wide 形式の `Y=[n, m]` を内部で task-id 付き long 形式に変換して MultiTaskGP 的に扱う | 目的を「同じ入力空間上の複数タスク」と見なし、タスク間の情報共有をしたい場合 | `multi_objective` の mixed input では標準 registry に `multitask` がないため、カテゴリ列がある場合は `base` + `MultiOutputConfig` や `kronecker` を優先 |
| `task_type="hybrid"` + `OutputConfig` | 出力ごとに task type / model type を変え、hybrid wrapper に束ねる | 回帰目的、binary 判定、multiclass、ordinal などを同時に扱いたい場合 | objective の指定も出力ごとに意識する必要がある |

重要な点として、`model_list` は `model_type` ではありません。`bochan` の公開 API では、`model_type="model_list"` とは書かず、`model_type="base"` と `MultiOutputConfig()` の組み合わせで `ModelListGP` 相当の構造を作ります。

---

## 3. 推奨順序

通常は、次の順に試すのが安全です。

1. `model_type="base"` + `MultiOutputConfig()`
2. 目的間に明確な相関がありそうなら `model_type="kronecker"`
3. 目的を task として共有構造で扱いたいなら `model_type="multitask"`
4. 目的の種類が混在するなら `task_type="hybrid"`

最初から `kronecker` や `multitask` を使うより、まず独立モデルで予測精度・Pareto front・候補点の妥当性を確認し、その後に相関を使うモデルと比較する方が原因切り分けしやすくなります。

---

## 4. ModelListGP 相当の多目的最適化

### 4.1 学習

```python
import pandas as pd
import torch

from bochan.api import DataContext, MultiOutputConfig
from bochan.tabular import TabularBayesianOptimizer


df = pd.read_csv("data.csv")

bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_names=["yield", "strength"],
    ),
    input_cols=["temperature", "pressure", "machine"],
    target_cols=["yield", "strength"],
    categorical_cols=["machine"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
        "machine": [0, 2],
    },
    maxiter=128,
)

bo.fit(df)
```

この設定では、`yield` 用の GP と `strength` 用の GP が別々に作られます。目的間の相関を無理に学習しないため、少量データやノイズが大きい実験データでは堅実な初期設定になります。

### 4.2 NEHVI で候補点を出す

`NEHVI` / `EHVI` などの hypervolume 系獲得関数では、`X_baseline` と `ref_point` が必要です。`ref_point` は「これより悪い点」を表す基準点であり、目的空間の次元数に合わせます。

```python
ref_point = torch.tensor(
    [0.0, 0.0],
    dtype=bo.train_Y.dtype,
    device=bo.train_Y.device,
)

candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=3,
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "maximize"],
    objective_weights=[1.0, 1.0],
    data_context=DataContext(
        X_baseline=bo.train_X,
        Y_baseline=bo.train_Y,
        ref_point=ref_point,
    ),
    optimizer="optimize_acqf",
    num_restarts=10,
    raw_samples=256,
)
```

`objective_outputs` を指定した場合は、`Y_baseline` と `ref_point` もその出力次元に合わせるのが安全です。たとえば `target_cols=["yield", "cost", "strength"]` のうち `yield` と `strength` だけを使う場合は、`objective_outputs=[0, 2]` とし、`Y_baseline=bo.train_Y[:, [0, 2]]`、`ref_point` は2次元にします。

```python
selected_outputs = [0, 2]

candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=3,
    objective_mode="multi_output",
    objective_outputs=selected_outputs,
    objective_directions=["maximize", "maximize"],
    data_context=DataContext(
        X_baseline=bo.train_X,
        Y_baseline=bo.train_Y[:, selected_outputs],
        ref_point=torch.tensor([0.0, 0.0], dtype=bo.train_Y.dtype, device=bo.train_Y.device),
    ),
)
```

### 4.3 最小化目的を含む場合

たとえば `yield` は最大化、`cost` は最小化したい場合は、`objective_directions` に方向を指定できます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=3,
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "minimize"],
    objective_weights=[1.0, 1.0],
    data_context=DataContext(
        X_baseline=bo.train_X,
        Y_baseline=bo.train_Y,
        ref_point=ref_point,
    ),
)
```

ただし、hypervolume の `ref_point` は acquisition が見る objective 空間に合わせる必要があります。方向変換を含むと混乱しやすい場合は、事前に `minus_cost = -cost` のような列を作り、すべて最大化として扱うと解釈が単純になります。

```python
df["minus_cost"] = -df["cost"]

bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="base",
    multi_output_config=MultiOutputConfig(output_names=["yield", "minus_cost"]),
    input_cols=["temperature", "pressure"],
    target_cols=["yield", "minus_cost"],
    bounds={"temperature": [800.0, 1200.0], "pressure": [0.1, 1.0]},
)
```

---

## 5. `kronecker` を使う場合

`kronecker` は、目的変数を multi-task output として扱い、目的間の相関構造を学習します。

```python
bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="kronecker",
    input_cols=["temperature", "pressure"],
    target_cols=["yield", "strength"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
    },
    maxiter=128,
)

bo.fit(df)
```

`kronecker` は次のような場合に有効です。

- すべての目的が同じ候補点 `X` で測定されている
- 目的間に相関があると考える材料・プロセス上の理由がある
- 一方の目的の観測から、他方の目的の予測にも情報を共有したい
- 目的数が多く、独立モデルではデータ効率が悪いと感じる

一方で、目的間相関の仮定が外れている場合は、独立な `ModelListGP` より悪くなることがあります。`plot_yy(target=...)` や validation RMSE を目的ごとに比較し、独立モデルとの性能差を確認してください。

---

## 6. `multitask` を使う場合

`multitask` は、公開 API では `X=[n, d]`, `Y=[n, m]` の wide 形式で受け取り、内部では task-id を付けた long 形式に変換して扱うモデルです。目的を「同じ入力空間上の複数タスク」と見なしたい場合に使います。

```python
bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="multitask",
    input_cols=["temperature", "pressure"],
    target_cols=["yield", "strength"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
    },
    maxiter=128,
)

bo.fit(df)
```

`multitask` は次のような場合に検討します。

- 各目的を task として扱いたい
- 目的間で滑らかさや応答傾向をある程度共有できると考える
- 各目的が同じ説明変数セットで定義される
- 目的ごとのデータ量が少なく、task 間で情報共有したい

注意点として、`multi_objective` の mixed input には、現在の標準 registry では `model_type="multitask"` が登録されていません。カテゴリ列を含む tabular データでは、まず `model_type="base"` + `MultiOutputConfig()` を使うか、目的間相関を使いたい場合は `model_type="kronecker"` を検討してください。

---

## 7. `hybrid` を使う場合

すべての出力が連続値の目的変数なら `task_type="multi_objective"` で十分です。一方で、次のように出力の種類が混在する場合は `task_type="hybrid"` を使います。

- `yield`: 連続値として最大化
- `defect`: binary probability として最小化
- `rank`: ordinal utility として最大化

```python
from bochan.api import MultiOutputConfig, OutputConfig

bo = TabularBayesianOptimizer(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="yield"),
            OutputConfig(task_type="binary", model_type="base", name="defect"),
            OutputConfig(task_type="ordinal", model_type="base", name="rank", model_kwargs={"num_classes": 4}),
        ],
        use_hybrid=True,
    ),
    input_cols=["temperature", "pressure"],
    target_cols=["yield", "defect", "rank"],
    bounds={
        "temperature": [800.0, 1200.0],
        "pressure": [0.1, 1.0],
    },
)

bo.fit(df)
```

`hybrid` では、出力名を使って objective や feasibility constraint を指定しやすくなります。連続値だけの多目的最適化では過剰な設定になりやすいため、まずは `multi_objective` を使う方が簡潔です。

---

## 8. 獲得関数の選び方

| 獲得関数 | 用途 |
| --- | --- |
| `NEHVI` | 実験ノイズがある多目的最適化の標準候補。実験データではまずこれを推奨 |
| `EHVI` / `EHI` | ノイズが小さい、または deterministic な評価を想定する場合 |
| `NParEGO` | 多目的をランダム scalarization に落として扱う。目的数が多い場合や軽量に試したい場合 |
| `qSimpleRegret` / posterior mean 系 | モデルの予測平均から単純に良い点を探す確認用 |
| `NIPV` / variance 系 | 最適化よりも active learning 的に不確かさを減らしたい場合 |

多目的最適化で Pareto front を改善したい場合は `NEHVI` を第一候補にしてください。単一の重み付きスコアだけで良い場合は、`objective_mode="multi_output"` と `objective_weights` で scalar 化して `EI` / `UCB` を使う方法もあります。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="UCB",
    q=3,
    acqf_kwargs={"beta": 2.0},
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "minimize"],
    objective_weights=[0.7, 0.3],
)
```

この場合は Pareto front 全体ではなく、重みで定義した単一スコアを改善する候補になります。

---

## 9. optimizer backend の選び方

| backend | 推奨場面 |
| --- | --- |
| `optimizer="optimize_acqf"` | 連続変数中心で、BoTorch 標準の勾配ベース最適化を使いたい場合 |
| `optimizer="evo", evo_method="ga"` | カテゴリ列、step 丸め、k-sparse、複雑な repair がある場合 |
| `optimizer="nsgaii"` | population ベースで探索したい場合。獲得関数自体は最終的に候補点スコアを返す必要がある |
| `optimizer="torch"` | torch optimizer で直接 candidate を更新したい実験的用途 |

カテゴリ列を含む tabular データでは、最初は `optimizer="evo", evo_method="ga"` の方が扱いやすい場合があります。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=5,
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "maximize"],
    data_context=DataContext(
        X_baseline=bo.train_X,
        Y_baseline=bo.train_Y,
        ref_point=ref_point,
    ),
    optimizer="evo",
    evo_method="ga",
    optimizer_kwargs={
        "population_size": 128,
        "num_generations": 80,
    },
)
```

---

## 10. 予測と可視化

複数目的では、`predict()` の出力列は目的変数ごとに作られます。

```python
pred_df = bo.predict(df[["temperature", "pressure"]])
print(pred_df)
```

出力例です。

```text
yield_mean
yield_variance
strength_mean
strength_variance
```

2目的の実測値や候補点予測を確認する場合は、`plot_pareto()` を使います。

```python
fig = bo.plot_pareto("yield", "strength")
fig.show()
```

目的ごとの予測精度は、`plot_yy(target=...)` で個別に確認します。

```python
fig = bo.plot_yy(target="yield")
fig.show()

fig = bo.plot_yy(target="strength")
fig.show()
```

`kronecker` や `multitask` を使う場合でも、最終的には目的ごとの予測精度と Pareto front の改善が重要です。相関を入れたモデルが必ず良くなるわけではないため、`ModelListGP` 相当の独立モデルと比較してください。

---

## 11. 典型的なトラブル

### 11.1 `model_type="model_list"` が使えない

`model_list` は `bochan` の `model_type` ではありません。次のように書きます。

```python
bo = TabularBayesianOptimizer(
    task_type="multi_objective",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
    target_cols=["y1", "y2"],
)
```

### 11.2 `NEHVI` で `ref_point` の次元が合わない

`objective_outputs` の数、`Y_baseline` の列数、`ref_point` の長さを揃えてください。

```python
objective_outputs = [0, 2]
Y_baseline = bo.train_Y[:, objective_outputs]
ref_point = torch.tensor([0.0, 0.0], dtype=bo.train_Y.dtype, device=bo.train_Y.device)
```

### 11.3 `multitask` と `MultiOutputConfig` を混同している

- `MultiOutputConfig()` は、複数 submodel を作って wrapper に束ねる設定です。
- `model_type="multitask"` は、1つの multitask 系モデルで task 間の共有構造を学習する設定です。

通常はどちらか一方を選びます。独立モデルとして始めるなら `MultiOutputConfig()`、task 間相関を明示的に使いたいなら `model_type="multitask"` を使います。

### 11.4 mixed input で `multitask` が見つからない

`categorical_cols` を指定すると、tabular API は mixed input として扱います。`multi_objective` の mixed input では標準 registry に `multitask` がないため、`base` + `MultiOutputConfig()` または `kronecker` を使ってください。

---

## 12. まとめ

- まずは `model_type="base"` + `MultiOutputConfig()` を標準設定にする。
- 目的間相関を使いたい場合だけ `kronecker` や `multitask` を比較する。
- `model_list` は `model_type` ではなく、`MultiOutputConfig()` によって作られる wrapper の考え方として理解する。
- `NEHVI` では `DataContext(X_baseline, Y_baseline, ref_point)` の次元を objective と揃える。
- 最小化目的は `objective_directions` で指定できるが、混乱する場合は `minus_cost=-cost` のように事前変換して全目的最大化に寄せる。
- 相関モデルを使う場合でも、目的ごとの予測精度と Pareto front の妥当性を独立モデルと比較する。
