# bochan.optim

`bochan.optim` は、BoTorch の獲得関数を最適化して次の候補点を生成するための
optimizer wrapper 群です。標準の勾配ベース最適化に加えて、k-sparse、mixed、
進化計算、PyTorch optimizer、NSGA-II、有限候補集合上の Thompson sampling を提供します。

```python
from bochan.optim import (
    optimize_acqf_k_sparse,
    optimize_acqf_mixed_k_sparse,
    optimize_acqf_evo,
    optimize_acqf_evo_mixed,
    optimize_acqf_evo_k_sparse,
    optimize_acqf_evo_mixed_k_sparse,
    optimize_acqf_torch,
    optimize_acqf_torch_mixed,
    optimize_acqf_torch_k_sparse,
    optimize_acqf_torch_mixed_k_sparse,
    optimize_acqf_nsgaii,
    optimize_thompson_sampling,
    optimize_thompson_sampling_mixed,
)
```

## 共通の戻り値

多くの関数は次の2値を返します。

```python
candidates, acq_value = optimize_func(...)
```

- `candidates`: 形状 `[q, d]` の候補点
- `acq_value`: 候補点に対応する獲得関数値または評価値

Thompson sampling では、`acq_value` 相当として選択候補における posterior mean を返します。
`MaxPosteriorSampling` が内部で使用したサンプル関数値そのものではありません。

---

## 1. 標準最適化 + k-sparse

### `optimize_acqf_k_sparse`

BoTorch の `optimize_acqf` を使いながら、最終候補を k-sparse に補修します。
`comp_idx` で指定した変数のうち、最大 `k` 個だけを非ゼロにします。

```python
from bochan.optim import optimize_acqf_k_sparse

candidates, acq_value = optimize_acqf_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    num_restarts=10,
    raw_samples=256,
    comp_idx=[0, 1, 2, 3],
    k=2,
    sequential=True,
    inequality_constraints=inequality_constraints,
    equality_constraints=equality_constraints,
)
```

主な引数:

- `comp_idx`: sparse化対象の説明変数index
- `k`: 非ゼロを許す最大変数数
- `final_sum_constraint`: 最終候補へ適用する和制約
- `diversify`: バッチ候補を分散させるか
- `support_selection`: `"topk"` または `"sample"`
- `inequality_sense`: `"le"` または `"ge"`

### `optimize_acqf_mixed_k_sparse`

`optimize_acqf_mixed` と k-sparse 補修を組み合わせます。
カテゴリ変数は通常 `comp_idx` から除外してください。

```python
from bochan.optim import optimize_acqf_mixed_k_sparse

fixed_features_list = [
    {4: 0.0},
    {4: 1.0},
    {4: 2.0},
]

candidates, acq_value = optimize_acqf_mixed_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    fixed_features_list=fixed_features_list,
    q=3,
    num_restarts=10,
    raw_samples=256,
    comp_idx=[0, 1, 2, 3],
    k=2,
    sequential=True,
)
```

`fixed_features_list` の代わりに、次のような `categorical_features` も指定できます。

```python
categorical_features = {
    4: [0.0, 1.0, 2.0],
    5: [0.0, 1.0],
}
```

---

## 2. 進化計算 optimizer

### `optimize_acqf_evo`

勾配を使わず、GA、PSO、SA、CMA-ES で獲得関数を最適化します。
丸め・離散化・複雑な post-processing を使う場合に有効です。

```python
from bochan.optim import optimize_acqf_evo

candidates, acq_value = optimize_acqf_evo(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="ga",  # "ga", "pso", "sa", "cmaes"
    sequential=True,
    options={
        "population_size": 128,
        "num_generations": 100,
        "seed": 42,
    },
    post_processing_func=repair,
    inequality_constraints=inequality_constraints,
    equality_constraints=equality_constraints,
)
```

### `candidate_transform_mixed_factory`

進化計算の連続候補を、最も近いカテゴリ値へ写像する transform を作ります。

```python
from bochan.optim import candidate_transform_mixed_factory

candidate_transform = candidate_transform_mixed_factory(
    categorical_features={
        4: [0.0, 1.0, 2.0],
        5: [0.0, 1.0],
    },
    bounds=bounds,
)
```

### `optimize_acqf_evo_mixed`

進化計算とカテゴリ変数の写像をまとめて使います。

```python
from bochan.optim import optimize_acqf_evo_mixed

candidates, acq_value = optimize_acqf_evo_mixed(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="ga",
    categorical_features={4: [0.0, 1.0, 2.0]},
    sequential=True,
    options={"population_size": 128, "num_generations": 100},
)
```

### `optimize_acqf_evo_k_sparse`

進化計算に k-sparse 補修を追加します。

```python
from bochan.optim import optimize_acqf_evo_k_sparse

candidates, acq_value = optimize_acqf_evo_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="pso",
    comp_idx=[0, 1, 2, 3],
    k=2,
    sequential=True,
)
```

### `optimize_acqf_evo_mixed_k_sparse`

mixed、進化計算、k-sparse を同時に扱います。

```python
from bochan.optim import optimize_acqf_evo_mixed_k_sparse

candidates, acq_value = optimize_acqf_evo_mixed_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="ga",
    categorical_features={4: [0.0, 1.0, 2.0]},
    comp_idx=[0, 1, 2, 3],
    k=2,
    sequential=True,
)
```

---

## 3. PyTorch optimizer

### `optimize_acqf_torch`

候補点を PyTorch の parameter として扱い、Adam などで獲得関数を最大化します。

```python
from bochan.optim import optimize_acqf_torch

candidates, acq_value = optimize_acqf_torch(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="adam",
    num_restarts=10,
    raw_samples=256,
    sequential=True,
    options={
        "lr": 0.03,
        "num_steps": 200,
        "penalty_factor": 1e3,
    },
    post_processing_func=repair,
    inequality_constraints=inequality_constraints,
    equality_constraints=equality_constraints,
)
```

### `optimize_acqf_torch_mixed`

カテゴリ組合せごとに PyTorch optimizer を実行します。

```python
from bochan.optim import optimize_acqf_torch_mixed

candidates, acq_value = optimize_acqf_torch_mixed(
    acq_function=acqf,
    bounds=bounds,
    fixed_features_list=[{4: 0.0}, {4: 1.0}, {4: 2.0}],
    q=3,
    method="adam",
    num_restarts=10,
    raw_samples=256,
    sequential=True,
    options={"lr": 0.03, "num_steps": 200},
)
```

### `optimize_acqf_torch_k_sparse`

PyTorch optimizer に k-sparse 補修を追加します。

```python
from bochan.optim import optimize_acqf_torch_k_sparse

candidates, acq_value = optimize_acqf_torch_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    method="adam",
    comp_idx=[0, 1, 2, 3],
    k=2,
    num_restarts=10,
    raw_samples=256,
)
```

### `optimize_acqf_torch_mixed_k_sparse`

mixed、PyTorch optimizer、k-sparse を同時に扱います。

```python
from bochan.optim import optimize_acqf_torch_mixed_k_sparse

candidates, acq_value = optimize_acqf_torch_mixed_k_sparse(
    acq_function=acqf,
    bounds=bounds,
    fixed_features_list=[{4: 0.0}, {4: 1.0}],
    q=3,
    method="adam",
    comp_idx=[0, 1, 2, 3],
    k=2,
    num_restarts=10,
    raw_samples=256,
)
```

---

## 4. NSGA-II

### `optimize_acqf_nsgaii`

多目的関数またはベクトル値 acquisition を NSGA-II で最適化します。

```python
from bochan.optim import optimize_acqf_nsgaii

candidates, values = optimize_acqf_nsgaii(
    acq_function=acqf,
    bounds=bounds,
    q=10,
    options={
        "population_size": 128,
        "num_generations": 200,
        "seed": 42,
    },
)
```

### `equality_constraints_to_inequality_constraints`

等式制約を、許容幅付きの2本の不等式制約へ変換します。

```python
from bochan.optim import equality_constraints_to_inequality_constraints

ineq_constraints = equality_constraints_to_inequality_constraints(
    equality_constraints=equality_constraints,
    tolerance=1e-4,
)
```

### `validate_discrete_choices`

NSGA-IIに渡す離散値候補の定義を検証します。

```python
from bochan.optim import validate_discrete_choices

validate_discrete_choices(
    discrete_choices={
        2: [0.0, 0.5, 1.0],
        4: [0.0, 1.0, 2.0],
    },
    bounds=bounds,
)
```

---

## 5. Thompson sampling

Thompson sampling は、獲得関数値を勾配最適化するのではなく、posterior から
サンプルした関数上で良い候補を選ぶ戦略です。本実装では、有限候補集合に対して
BoTorch の `MaxPosteriorSampling` を適用します。

`acq_function` には通常の acquisition function を渡せます。また、`posterior()` を持つ
model 自体を直接渡すこともできます。

### `optimize_thompson_sampling`

通常入力向けの Thompson sampling です。

```python
from bochan.optim import optimize_thompson_sampling

candidates, values = optimize_thompson_sampling(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    num_restarts=5,
    raw_samples=128,
    options={
        "n_candidates": 4096,
        "replacement": False,
        "observation_noise": False,
        "seed": 42,
    },
    post_processing_func=repair,
    inequality_constraints=inequality_constraints,
    equality_constraints=equality_constraints,
)
```

model を直接渡す例:

```python
candidates, values = optimize_thompson_sampling(
    acq_function=model,
    bounds=bounds,
    q=3,
    options={"n_candidates": 2048},
)
```

明示候補集合を使う例:

```python
candidate_set = torch.rand(5000, bounds.shape[-1], dtype=bounds.dtype)
candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set

candidates, values = optimize_thompson_sampling(
    acq_function=model,
    bounds=bounds,
    q=3,
    options={
        "candidate_set": candidate_set,
        "replacement": False,
    },
)
```

`options`:

- `n_candidates`: Sobol候補集合の点数
- `candidate_set`: 明示的な候補集合 `[n_candidates, d]`
- `replacement`: 同じ候補を重複選択できるか
- `observation_noise`: 観測ノイズ込みposteriorを使うか
- `seed`: Sobol候補生成seed
- `constraint_tolerance`: 線形制約の許容誤差
- `duplicate_tolerance`: 重複判定の許容誤差

`n_candidates`を省略すると、次の値が使われます。

```python
max(num_restarts * raw_samples, 1024)
```

### `optimize_thompson_sampling_mixed`

mixed入力向けです。各カテゴリ組合せについてSobol候補集合を生成し、まとめて
`MaxPosteriorSampling`へ渡します。

```python
from bochan.optim import optimize_thompson_sampling_mixed

fixed_features_list = [
    {4: 0.0, 5: 0.0},
    {4: 0.0, 5: 1.0},
    {4: 1.0, 5: 0.0},
    {4: 1.0, 5: 1.0},
]

candidates, values = optimize_thompson_sampling_mixed(
    acq_function=acqf,
    bounds=bounds,
    fixed_features_list=fixed_features_list,
    q=3,
    num_restarts=5,
    raw_samples=128,
    options={
        "n_candidates": 4096,
        "replacement": False,
        "observation_noise": False,
        "seed": 42,
    },
    post_processing_func=repair,
    inequality_constraints=inequality_constraints,
    equality_constraints=equality_constraints,
)
```

`candidate_set`を指定した場合は、その候補集合を各`fixed_features_list`の組合せへ
複製してカテゴリ値を固定します。

### Thompson samplingの制限

初期実装では以下に対応していません。

- `return_best_only=False`
- 非線形制約
- inter-point線形制約
- 内部で使われたThompson sample値の返却

制約・step丸め・k-sparseを使う場合は、`post_processing_func`で候補集合を補修した後、
線形制約の再検査と重複除去を行います。

---

## 6. post-processing の共通例

```python
from bochan.constraints.postprocess import (
    make_grid_k_sparse_post_processing_func,
)

repair = make_grid_k_sparse_post_processing_func(
    bounds=bounds,
    numeric_indices=[0, 1, 2, 3],
    steps=torch.tensor([0.1, 0.1, 0.05, 0.05], dtype=bounds.dtype),
    comp_idx=[0, 1, 2, 3],
    k=2,
    equality_constraints=equality_constraints,
    inequality_constraints=inequality_constraints,
)
```

作成した`repair`は、各optimizerの`post_processing_func`へ渡せます。

```python
candidates, values = optimize_thompson_sampling(
    acq_function=model,
    bounds=bounds,
    q=3,
    post_processing_func=repair,
)
```

## 7. optimizerの使い分け

| optimizer | 向いている用途 |
|---|---|
| `optimize_acqf`系 | 滑らかで勾配が安定した獲得関数 |
| `optimize_acqf_*_k_sparse` | 使用変数数を制限したい場合 |
| `optimize_acqf_evo*` | 非滑らか、丸め、カテゴリ、複雑な補修 |
| `optimize_acqf_torch*` | optimizerや学習率を細かく制御したい場合 |
| `optimize_acqf_nsgaii` | 多目的・Pareto候補集合 |
| `optimize_thompson_sampling*` | posterior sampleによる探索、DeepGP、バッチ候補 |

## 注意

各optimizerの詳細な引数は、実装のdocstringまたは`help()`でも確認できます。

```python
from bochan.optim import optimize_thompson_sampling

help(optimize_thompson_sampling)
```
