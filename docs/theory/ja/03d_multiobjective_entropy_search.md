# 03d. 多目的 Entropy Search（MO-MES / MO-JES）

bochan は BoTorch の Pareto 多目的 Entropy Search を高位 API から利用できます。

- MO-MES: `qLowerBoundMultiObjectiveMaxValueEntropySearch`
- MO-JES: `qLowerBoundMultiObjectiveJointEntropySearch`

単目的 `mes` / `jes` を multi-output posterior に対して scalarize する経路とは別機能です。MO-MES / MO-JES は複数のモデル出力を Pareto 目的として保持したまま情報獲得を行います。

## 1. 使い分け

| 問題 | 推奨 acquisition |
|---|---|
| 単目的の最適値について学ぶ | MES |
| 単目的の最適入力・最適値について学ぶ | JES |
| Pareto 最適出力について学ぶ | MO-MES / MESMO |
| Pareto 最適入力・出力について学ぶ | MO-JES |
| 将来の Pareto hypervolume を改善する | HVKG |

MO-MES は候補観測と Pareto optimal outputs の相互情報量を評価します。MO-JES は Pareto optimal input-output pairs まで学習対象に含めます。

## 2. API 名

### MO-MES

```python
AcquisitionConfig(name="mo_mes")
AcquisitionConfig(name="mesmo")
```

次の alias も同じ BoTorch class に解決されます。

```text
qmo_mes
qmesmo
multi_objective_mes
qmulti_objective_mes
```

### MO-JES

```python
AcquisitionConfig(name="mo_jes")
```

次の alias も利用できます。

```text
qmo_jes
multi_objective_jes
qmulti_objective_jes
```

## 3. Pareto sample の自動生成

MO-MES / MO-JES の補助量を省略した場合、bochan は BoTorch の公開 utility を使います。

```text
model + bounds
    ↓
sample_optimal_points
    ↓
pareto_sets / pareto_fronts
    ↓
compute_sample_box_decomposition
    ↓
hypercell_bounds
```

BoTorch の `sample_optimal_points` は posterior path を生成し、それぞれの path を最適化して Pareto set / front を近似します。bochan はこのアルゴリズムを再実装しません。

既定値は次の通りです。

```python
DataContext(
    bounds=bounds,
    extra={
        "mo_entropy_num_pareto_samples": 8,
        "mo_entropy_num_pareto_points": 8,
        "mo_entropy_num_samples": 64,
        "mo_entropy_estimation_type": "LB",
    },
)
```

利用可能な entropy estimate は BoTorch に合わせて `"0"`, `"LB"`, `"LB2"`, `"MC"` です。

Pareto path の内部最適化を調整する場合は、次を利用できます。

```python
DataContext(
    bounds=bounds,
    extra={
        "mo_entropy_optimizer_kwargs": {
            "pop_size": 1024,
            "max_tries": 10,
        },
    },
)
```

高度な利用では `mo_entropy_optimizer` に BoTorch `sample_optimal_points` と互換な optimizer callable を渡せます。

## 4. MO-MES

MO-MES が acquisition constructor に必要とする主な補助量は `hypercell_bounds` です。

通常は自動生成されます。

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_mes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

既に適切な box decomposition を持っている場合は明示できます。

```python
AcquisitionConfig(
    name="mo_mes",
    acqf_kwargs={"hypercell_bounds": hypercell_bounds},
)
```

明示した `hypercell_bounds` は再生成されません。この場合、補助量生成用の `bounds` は不要です。

## 5. MO-JES

MO-JES は次の3つを必要とします。

- `pareto_sets`
- `pareto_fronts`
- `hypercell_bounds`

省略時はすべて自動生成されます。

```python
X_next, value = optimizer.candidate(
    AcquisitionConfig(name="mo_jes"),
    OptimizeConfig(q=1),
    data_context=DataContext(bounds=bounds),
)
```

Pareto samples を明示し、box decomposition だけ bochan に計算させることもできます。

```python
AcquisitionConfig(
    name="mo_jes",
    acqf_kwargs={
        "pareto_sets": pareto_sets,
        "pareto_fronts": pareto_fronts,
    },
)
```

`pareto_sets` と `pareto_fronts` は必ず対応する pair として指定します。

`hypercell_bounds` も明示する場合は、同じ Pareto samples から作られたものを指定する必要があります。そのため、bochan は `hypercell_bounds` だけを指定して Pareto samples を自動生成する曖昧な経路を許可しません。

## 6. 目的の意味論

MO-MES / MO-JES は scalar objective を受け取る acquisition ではありません。モデルの複数出力そのものが Pareto objectives です。

したがって次は拒否されます。

- `AcquisitionConfig.objective`
- `objective_config`
- `objective_factory`
- `posterior_transform`
- `MultiObjectiveConfig` の自動 scalarization

複数出力を1つの utility に集約して情報獲得したい場合は、MO-MES / MO-JES ではなく通常の MES / JES を使います。

```text
multi-output → scalar utility → MES / JES
```

Pareto 構造を保持したい場合は、

```text
multi-output → Pareto objectives → MO-MES / MO-JES
```

です。

## 7. 最大化・最小化

BoTorch の Pareto sampling / box decomposition は、全 objectives を同じ向きで扱う `maximize` flag を持ちます。bochan では既定値を最大化としています。

```python
DataContext(
    bounds=bounds,
    extra={"mo_entropy_maximize": True},
)
```

全 objectives を最小化する場合は `False` にできます。

```python
extra={"mo_entropy_maximize": False}
```

一部を最大化・一部を最小化する mixed direction は、この native MO entropy path では自動変換しません。必要な場合は、モデル出力を事前に「すべて最大化」または「すべて最小化」に揃える transform を用意してください。

## 8. q > 1

BoTorch の lower-bound MO-MES / MO-JES は `q > 1` を評価できますが、lower bound は batch 要素追加に対して単調とは限りません。BoTorch の公式 information-theoretic tutorial は MO-MES / MO-JES の batch 候補を sequential greedy に最適化しています。

そのため bochan 高位 API では `q > 1` のとき自動的に `sequential=True` にします。

```python
OptimizeConfig(q=3)
```

は MO-MES / MO-JES では sequential candidate generation に正規化されます。

## 9. 現在の正式サポート範囲

自動 Pareto sampling の正式な中心範囲は、BoTorch `sample_optimal_points` と互換な Gaussian GP 系の連続入力・多目的回帰です。

### 自動生成

- 2出力以上
- Gaussian GP compatible model
- continuous bounds
- unconstrained Pareto objectives
- objectives は全て同じ最適化方向

### mixed / categorical input

BoTorch の標準 `sample_optimal_points` の既定 optimizer は連続 bounds 上で Pareto points を探索するため、bochan は mixed / categorical input での自動 Pareto sampling を行いません。

高度な利用では補助量を明示できます。

- MO-MES: `hypercell_bounds`
- MO-JES: `pareto_sets`, `pareto_fronts`, `hypercell_bounds`

候補点そのものの最適化は、通常の bochan mixed optimizer 設定を利用できます。

## 10. 制約付き多目的

BoTorch の box-decomposition utility 自体には constraint を表現する機能がありますが、objective / constraint output の分離や Pareto sampling まで含む一貫した高位 API が必要です。

このPRでは constraints を暗黙に無視せず、MO-MES / MO-JES の自動経路では明示的に reject します。制約付きMO entropy searchは別途、constraint outputs を含めた正式な設計として追加します。

## 11. 計算コスト

MO-MES / MO-JES は通常の EHVI 系より補助計算が重くなります。各 iteration で複数 posterior paths を生成し、それぞれについて Pareto optimization と box decomposition を行うためです。

物理実験が高価で、単純な即時 improvement より「次の実験が Pareto front についてどれだけ情報を与えるか」を重視したい場合に適しています。
