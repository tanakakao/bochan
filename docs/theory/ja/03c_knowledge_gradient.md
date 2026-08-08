# 03c. Knowledge Gradient (KG)

本章では `bochan` の高位APIから利用する Knowledge Gradient (KG) を整理します。
KG自体は以前からBoTorchの `qKnowledgeGradient` へalias接続されていましたが、
MES / JES / HVKG と同じ方針で、高位APIから必要な補助値を安全に自動解決する
経路を追加しています。

## 1. 位置づけ

KGは、候補点を観測した後に得られる「最終的な最適意思決定の価値」が、
現在よりどれだけ改善すると期待できるかを評価するlook-ahead型獲得関数です。

```python
AcquisitionConfig(name="kg")
AcquisitionConfig(name="qkg")
```

はBoTorchの `qKnowledgeGradient` に解決されます。

KGはone-shot optimizationを使います。候補点に加え、fantasy modelごとの
最適解を表す補助変数も同時に最適化します。

## 2. current_value の自動計算

`qKnowledgeGradient` の `current_value` は、観測済みデータだけから得られる
現在のexpected best objectiveです。

`current_value` を省略したBoTorch KGは、厳密なKG差分ではなく、候補を観測した後の
expected best objectiveそのものを返します。そのため `bochan` 高位APIでは、
`current_value` を明示しない場合は `DataContext.bounds` からBoTorchの
input constructorを使って自動計算します。

```python
acq = AcquisitionConfig(name="kg")
context = DataContext(bounds=bounds)
```

fantasy数の既定値は64です。

```python
context = DataContext(
    bounds=bounds,
    extra={"kg_num_fantasies": 32},
)
```

または獲得関数へ直接指定できます。

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={"num_fantasies": 32},
)
```

明示的な `acqf_kwargs["num_fantasies"]` が最優先です。

## 3. 明示 current_value

`current_value` を明示した場合は再計算しません。

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={
        "current_value": current_value,
        "num_fantasies": 32,
    },
)
```

この場合、`current_value` を求めるための `bounds` は不要です。

## 4. pending point

asynchronous BOで `X_pending` が存在する場合、KGの `current_value` はpending点を
条件付けしたterminal valueである必要があります。

BoTorchの通常の `construct_inputs_qKG(..., with_current_value=True)` は
pending-conditioned current valueを自動生成しないため、`bochan` は
`X_pending` がある状態での自動 `current_value` 計算を行いません。

この場合は明示的に `current_value` を渡します。

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={"current_value": pending_conditioned_current_value},
)
context = DataContext(X_pending=X_pending)
```

これはpending点を無視したcurrent valueを暗黙使用することを防ぐための安全側の仕様です。

## 5. 単目的回帰

もっとも基本的な利用対象は単出力の回帰モデルです。

```python
acq = AcquisitionConfig(name="kg")
opt = OptimizeConfig(q=1)

X_next, value = optimizer.candidate(
    acq,
    opt,
    data_context=DataContext(bounds=bounds),
)
```

objectiveを指定しない場合はposterior meanがterminal valueとして使われます。

## 6. multi-output回帰

multi-output modelへKGを適用する場合は、KGが最終的に比較するscalar terminal objectiveを
明示する必要があります。

方法は2つあります。

### 6.1 objectiveでscalar化

```python
acq = AcquisitionConfig(
    name="kg",
    objective=objective,
)
```

またはbochanの `objective_config` / `objective_factory` を利用できます。

### 6.2 posterior_transformでscalar posteriorへ変換

```python
acq = AcquisitionConfig(
    name="kg",
    acqf_kwargs={
        "posterior_transform": posterior_transform,
    },
)
```

multi-outputでどちらも指定されていない場合は、高位APIはエラーにします。
scalarizationを暗黙に推測しません。

## 7. KGとHVKGの違い

KGとHVKGはどちらもlook-ahead型ですが、目的が異なります。

| 問題 | 推奨 |
|---|---|
| 単目的・単出力回帰 | KG |
| multi-outputだが最終utilityは1つ | KG + objective / posterior_transform |
| 複数目的をPareto最適化 | HVKG |

例えば強度と靭性を重み付きutilityへまとめるならKGを使えます。
一方、強度と靭性を別々の目的としてPareto frontを改善するならHVKGを使います。

## 8. optimizer semantics

`qKnowledgeGradient` はone-shot acquisitionです。
したがって `OptimizeConfig.sequential=True` を指定しても、高位APIは
`sequential=False` へ正規化します。

```python
opt = OptimizeConfig(q=2, sequential=True)
# KGでは内部的に sequential=False へ正規化
```

BoTorchの `optimize_acqf` は `qKnowledgeGradient` を検出し、専用の
one-shot KG initializerを自動選択します。そのためbochan側で独自initializerを
複製しません。

## 9. task routing

short aliasの `kg` / `qkg` / `knowledgegradient` / `qknowledgegradient` は
regression系posteriorに限定します。

binary / multiclass / ordinalには標準KGを暗黙転送しません。これらのtaskでは
BALD、predictive entropy、margin uncertaintyなどのtask-specific acquisitionを
利用します。

## 10. KG / MES / JES / HVKG の整理

| 手法 | 主な対象 | 情報価値 | 最適化 |
|---|---|---|---|
| KG | 単目的 | 観測後の最終意思決定価値 | one-shot |
| MES | 単目的 | 最適値の情報 | q>1はsequential |
| JES | 単目的 | 最適点 + 最適値の情報 | 通常のjoint/q設定 |
| HVKG | 多目的 | 将来Pareto / hypervolume価値 | one-shot |

材料開発で1回の実験コストが高く、単なる直近改善量よりも「次の実験が最終判断を
どれだけ良くするか」を重視する場合、KGは有力な選択肢です。
