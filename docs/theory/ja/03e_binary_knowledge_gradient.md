# Binary Knowledge Gradient

`qBinaryKnowledgeGradient` は、binary classification の **離散観測 likelihood を考慮した one-step look-ahead acquisition** です。

## 目的

Binary BO では最終的に、指定クラスの予測確率が最大の条件を選ぶことを考えます。既定では class 1 を対象とします。

```math
V(D)=\max_x P(y=1\mid x,D)
```

候補 `x` を1回測定すると観測は `y=0` または `y=1` です。Binary KG はそれぞれの仮想ラベル後の terminal value を評価し、現在との差を計算します。

```math
KG(x)=\sum_{y\in\{0,1\}}P(y\mid x,D)V(D\cup\{(x,y)\})-V(D)
```

## 実装

回帰用 `qKnowledgeGradient` の Gaussian fantasy を分類へ流用しません。代わりに、候補と terminal decision set 上で coherent な latent posterior function samples を生成し、仮想 Bernoulli label の likelihood で function samples を Bayes 再重み付けします。

このため、各候補ごとのモデル再学習を行わずに one-step discrete-observation KG を評価できます。

## 利用例

```python
from bochan.acquisition.binary.bayesian_optimization import qBinaryKnowledgeGradient
from bochan.api import AcquisitionConfig, DataContext, OptimizeConfig

acq_config = AcquisitionConfig(
    name="binary_kg",
    acqf_cls=qBinaryKnowledgeGradient,
    acqf_kwargs={
        "bounds": bounds,
        "target_class": 1,
        "terminal_size": 128,
        "num_samples": 64,
    },
)

X_next, value = optimizer.candidate(
    acq_config,
    OptimizeConfig(q=1),
    data_context=DataContext(),
)
```

`terminal_set` を明示すれば、離散・カテゴリを含む探索空間でも valid な最終意思決定集合を直接与えられます。

## v1 の範囲

- single-output binary classification
- q=1
- target class 0 または 1
- continuous input は bounds から Sobol terminal set を自動生成可能
- mixed/categorical input は terminal set の明示指定を要求
- pending label の近似は行わない

複数候補を一度に返す場合に pending label を既知値として扱うことは、Binary KG の意味を変えてしまいます。そのため v1 では1点ずつ観測して refit する契約を明示しています。
