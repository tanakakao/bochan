# 36. Information-theoretic Bayesian Optimization

Information-theoretic BOは、objective improvementそのものではなく、最適解や最適値に関する不確かさを減らすcandidateを選びます。

## 1. 情報量としてのacquisition

未知量を `Z` とすると、candidate `x` の観測 `y_x` が与える情報量は

```math
I(y_x;Z\mid D)=H(Z\mid D)-E_{y_x}[H(Z\mid D,y_x)]
```

です。

`Z` を何にするかで手法が変わります。

## 2. PES

Predictive Entropy Searchは主にoptimizer location `x*` のentropy reductionを狙います。

```math
Z=x^*
```

「どこが最適か」を学ぶことが中心です。

## 3. MES

Max-value Entropy Searchは最適値

```math
f^*=\max_x f(x)
```

に関するentropy reductionを狙います。optimizer location全体よりscalarな最大値を扱うため計算上有利になる場合があります。

## 4. JES

Joint Entropy Searchは

```math
Z=(x^*,f^*)
```

として最適位置と最適値をjointに学習します。

## 5. KGとの違い

KGは最終decision valueの改善を測るvalue-of-information、entropy searchは未知量に関するinformation gainを測ります。

```text
EI  : objective improvement
KG  : decision value improvement
PES : optimizer information
MES : optimum-value information
JES : joint optimizer/value information
```

## 6. Active Learningとの境界

ALのBALDもmutual informationを使いますが、通常はmodel parameterやprediction uncertaintyを減らすことが目的です。

Information-theoretic BOは**optimumに関係する情報**へ焦点を絞る点が異なります。

## 7. Multi-objective

Pareto set/frontを未知量としてentropy reductionを考えるmulti-objective entropy searchも可能です。ただしposterior sampling、Pareto computation、conditional distribution approximationが必要になり計算量は大きくなります。

## 8. 実務的な選択

Information-theoretic acquisitionは魅力的ですが、approximation errorと計算costがあります。まずEI/NEI/UCBをbaselineとし、探索効率が課題になった場合に比較するのが安全です。

特にcandidate poolが有限、evaluationが非常に高価、optimum locationの同定自体が重要な問題で価値があります。

bochanではMES/PES/JESを単に名称で並べるだけでなく、**何についてentropyを減らしているか**をAPI・documentation上で明示することが重要です。
