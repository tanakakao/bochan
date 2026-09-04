# 34. Batch / Parallel Bayesian Optimization

実験装置やsimulation resourceを並列利用できる場合、1回に1点ではなく `q>1` のcandidate batchを選びます。

## 1. SequentialとBatch

逐次BOは

```math
x_{t+1}=\arg\max_x\alpha(x)
```

です。Batch BOでは

```math
X_{t+1}=\{x_1,\ldots,x_q\}
```

を同時に選びます。

単純にsingle-point acquisitionの上位q点を取ると、似たcandidateが集中しやすい問題があります。

## 2. q-acquisition

Monte Carlo qEIなどはbatch全体のutilityを評価します。

```math
qEI(X)=E[(\max_{x\in X}f(x)-f_{best})_+]
```

候補間のposterior correlationを考慮できるため、batch内の冗長性を抑えられます。

## 3. Joint optimization

`q` pointsを一つの高次元variableとして同時最適化します。

利点はbatch utilityを直接最適化できること、欠点は `q*d` 次元のoptimizationになり計算が難しくなることです。

## 4. Sequential batch construction

batchを1点ずつ構築し、既に選んだcandidateを条件に次を選ぶ方法です。

```text
select x1
 -> condition/fantasize
select x2
 -> ...
select xq
```

joint optimizationより軽量な場合があります。

## 5. Pending observations

既に実験開始済みだが結果が未取得の点を `X_pending` として扱います。これを無視すると同じ領域を重複して提案する可能性があります。

asynchronous BOでは特に重要です。

## 6. Fantasy model

pending pointの未知の結果をposteriorから仮想的に生成し、観測後のposteriorを平均化して次のdecisionを評価します。

```math
y_{pending}^{(s)}\sim p(y\mid X_{pending},D)
```

これがKnowledge Gradientやlookaheadにもつながります。

## 7. Diversity

batch diversityは単なるEuclidean distanceではありません。posterior correlationが高い点は情報的に冗長な場合があります。

実務上は

- q-acquisition
- minimum distance
- duplicate exclusion
- categorical diversity
- structure diversity

などを組み合わせます。

## 8. BOとActive Learning

AL batchでも同じ問題があります。posterior variance上位点だけを取ると近接点が集中するため、joint information gainやsequential conditioningが有効です。

## 9. 実務的選択

| 状況 | 方針 |
|---|---|
| qが小さく計算余力あり | joint q-acquisition |
| qが大きい | sequential construction |
| 非同期実験 | X_pendingを必ず反映 |
| finite candidate pool | discrete batch optimization |
| expensive structure evaluation | diversityも明示的に確認 |

Batch BOの本質は「q個の良い点」ではなく、**q個をまとめて評価したときに価値の高い集合**を選ぶことです。
