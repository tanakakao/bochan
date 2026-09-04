# 33. Multi-objective Bayesian Optimization 実践編

多目的BOでは単一の最良点ではなく、互いにtrade-offを持つPareto setを探索します。本章は07章の理論を補完し、手法選択と実装上の判断を中心に整理します。

## 1. Pareto dominance

最大化へ向きを統一したobjective vectorを `f(x)` とします。`x_a` が `x_b` をdominatesするとは、全objectiveで悪くなく、少なくとも1つで良いことです。

Pareto frontはdominateされないobjective vectorsの集合です。

## 2. Hypervolume

reference point `r` とPareto frontが支配する体積をhypervolumeと呼びます。

```math
HV(P;r)=\lambda\left(\bigcup_{y\in P}[r,y]\right)
```

reference pointは「十分に悪い点」である必要があり、選び方でhypervolumeの意味が変わります。

## 3. qEHVI

noiseが小さい場合、candidate batchによるexpected hypervolume improvementを評価します。

```math
\alpha_{EHVI}(X)=E[HV(P\cup f(X))-HV(P)]
```

複数candidateをjointに選ぶ `q` formulationでは候補間のposterior correlationも考慮されます。

## 4. qNEHVI

観測noiseがある場合、観測済みPareto front自体が不確かです。qNEHVIはbaseline uncertaintyを考慮するため、実験dataでは重要なdefault候補です。

`X_baseline`、pending points、pruningなどの設定が計算量と安定性へ影響します。

## 5. qNParEGO

objectiveをrandom scalarizationしてsingle-objective acquisitionへ変換する考え方です。

```math
u_\lambda(y)=\min_j \lambda_j(y_j-z_j)
```

などのscalarizationを用い、iterationごとにtrade-off方向を変えることでPareto frontを探索します。

objective数が増えhypervolume法が重くなる場合の有力候補です。

## 6. Scaling

例えば強度が1000単位、costが1単位なら、scaleを無視したscalarizationは大きいscaleのobjectiveに支配されます。

```text
raw outputs
 -> direction normalization
 -> scaling / standardization
 -> objective transform
 -> acquisition
```

の順序を明確にします。

## 7. Multi-outputとMulti-objective

複数outputがあっても全てをobjectiveにする必要はありません。

```text
output A -> objective
output B -> objective
output C -> constraint
output D -> diagnostic only
```

という構成も可能です。MultiTaskGPも必須ではなく、独立posteriorでもmulti-objective acquisitionは構築できます。

## 8. Constraintとの統合

制約付きmulti-objective BOではfeasible Pareto frontを改善するcandidateを探索します。32章のPoF/outcome constraintと組み合わせます。

## 9. 選択指針

| 状況 | 第一候補 |
|---|---|
| 2〜3目的、noise小 | qEHVI |
| 2〜3目的、noiseあり | qNEHVI |
| objective数が多い | qNParEGO |
| preferenceが明確 | scalarized single-objective BO |
| feasibilityあり | constrained EHVI/NEHVI等 |

## 10. 材料探索

典型例は

```text
maximize performance
maximize stability
minimize cost
minimize processing temperature
```

です。すべてをobjectiveにするとPareto setが巨大になるため、明確な規格値があるものはconstraintへ移す判断も重要です。

Multi-objective BOの目的は「全部最大化する1点」を作ることではなく、**意思決定に必要なtrade-off frontを少ない評価で得ること**です。
