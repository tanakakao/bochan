# 43. 結晶構造探索：生成・緩和・評価・Bayesian Optimization

組成が決まっても結晶構造は一意ではありません。同じ組成でも原子配置、格子、space group、polymorphによってenergyやpropertyが変わります。したがって結晶構造探索はcomposition optimizationより高次元で複雑な問題です。

## 1. 構造表現

結晶構造を概念的に

```math
S=(Z,R,L)
```

と表します。`Z` は元素種、`R` は原子座標、`L` は格子です。

propertyは

```math
y=f(S)
```

またはprocessを含め

```math
y=f(S,p)
```

となります。

## 2. 構造探索と構造緩和は別

structure relaxationは与えられた初期構造 `S_0` からlocal minimumを探します。

```math
S^*=\arg\min_S E(S)
```

一方structure discoveryは、どのinitial topology / configurationを試すかというglobal searchです。

```text
structure generation -> local relaxation -> candidate comparison
```

relaxation optimizerだけでは新しい結晶構造候補を網羅的に生成できません。

## 3. Candidate source

候補構造はdatabase、prototype substitution、enumeration、random structure generation、generative model等から得られます。

bochanではcandidate generationそのものとcandidate selectionを分離する方が明確です。

## 4. Invariance / equivariance

結晶propertyは座標系の任意なtranslationやrotationで不自然に変わるべきではありません。forceはrotationに応じて変換されるequivariant quantityです。

このためstructure modelではgraph neural networkやequivariant representationが重要になります。

## 5. Relaxation with MLIP

MLIPがenergy `E(S)` とforce

```math
F_i=-\frac{\partial E}{\partial r_i}
```

を予測できれば、FIRE/BFGS/LBFGS等でrelaxationできます。

```text
initial structure
 -> MLIP energy/forces
 -> optimizer
 -> relaxed structure
```

bochanのMACE / CHGNet / M3GNet / ALIGNN-FF relaxation workflowはこの層に位置します。

## 6. Relax -> Rank

最も単純なworkflowは、全候補をrelaxして予測propertyでrankする方法です。

```text
candidate bank -> relaxation -> predicted score -> ranking
```

候補数が有限でMLIP evaluationが十分安い場合に有効です。

## 7. Relax -> Acquisition

高精度評価が高価なら、relaxed structuresにGP posteriorを構築してacquisitionで次候補を選びます。

```text
candidate bank
 -> MLIP relaxation
 -> representation
 -> GP posterior
 -> acquisition
 -> DFT / experiment
```

posterior mean rankingとacquisition selectionは異なります。後者はuncertaintyを利用して探索と活用を調整します。

## 8. Structure + Process

実際の材料propertyは結晶構造だけでなくprocess historyにも依存します。

```math
y=f(S,p)
```

したがってstructure index/embeddingとprocess variablesを結合したsurrogateを作ることができます。

## 9. Discrete vs continuous structure search

既存structure bankから選ぶならfinite/discrete optimizationです。原子座標や格子を直接動かす場合はcontinuousですが、permutation symmetryやvalidity constraintがあり単純なbox BOには向きません。

実務では

```text
generator / physics -> valid structure candidates
BO -> expensive-evaluation candidate selection
```

という分業が扱いやすいです。

## 10. Multi-objective structure discovery

energyだけでなくband gap、elastic property、stability、cost等を同時に扱えます。energy stabilityをconstraint、functional propertyをobjectiveとする構成も自然です。

## 11. Uncertaintyの役割

pretrained MLIPの低energy predictionだけで候補を選ぶとdomain shiftで誤る可能性があります。GP residualやensemble uncertainty、AL criterionを使うことでhigh-fidelity evaluationの配分を改善できます。

## 12. bochanでの位置付け

```text
structure source
  -> structure relaxation
  -> representation / baseline
  -> GP / residual GP
  -> BO / AL acquisition
  -> expensive evaluator
```

bochanは特にrelaxation以降のsurrogate decision layerを担い、structure generatorやDFT engineとは疎結合に保つのが適切です。

結晶構造探索では、**生成・緩和・property prediction・uncertainty・candidate selectionを別の問題として設計すること**が重要です。
