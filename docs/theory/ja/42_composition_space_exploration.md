# 42. 組成空間探索：元素選択・組成比・プロセス条件

材料探索では「どの元素を使うか」と「各元素を何%にするか」は異なる最適化問題です。さらに熱処理温度などのprocess variableを同時に扱うと、探索空間はdiscrete・continuous・conditionalな変数が混在します。

## 1. 組成の基本制約

K成分の組成 `c=(c_1,...,c_K)` は通常

```math
c_j\ge0,\qquad \sum_{j=1}^K c_j=1
```

を満たすsimplex上の点です。通常のbox constraintだけで探索すると物理的に無効な組成を生成します。

## 2. 元素選択と組成比を分ける

候補元素集合からk元素を選ぶ問題はcombinatorialです。

```math
S\subseteq E,\qquad |S|=k
```

選ばれた元素集合 `S` の内部で組成比を最適化します。

```text
元素subset selection
       ↓
composition ratio optimization
       ↓
process optimization
```

この階層化により、元素の有無と濃度の意味を混同しにくくなります。

## 3. Representation

組成式からpropertyを予測する場合、単純な元素fraction vectorのほか、element descriptor aggregation、CrabNet、Roost等のlearned representationを利用できます。

```text
composition -> encoder -> latent z -> predictor / GP
```

GPをlatent representation上に置けば、組成式を直接Euclidean vectorとして扱うより意味のあるsimilarityを得られる場合があります。

## 4. CLR / ILR

compositional dataではclosureにより成分が独立ではありません。正の組成に対してlog-ratio transformを使えます。

CLRは

```math
z_j=\log\frac{c_j}{g(c)}
```

です。ILRはsimplexを `K-1` 次元のorthonormal coordinateへ写します。

ただしzero compositionを含む元素選択問題では、そのままlog-ratio transformを適用できないため、subset selectionとratio optimizationを分離する設計が有効です。

## 5. Process variablesとの統合

材料propertyを

```math
y=f(c,p)
```

と考えます。`c` が組成、`p` が温度・時間・圧力等です。

同じ組成でもprocessによってpropertyが変わるため、組成だけを最適化した結果が実験最適条件とは限りません。

## 6. Conditional search space

元素を選ばなければその濃度は0です。このような条件付き構造を無視して全元素fractionを独立変数にすると探索効率が低下します。

```text
selected element? -- no --> fraction = 0
                  -- yes -> optimize fraction
```

## 7. Candidate generation

実務では次の方式を使い分けます。

- finite composition libraryから選択
- simplex上でcontinuous optimization
- grid / lattice composition
- subsetを列挙して内部でBO
- combinatorial optimizerとcontinuous optimizerの階層化

## 8. Multi-objective composition discovery

例えばstrength最大化とcost最小化を同時に扱えばPareto探索になります。さらに元素価格・毒性・供給risk等をconstraintへ入れられます。

## 9. Robust composition optimization

調合誤差がある場合は

```math
\tilde c=c+\xi
```

を考えます。ただしperturbation後もsimplex制約を満たす必要があります。renormalizationやlog-ratio spaceでのperturbationが候補です。

## 10. bochanでの位置付け

bochanでは組成探索を

```text
composition representation
    × subset selection
    × ratio optimization
    × process variables
    × surrogate model
    × acquisition
```

として分離すると拡張しやすくなります。

CrabNet/Roost系は組成representation、best-subset系は元素選択、BoTorch acquisitionは次候補決定という異なる責務を持ちます。

## 11. 推奨workflow

```text
candidate elements
  -> element subset
  -> feasible composition
  -> composition encoder
  -> GP / DKL
  -> acquisition
  -> process-condition optimization
  -> experiment
  -> update
```

組成探索の核心は、単に元素fractionを連続変数として扱うのではなく、**元素選択・simplex geometry・材料representation・process条件をそれぞれ正しくmodel化すること**です。
