# 39. Mixed / Discrete / Combinatorial Bayesian Optimization

材料・工程探索ではcontinuous variableだけでなく、元素、装置、原料、処理method、integer countなどが混在します。この場合、continuous BOのacquisition optimizerをそのまま使うことはできません。

## 1. Mixed search space

```math
x=(x_{cont},x_{int},x_{cat},x_{comb})
```

のようなheterogeneous spaceを考えます。

例:

```text
temperature : continuous
time        : continuous
cycle count : integer
atmosphere  : categorical
elements    : combinatorial subset
```

## 2. Categorical variables

category番号を単純な連続値として扱うと、category間に偽の順序・距離を導入します。categorical-aware kernel/modelやone-hot representation、mixed GPなどを検討します。

BoTorchのmixed optimizationではcategorical/fixed feature combinationsごとにcontinuous acquisition optimizationを行う考え方があります。

## 3. Integer variables

roundingは簡単ですが、training representation、acquisition optimization、post-processingのどこで整数性を保証するかを統一する必要があります。

candidate提案後だけroundするとduplicateやacquisition mismatchが生じる場合があります。

## 4. Finite candidate set

候補が既知の有限集合

```math
X_{pool}=\{x_1,\ldots,x_M\}
```

なら、各candidateのacquisitionを評価してdiscrete optimizationできます。結晶構造bankや既存材料databaseの選択に適します。

## 5. Combinatorial selection

元素集合 `E` から `k` 種選ぶ場合

```math
\mathcal S_k=\{S\subseteq E:|S|=k\}
```

を探索します。候補数は

```math
|\mathcal S_k|={|E|\choose k}
```

で急増します。

## 6. 元素選択と組成比

重要なのは

```text
which elements?   -> combinatorial variable
what fractions?   -> constrained continuous variable
```

を分けることです。

元素subset `S` を選んだ後、

```math
c_j\ge0,\qquad\sum_{j\in S}c_j=1
```

のsimplex上でcomposition ratioを最適化できます。

## 7. Hierarchical optimization

```text
element subset
   ↓
composition ratios
   ↓
process conditions
```

という階層構造を持たせると、無効な組合せを避けやすくなります。

一方、すべてを1つの巨大なvectorへ埋め込む方式は柔軟ですがsearch dimensionが増えます。

## 8. Constraints

組合せ探索では

- 必須元素
- 禁止元素pair
- 最大元素数
- charge balance
- composition closure
- process compatibility

などのconstraintが重要です。候補生成段階で除外できるdeterministic constraintsは早期に適用します。

## 9. Surrogate representation

subsetをbinary vectorだけで表す方法に加え、CrabNet/Roost等のcomposition encoderを使ってmaterial similarityを表現できます。

```math
z=h(c)
```

としてGP/DKLへ接続すれば、単なるelement ID距離より有用なrepresentationを学習できる可能性があります。

## 10. bochanでの位置付け

bochanのcomposition探索では

```text
candidate generation
 -> combinatorial filtering
 -> composition representation
 -> surrogate posterior
 -> acquisition
 -> candidate selection
```

を分離すると、現在のenumeration型探索と将来のcombinatorial optimizerを共存させやすくなります。

Mixed BOではmodelだけでなく、**search-space representationとacquisition optimizerの整合性**が重要です。
