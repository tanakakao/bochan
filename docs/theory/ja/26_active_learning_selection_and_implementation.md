# 26. Active Learningの選択基準と実装対応

Active Learning（AL）は、最良候補を直接探すBayesian Optimizationとは目的が異なります。ALの目的は、追加観測によってモデル、境界、または意思決定対象に関する不確かさを減らすことです。

## 1. 何の不確かさを減らしたいか

AL criterionを選ぶ前に、learning targetを明示します。

- 点ごとの予測誤差を減らしたい
- 空間全体のposterior uncertaintyを減らしたい
- classification boundaryを学びたい
- model parameter uncertaintyを減らしたい
- feasibility boundaryを学びたい

同じ「不確かな点を選ぶ」でも目的が違えばcriterionは変わります。

## 2. Posterior variance

Gaussian regressionでは最も直接的なuncertainty criterionです。

```math
\alpha_{\mathrm{var}}(x)=\mathrm{Var}[f(x)\mid\mathcal D]
```

**使う場面**

- 単純なuncertainty sampling
- GPの未観測領域を埋めたい
- baselineとしてAL performanceを比較したい

**注意点**

高variance点が必ずしもglobal model improvementへ最も寄与するとは限りません。局所的に孤立した領域を過度に選ぶ場合があります。

## 3. Predictive entropy

classificationで予測クラスが不確かな点を選びます。

binary probability `p(x)` なら

```math
H[y\mid x,\mathcal D]
=-p\log p-(1-p)\log(1-p)
```

です。`p=0.5`付近で最大になります。

multiclassでは

```math
H[y\mid x,\mathcal D]
=-\sum_k p_k(x)\log p_k(x)
```

です。

**使う場面**

- classification boundary周辺を観測したい
- predictive ambiguityを減らしたい

ただしpredictive entropyはaleatoric uncertaintyとepistemic uncertaintyを区別しません。

## 4. BALD

BALDは予測labelとmodel parameterのmutual informationを評価します。

```math
I(y;\theta\mid x,\mathcal D)
=H[y\mid x,\mathcal D]
-\mathbb E_{p(\theta\mid\mathcal D)}
[H[y\mid x,\theta]]
```

**意味**

- 左項: 予測全体の不確かさ
- 右項: modelを固定しても残るdata uncertainty
- 差: model uncertaintyに由来する情報量

そのため、単にlabelが曖昧な点ではなく「観測するとmodel beliefが変わる点」を選びやすくなります。

**使う場面**

- epistemic uncertaintyを重視
- classification AL
- ensemble / posterior sampleを利用可能

## 5. Margin uncertainty

multiclass classificationで上位2classの確率差を使います。

```math
\alpha_{\mathrm{margin}}(x)
=-(p_{(1)}(x)-p_{(2)}(x))
```

差が小さいほどclass decisionが不確かです。

entropyより計算が単純で、class boundary explorationのbaselineとして使いやすいcriterionです。

## 6. Latent straddle

threshold `h` の境界を学びたい場合、meanのthreshold距離とuncertaintyを組み合わせます。

```math
\alpha_{\mathrm{straddle}}(x)
=\beta\sigma(x)-|\mu(x)-h|
```

threshold付近かつuncertaintyが大きい点を選びます。

**使う場面**

- level-set estimation
- pass/fail boundary
- specification limit探索

通常のvariance samplingよりboundaryへ集中します。

## 7. Integrated uncertainty / NIPV

局所点だけでなく、candidateを観測した後に空間全体のposterior varianceがどれだけ減るかを評価する考え方です。

概念的には

```math
\alpha(x)
=-\int_{\mathcal X}
\mathrm{Var}_{t+1}[f(x')\mid y_x]\,dx'
```

を最大化します。

実装では積分をMC pointsやfinite reference setで近似します。

**使う場面**

- 空間全体のsurrogateを改善したい
- 後段で複数箇所のpredictionを使う
- global uncertainty reductionが目的

**注意点**

reference pointsの分布が実質的な重み付けになります。探索対象domainを正しく表すpoint setが必要です。

## 8. Boundary variance

classification probability `p(x)`のvarianceやlatent boundary uncertaintyを利用し、decision boundary近傍を優先するcriterionがあります。

単純entropyと違い、posterior sample間の変動を利用する場合にはepistemic boundary uncertaintyを反映できます。

## 9. Batch Active Learning

`q>1`で単純にtop-q uncertaintyを選ぶと、互いにほぼ同じ候補が選ばれる場合があります。

batch criterionでは

```math
\alpha(X),\qquad X=\{x_1,\ldots,x_q\}
```

をjointに評価するか、sequential selection + pending conditioningを使います。

重要なのは

- diversity
- posterior correlation
- duplicate exclusion
- pending observations

です。

## 10. BOとALの使い分け

| 目的 | 適したcriterion |
|---|---|
| 最大値を探す | EI / NEI / UCB |
| classifierを改善 | entropy / BALD |
| threshold boundary | straddle / boundary variance |
| global surrogate改善 | variance / NIPV |
| Pareto optimum探索 | EHVI / NEHVI |

「探索的に見える」という理由だけでUCBとALを同一視しないことが重要です。UCBはutility optimization、varianceやBALDはlearning objectiveです。

## 11. bochanのfamily構造

bochanではAL acquisitionもmodel semanticsごとに分離します。

```text
regression/
  active_learning/
binary/
  active_learning/
multiclass/
  active_learning/
ordinal/
  active_learning/
non_gaussian/
  active_learning/
```

これにより、同じentropyという名前でもcontinuous regression、binary classification、multiclass、ordinalで必要なposterior transformationを分離できます。

## 12. 実務向け選択表

| 状況 | 第一候補 |
|---|---|
| GP回帰の未観測領域 | posterior variance |
| binary boundary | entropy / BALD |
| multiclass ambiguity | entropy / margin |
| model uncertainty重視 | BALD |
| threshold判定 | latent straddle |
| 空間全体を改善 | NIPV |
| batchで類似点を避けたい | joint / sequential batch AL |

最初から複雑なcriterionを使うより、varianceまたはentropyをbaselineに置き、BALD/NIPVが実際のlearning curveを改善するか比較する方が堅実です。
