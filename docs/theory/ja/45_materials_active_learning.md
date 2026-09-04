# 45. Materials Active Learning：何を測ればモデルが最も賢くなるか

Bayesian Optimizationが主に良い材料を見つけることを目的とするのに対し、Active Learning (AL) は**次にどのdataを取得すればmodel uncertaintyを最も効率よく減らせるか**を考えます。

材料開発では高価なDFT・実験dataをどこへ配分するかという問題に直結します。

## 1. BOとAL

BO:

```math
x_{next}=\arg\max_x \text{expected optimization utility}
```

AL:

```math
x_{next}=\arg\max_x \text{expected information utility}
```

高性能候補を見つけることと、材料空間全体を理解することは同じではありません。

## 2. Posterior variance

GP回帰で最も単純なAL criterionはposterior varianceです。

```math
\alpha(x)=\operatorname{Var}[f(x)\mid D]
```

未観測領域を広く探索するbaselineとして有効です。

## 3. Classification entropy

材料の合否、相安定/不安定などのclassificationではpredictive entropyを使えます。

```math
H[y|x,D]=-\sum_c p_c\log p_c
```

class probabilityが拮抗するboundary付近を選びやすくなります。

## 4. BALD

BALDはpredictive uncertaintyからaleatoric componentを差し引き、model parameterに関するinformation gainを狙います。

```math
I(y,\theta|x,D)=H[y|x,D]-\mathbb E_{\theta}[H[y|x,\theta]]
```

measurement自体が曖昧な領域より、modelがまだ分かっていない領域を優先できます。

## 5. Boundary learning

材料設計ではoptimumより「成立領域の境界」が重要な場合があります。

例:

- phase boundary
- crack/no-crack
- specification pass/fail
- synthesis success/failure

Straddle、margin、boundary variance等を使いboundary近傍を重点的に取得できます。

## 6. Integrated uncertainty reduction

1点の局所uncertaintyではなく、候補空間全体のvariance reductionを見る方法があります。

```math
\alpha(x)=-\mathbb E[\text{integrated posterior variance after observing }x]
```

NIPV型criterionはこの考え方に近く、global surrogate qualityを改善したい場合に有効です。

## 7. Batch AL

DFTや実験を並列実行できる場合、単純なtop-q varianceでは似た候補が重複することがあります。

batch acquisition、fantasy update、diversity penalty等でinformation redundancyを抑えます。

## 8. Representation space coverage

composition/structure modelではraw input distanceよりlatent representation上のcoverageが意味を持つ場合があります。

```text
material -> encoder -> latent z -> uncertainty / diversity
```

ただしlatent distanceが物理的多様性を必ず表すとは限らないためvalidationが必要です。

## 9. MLIP Active Learning

MLIP training dataを増やす場合、energyだけでなくforce/stress uncertaintyやconfiguration diversityを考えます。

```text
MD / structure generation
 -> uncertainty detection
 -> selected configurations
 -> DFT labels
 -> MLIP retraining
```

これはproperty BOとは目的が異なり、potentialのvalid domainを広げるALです。

## 10. Residual-model AL

pretrained MLIP + residual GPでは、MLIPそのものではなくtarget-domain correction `delta(x)` のuncertaintyを使ってhigh-fidelity dataを選ぶこともできます。

```text
large residual uncertainty
 -> DFT / experiment
 -> update correction GP
```

## 11. Cost-aware AL

実験ごとにcostが違う場合、information gainだけでなく

```math
\frac{\text{information gain}}{\text{cost}}
```

を考えます。これはMulti-fidelity ALへ自然につながります。

## 12. BOとALのhybrid

材料開発では探索初期はmodel understandingを優先し、後半はoptimizationへ寄せる戦略が考えられます。

```text
early stage: AL / exploration
middle: mixed criterion
late stage: BO / exploitation
```

またbatch内で一部をBO、一部をALに割り当てる方法もあります。

## 13. Stopping rule

ALは「何点集めるか」も重要です。

- uncertaintyが閾値以下
- validation/calibrationが十分
- boundary uncertaintyが十分小さい
- marginal information gainがcostを下回る

などを停止条件にできます。

## 14. bochanでの位置付け

bochanのAL familyはregression / binary / multiclass / ordinal等のposteriorに対して、variance、entropy、BALD、NIPV、boundary-oriented criterion等を選択するdecision layerです。

材料workflowでは

```text
composition / structure / process
 -> surrogate posterior
 -> AL acquisition
 -> DFT / experiment
 -> retrain
```

として利用できます。

Materials Active Learningの目的は「良い候補だけを追う」ことではなく、**限られた評価budgetで次の意思決定に必要な材料空間の理解を最大化すること**です。
