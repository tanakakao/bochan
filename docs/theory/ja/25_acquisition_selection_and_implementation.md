# 25. 獲得関数の選択と実装対応

この章では、獲得関数を「数式を知る」段階から「どれを使うか判断する」段階へ進めます。既存の03章・04章・07章で扱った理論を、BoTorchとbochanの実装へ対応付けます。

## 1. 獲得関数選択の基本軸

最初に次の5点を確認します。

1. 目的は最適化か、学習か、境界探索か
2. observation noiseは無視できるか
3. `q=1`かbatch selectionか
4. single objectiveかmulti-objectiveか
5. posterior outputはcontinuous regressionかclassification/ordinalか

この順序で考えると、獲得関数名から逆算するより誤用が減ります。

## 2. 単目的continuous regression

### Probability of Improvement

```math
\mathrm{PI}(x)=P(f(x)>f_{\mathrm{best}}+\xi\mid\mathcal D)
```

改善の「大きさ」ではなく改善確率を見るため、改善幅が小さくても確実性の高い候補を好みます。

**使う場面**

- thresholdを超えること自体が重要
- 小さい改善でも十分
- exploitationを強めたい

**BoTorch**

- `ProbabilityOfImprovement`
- `qProbabilityOfImprovement`

**bochan**

standard Gaussian regressionではBoTorch実装を優先します。binary / multiclass / ordinalでは各family専用のPI classを利用します。

---

### Expected Improvement

```math
\mathrm{EI}(x)
=\mathbb E[(f(x)-f_{\mathrm{best}})_+]
```

改善確率と改善量の両方を考慮します。

**使う場面**

- noiseが小さい単目的BO
- exploration/exploitationを自動的に両立したい
- standardなbaselineが欲しい

**BoTorch**

- `ExpectedImprovement`
- `qExpectedImprovement`
- `LogExpectedImprovement`
- `qLogExpectedImprovement`

**推奨**

数値安定性が必要な場合はLogEI系を優先します。EIとLogEIは別の探索戦略ではなく、同じimprovement criterionを安定に最適化するための実装差です。

---

### Noisy Expected Improvement

```math
\mathrm{NEI}(X)
=\mathbb E\left[\left(\max f(X)-\max f(X_{\mathrm{baseline}})\right)_+\right]
```

観測noiseによりbest baseline自体が不確かな場合に使います。

**使う場面**

- 実験値に測定noiseがある
- simulatorがstochastic
- replicate間ばらつきが大きい

**BoTorch**

- `qNoisyExpectedImprovement`
- `qLogNoisyExpectedImprovement`

**重要な設定**

- `X_baseline`
- `X_pending`
- sampler
- constraints
- baseline pruning

`best_f`だけを固定して扱うEIより、noisy experimentではNEIの方が理論的に自然です。

---

### Upper Confidence Bound

```math
\mathrm{UCB}(x)=\mu(x)+\sqrt\beta\,\sigma(x)
```

**使う場面**

- exploration量を明示的に調整したい
- improvement thresholdへ依存したくない
- uncertaintyの大きい領域を積極的に調べたい

**BoTorch**

- `UpperConfidenceBound`
- `qUpperConfidenceBound`

`beta`の意味は実装上のparameterizationに依存するので、standard deviationに対する係数か、その平方かを確認します。

---

### Thompson Sampling

```math
\tilde f\sim p(f\mid\mathcal D),\qquad
x^*=\arg\max_x\tilde f(x)
```

**使う場面**

- large discrete candidate set
- batch diversityを自然に出したい
- acquisition optimizationを単純化したい

posterior marginalを各点独立にsampleするとfunction correlationを失うため、pathwise sampleやjoint posterior sampleが必要です。

## 3. Knowledge Gradient

KGは「この点を観測すると、将来のbest decisionがどれだけ改善するか」を評価します。

```math
\mathrm{KG}(x)
=\mathbb E_{y_x}\left[\max_{x'}\mu_{t+1}(x';y_x)\right]
-\max_{x'}\mu_t(x')
```

**EIとの違い**

EIは候補点そのもののimprovementを評価します。KGは観測後に別の点を選び直せる価値まで含みます。

**BoTorch**

- `qKnowledgeGradient`

**使う場面**

- 1回の評価コストが高い
- myopic improvementよりvalue of informationを重視する
- 最終decision qualityが重要

nested optimizationが必要なのでEI/UCBより計算コストは高くなります。

## 4. Multi-step lookahead

multi-step lookaheadは、複数step先のfuture observationとfuture decisionを明示的に扱います。

```math
\alpha_t(x_t)
=\mathbb E\left[V(\mathcal D_{t+H})\mid x_t,\mathcal D_t\right]
```

**BoTorch**

- `qMultiStepLookahead`

**使う場面**

- 残り評価回数が少ない
- horizonが明確
- 最初の一手が後続探索へ強く影響する

計算量が急増するため、通常のBOではEI/NEI/UCBをbaselineとして比較します。

## 5. Multi-objective

### qEHVI

```math
\mathrm{EHVI}(X)
=\mathbb E[\Delta\mathrm{HV}(X)]
```

**使う場面**

- 複数objective
- observation noiseが比較的小さい
- Pareto frontierを直接改善したい

**BoTorch**

- `qExpectedHypervolumeImprovement`

### qNEHVI

noisy baseline frontierのuncertaintyを積分します。

**使う場面**

- noisy multi-objective experiment
- Pareto frontier自体が不確か

**BoTorch**

- `qNoisyExpectedHypervolumeImprovement`

### qNParEGO

random scalarizationを使ってPareto frontierを探索します。

**使う場面**

- objective数が多い
- hypervolume計算が重い
- scalarized BOとして扱いたい

**bochan**

standard regressionだけでなくbinary / multiclass / ordinalのmulti-output familyにもNParEGO系実装があります。

## 6. 制約付きBO

unknown constraintを

```math
c_j(x)\le0
```

とします。feasibility probabilityは

```math
P(\text{feasible}\mid x)
=P(c_1(x)\le0,\ldots,c_J(x)\le0)
```

です。

一般には

```math
\alpha_{\mathrm{constrained}}(x)
=\alpha_{\mathrm{base}}(x)\times P(\text{feasible}\mid x)
```

のように組み合わせられます。

bochanでは`acquisition/feasible/`にconstraint helperとwrapper logicを分離しており、獲得関数ごとにfeasibility処理を重複実装しない設計です。

## 7. Classification / ordinalへの拡張

classification modelのlatent GP outputをそのままEIへ入れると、利用者が欲しい確率やutilityと意味が一致しない場合があります。

したがって

```text
posterior latent
  -> probability / expected utility
  -> acquisition score
```

という変換が必要です。

### Binary

代表的なbochan class:

- `qBinaryProbabilityOfFeasibility`
- `qBinaryExpectedImprovement`
- `qBinaryProbabilityOfImprovement`
- `qBinaryUpperConfidenceBound`

### Multiclass

代表的なbochan class:

- `qMulticlassProbabilityOfFeasibility`
- `qMulticlassExpectedImprovement`
- `qMulticlassProbabilityOfImprovement`
- `qMulticlassUpperConfidenceBound`

### Ordinal

代表的なbochan class:

- `qOrdinalExpectedImprovement`
- `qOrdinalProbabilityOfImprovement`
- `qOrdinalUpperConfidenceBound`
- `qOrdinalProbabilityOfFeasibility`

ordinalではclass番号そのものではなく、ordered categoryへ割り当てたutilityの意味を明示する必要があります。

## 8. Multi-output classification / ordinal

bochanではfamilyごとにmulti-objective acquisitionを分けています。

| Family | EHVI | NEHVI | NParEGO |
|---|---|---|---|
| Binary | `qMultiOutputBinaryExpectedHypervolumeImprovement` | `qMultiOutputBinaryNoisyExpectedHypervolumeImprovement` | `qMultiOutputBinaryNParEGO` |
| Multiclass | `qMultiOutputMulticlassExpectedHypervolumeImprovement` | `qMultiOutputMulticlassNoisyExpectedHypervolumeImprovement` | `qMultiOutputMulticlassNParEGO` |
| Ordinal | `qMultiOutputOrdinalExpectedHypervolumeImprovement` | `qMultiOutputOrdinalNoisyExpectedHypervolumeImprovement` | `qMultiOutputOrdinalNParEGO` |

モデルがmulti-outputであることと、optimizationがmulti-objectiveであることは同義ではありません。複数outputのうち一つだけをobjectiveに使う場合もあります。

## 9. Active Learningとの違い

BO acquisitionは高いutilityを持つ入力を探します。一方Active Learningでは、parameter uncertainty、predictive uncertainty、decision boundaryなどを改善することが目的です。

```text
BO:
posterior -> utility improvement -> next candidate

AL:
posterior -> information / uncertainty reduction -> next observation
```

variance、predictive entropy、BALD、NIPVは、単に「探索的なBO」と見るよりAL criterionとして理解した方が明確です。

## 10. 実務向け選択表

| 状況 | 第一候補 |
|---|---|
| noise小・単目的 | LogEI / qLogEI |
| noisy・単目的 | qLogNEI |
| explorationを調整 | UCB |
| value of information重視 | KG |
| 複数stepを明示 | Multi-step lookahead |
| noise小・多目的 | qEHVI |
| noisy・多目的 | qNEHVI |
| objective数が多い | qNParEGO |
| feasibilityが重要 | base acquisition + feasibility |
| uncertainty reduction | variance / NIPV |
| classification information | entropy / BALD |

## 11. bochan実装の原則

bochanのacquisition packageは、taskとmodel familyを分離しています。

```text
regression / binary / multiclass / ordinal / non_gaussian
    ×
bayesian_optimization / active_learning / levelset_estimation
```

standard Gaussian regressionではBoTorch標準実装を優先し、classification・ordinal・heteroscedastic・wrapper固有処理など、意味変換が必要な部分をbochan側で補います。

この設計により、「新しい獲得関数を独自実装すること」自体を目的にせず、BoTorchの標準実装とbochan固有のmodel semanticsを適切に接続できます。
