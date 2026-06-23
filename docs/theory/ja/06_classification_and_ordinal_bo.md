# 06. 分類・順序出力を用いたベイズ最適化

二値、多クラス、順序モデルの観測は連続的な物理量ではありません。ベイズ最適化へ利用するには、class probabilityやlatent valueを意思決定上のutilityへ明示的に変換する必要があります。

model likelihoodとposterior contractは11・12章で扱い、本章ではdecision layerを扱います。

---

## 1. Labelを通常の回帰targetとして扱わない理由

```math
Y\in\{0,1,\ldots,K-1\}
```

をGaussian regression targetとすると、次を暗黙に仮定します。

- class間距離が数値として意味を持つ
- class `2`がclass `1`の2倍に相当する
- Gaussian residualが妥当
- valid range外のpredictionを許す
- unordered classに自然な順序がある

classification／ordinal likelihoodはclass probability

```math
p_k(x)=P(Y=k\mid x,\mathcal D)
```

を推定し、BOはそのprobability vectorのfunctionalをoptimizationします。

---

## 2. Decision-space transformation

```math
\mathbf p(x)=[p_0(x),\ldots,p_{K-1}(x)]
```

に対しscalar objectiveを

```math
u(x)=T(\mathbf p(x))
```

とします。multi-objectiveなら

```math
\mathbf u(x)=[T_1(\mathbf p(x)),\ldots,T_m(\mathbf p(x))].
```

classifierはprobabilityを提供し、`T`が何をoptimizationするかを定義します。

---

## 3. Binary probabilityとexpected utility

class `1`を望むなら

```math
u(x)=p(x)=P(Y=1\mid x,\mathcal D).
```

class `0`なら`u(x)=1-p(x)`です。

utility`u_0,u_1`を与えると

```math
U(x)=u_0[1-p(x)]+u_1p(x).
```

これは`p`のaffine functionですが、utility scaleはEIの改善量、`best_f`、reference point、経済的解釈に影響します。

input-dependent costを含むなら

```math
U(x)=u_0(x)[1-p(x)]+u_1(x)p(x)
```

となり、probabilityだけではrankingが決まりません。

---

## 4. Multiclass objective

target class`k^*`なら

```math
U_{k^*}(x)=p_{k^*}(x).
```

acceptable class set`A`なら

```math
U_A(x)=P(Y\in A\mid x)=\sum_{k\in A}p_k(x).
```

class utility`u_k`を与えると

```math
U(x)=\sum_{k=0}^{K-1}u_kp_k(x).
```

failure modeごとのcostやproduct categoryごとのrevenueを表せます。expected utilityを導入しても、unordered multiclass likelihoodがordinalになるわけではありません。

---

## 5. Ordinal expected utility

ordered class`0<1<\cdots<K-1`に対し、通常

```math
u_0\le u_1\le\cdots\le u_{K-1}
```

を設定します。

```math
U(x)=\sum_{k=0}^{K-1}u_kp_k(x).
```

`u_k=k`はadjacent grade間の価値差が等しいと仮定します。domain-specific utility

```math
\mathbf u=[0,0.1,0.7,1.0]
```

なら、grade 1から2への改善を大きく評価できます。

likelihoodは統計的な順序、utilityは運用上の価値を表します。

---

## 6. Minimum-grade probability

required grade`g`に対して

```math
P(Y\ge g\mid x)
=
\sum_{k=g}^{K-1}p_k(x).
```

をobjective、constraint、LSE target、reliability metricとして使えます。

engineering requirementがgrade基準なら、expected class indexより解釈しやすい場合があります。

---

## 7. Latent-score optimization

latent score`f(x)`を直接optimizationすることもできます。linkがmonotoneでrankingだけが重要な場合、boundary exploration、probability saturation回避には有用です。

ただしlatent scaleはmodel dependentであり、cutpointやlink calibrationに依存します。observed labelをそのままlatent `best_f`として使うことはできません。

user-facing BOではprobability／utility spaceが通常は適切です。

---

## 8. Probability／utility spaceでのEI

transformed objective`U(x)`に対し

```math
I(x)=\max[U(x)-U_{\mathrm{best}},0]
```

```math
\mathrm{EI}_U(x)=\mathbb E[I(x)\mid\mathcal D].
```

です。

posterior mean probabilityだけへdeterministic EI formulaを適用すると、probability function uncertaintyを無視します。

binary probabilityなら`U_{\mathrm{best}}\in[0,1]`、ordinal utilityなら`[\min u_k,\max u_k]`です。observed class integerと同一とは限りません。

---

## 9. UCBで使うvariance

```math
\mathrm{UCB}_U(x)=\mu_U(x)+\lambda\sigma_U(x).
```

`\sigma_U`の意味を区別します。

- posterior sampleから得るprobability function variance
- Bernoulli observation variance`p(1-p)`
- realized class utility variance
- expected-utility functionのepistemic variance

同じ「分類分散」と呼ぶと誤解が生じます。

---

## 10. Probability of Improvementの二重の意味

```math
\mathrm{PI}_U(x)=P(U(x)\ge\tau\mid\mathcal D)
```

は、uncertain utility functionがthresholdを超えるposterior probabilityです。

一方

```math
P(Y=1\mid x)=p(x)
```

はfuture label probabilityです。両者は異なります。

---

## 11. Classificationをconstraintとして使う

continuous objective`f(x)`とbinary feasibility modelがある場合、

```math
\alpha_c(x)=\alpha_f(x)P(\mathrm{feasible}\mid x)
```

というweightingを使えます。

複数constraintをindependentと仮定するならproductを使いますが、correlated constraintではjoint sampleまたはmultivariate probabilityが必要です。

chance constraint

```math
P(\mathrm{feasible}\mid x)\ge\gamma
```

は、probability weightingより強い運用要件です。

ordinal minimum-grade constraintは

```math
P(Y\ge g\mid x)\ge\gamma.
```

expected utility constraintとは異なります。

---

## 12. Calibration

optimizationはmodel errorを積極的に利用します。overconfident probability modelは、誤って高いprobabilityを予測する領域へBOを誘導します。

確認するmetricはreliability diagram、Brier score、log loss、ECE、class-specific calibration、ordinal cumulative calibrationです。

random validation data上のcalibrationがadaptive BO candidateにそのまま成立するとは限りません。

---

## 13. Risk-sensitive class utility

class utility distributionは

```math
P(U=u_k\mid x)=p_k(x).
```

です。meanだけでなく、lower quantile、CVaR、unacceptable utility probabilityを使えます。

```math
P(U<u_{\min}).
```

discrete distributionのVaRはprobabilityがclass boundaryを跨ぐと不連続に変化します。chance constraintやCVaRの方が安定する場合があります。

---

## 14. Input perturbation

execution`\tilde x=x+\delta`に対し

```math
\mathbb E_\delta[p(Y=1\mid x+\delta)],
```

```math
P_\delta(p(Y=1\mid x+\delta)\ge\gamma),
```

```math
\mathrm{CVaR}_\alpha[U(x+\delta)]
```

などを定義できます。

linear expected utilityではexpectationとutility sumが交換できますが、nonlinear riskやtarget-distance transformでは交換できません。

---

## 15. Multi-objective discrete output

```math
\mathbf U(x)=[U_1(x),\ldots,U_m(x)]
```

として、continuous property、success probability、ordinal utility、failure probabilityを同時にoptimizationできます。

EHVI／NEHVIの前に、maximization direction、scale、reference point、objective／constraint role、cross-output dependenceを確認します。

---

## 16. Hybrid model

別々のsubmodelを

```text
regression property model
binary feasibility model
ordinal quality model
multiclass failure-mode model
```

としてfitし、

```math
[t_{\mathrm{property}},
 p_{\mathrm{feasible}},
 E[u_{\mathrm{grade}}],
 p_{\mathrm{acceptable}}]
```

という共通decision vectorへ変換できます。

`HybridMultiOutputModel`は共通interfaceを提供しますが、自動的にcross-output covarianceを導入しません。詳細は15章です。

---

## 17. `bochan`実装との対応

### Binary BO

```text
src/bochan/acquisition/binary/bayesian_optimization/
```

代表class：`qBinaryProbabilityOfFeasibility`、`qBinaryExpectedImprovement`、`qBinaryProbabilityOfImprovement`、`qBinaryUpperConfidenceBound`。

### Multiclass BO

```text
src/bochan/acquisition/multiclass/bayesian_optimization/
```

target classまたはclass reductionをprobability posteriorへ適用します。

### Ordinal BO

```text
src/bochan/acquisition/ordinal/bayesian_optimization/
```

代表class：`qOrdinalExpectedImprovement`、`qOrdinalProbabilityOfImprovement`、`qOrdinalUpperConfidenceBound`、`qOrdinalProbabilityOfFeasibility`。

ordinal base `posterior()`はlatentであり、class probabilityは`class_probs()`から得ます。

### Objectiveとhybrid

```text
src/bochan/acquisition/objective/ordinal.py
src/bochan/acquisition/objective/hybrid.py
src/bochan/models/hybrid/
src/bochan/models/transforms/posterior/
```

---

## 18. 設定checklist

1. class semantics
2. target class／acceptable set
3. probability／latent／utility objective
4. utility valuesとunit
5. direction
6. `best_f` scale
7. variance definition
8. calibration
9. objective／constraint role
10. risk
11. perturbation aggregation
12. output dependence
13. acquisitionとposterior accessor
