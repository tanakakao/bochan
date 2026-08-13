# 05. レベル集合推定

レベル集合推定（Level-set Estimation; LSE）は、未知関数がthreshold以上／未満となる領域や、その境界を同定する問題です。global maximumではなく、集合または境界の推定精度を目的とします。

本章では問題、loss、confidence set、停止条件、評価を扱います。現行`bochan` classの具体式は16章を参照してください。

---

## 1. 基本定義

未知関数`f:\mathcal X\to\mathbb R`とthreshold`h`に対し、upper level setは

```math
L_h^+=\{x\in\mathcal X:f(x)\ge h\},
```

lower level setは

```math
L_h^-=\{x\in\mathcal X:f(x)<h\},
```

boundaryは

```math
B_h=\{x\in\mathcal X:f(x)=h\}
```

です。

membership indicatorを

```math
z_h(x)=\mathbf1[f(x)\ge h]
```

とし、`\widehat z_{h,t}`または推定集合`\widehat L_h^+`を少数観測から学習します。

---

## 2. BOとの違い

EIは高いobjectiveが期待できるpeak付近へ集中し得ます。LSEはthreshold contour全体の分類やboundary positionを学ぶ必要があります。

用途は次です。

- safe operating region
- material phase boundary
- defect transition
- process window
- reliability region
- specification exceedance map

評価もbest responseではなくset／boundary errorで行います。

---

## 3. Posterior membership probability

Gaussian latent posterior

```math
f(x)\mid\mathcal D_t
\sim
\mathcal N(\mu_t(x),\sigma_t^2(x))
```

なら

```math
\pi_t(x)
=
P(f(x)\ge h\mid\mathcal D_t)
=
\Phi\left(
\frac{\mu_t(x)-h}{\sigma_t(x)}
\right).
```

symmetric 0-1 lossでは

```math
\widehat z_t(x)=\mathbf1[\pi_t(x)\ge1/2]
```

がBayes classifierです。false-safeとfalse-unsafeのcostが異なる場合はprobability thresholdを変更します。

---

## 4. Confidence-bound classification

```math
l_t(x)=\mu_t(x)-\sqrt{\beta_t}\sigma_t(x),
```

```math
u_t(x)=\mu_t(x)+\sqrt{\beta_t}\sigma_t(x)
```

として

```math
H_t=\{x:l_t(x)\ge h\},
```

```math
L_t=\{x:u_t(x)<h\},
```

```math
U_t=\mathcal X\setminus(H_t\cup L_t)
```

と分類します。

`H_t`はconfidently above、`L_t`はconfidently below、`U_t`はunresolvedです。large betaはpremature classificationを防ぎますが、unresolved regionを広くします。

---

## 5. Loss

### Pointwise misclassification

有限grid`\mathcal G`上で

```math
\mathcal L_{\mathrm{mis}}
=
\frac1{|\mathcal G|}
\sum_{x\in\mathcal G}
\mathbf1[\widehat z_t(x)\ne z_h(x)].
```

### Weighted classification

```math
\mathcal L_{\mathrm{weighted}}
=
\frac1{|\mathcal G|}
\sum_x
\left[
 c_{\mathrm{FS}}\mathbf1(\widehat z=1,z=0)
+
 c_{\mathrm{FU}}\mathbf1(\widehat z=0,z=1)
\right].
```

safetyではfalse-safe costを大きくします。

### Symmetric difference

```math
\mathcal L_\Delta
=
\nu(\widehat L_h^+\triangle L_h^+).
```

### Jaccard

```math
J
=
\frac{\nu(\widehat L_h^+\cap L_h^+)}
{\nu(\widehat L_h^+\cup L_h^+)},
\qquad
\mathcal L_J=1-J.
```

### Hausdorff distance

```math
d_H(\widehat B_h,B_h)
=
\max\left\{
\sup_{x\in\widehat B_h}\inf_{y\in B_h}\|x-y\|,
\sup_{y\in B_h}\inf_{x\in\widehat B_h}\|x-y\|
\right\}.
```

### Integrated Bayes risk

posterior pointwise risk

```math
r_t(x)=\min[\pi_t(x),1-\pi_t(x)]
```

を積分して

```math
R_t=\int_{\mathcal X}r_t(x)\,d\nu(x)
```

とします。

---

## 6. Region-of-interest weighting

scientific weight`w(x)`を用いて

```math
\mathcal L_w
=
\int
w(x)
\mathbf1[\widehat z_t(x)\ne z_h(x)]
\,d\nu(x)
```

とします。

production頻度、nominal condition近傍、false-safe cost、physically valid domainなどを反映できます。

---

## 7. Multiple threshold

`h_1<\cdots<h_R`はresponse spaceを複数bandへ分割します。

- 全thresholdを均等に学ぶ
- 1つのthresholdをtargetにする
- importance weightを付ける
- least-resolved thresholdを優先する
- multiclass region labelingとして扱う

mean、sum、max、min reductionは異なるpolicyです。

---

## 8. Excursionとreliability set

posterior excursion setは

```math
E_{h,\gamma}
=
\{x:P(f(x)\ge h\mid\mathcal D)\ge\gamma\}
```

です。

future noisy observationに対するreliability setは

```math
R_{h,\gamma}
=
\{x:P(Y(x)\ge h\mid\mathcal D)\ge\gamma\}.
```

前者はlatent function uncertainty、後者はobservation noiseを含み得ます。

---

## 9. Classification／ordinal boundary

binary probability level setは

```math
L_\tau
=
\{x:P(Y=1\mid x,\mathcal D)\ge\tau\}.
```

`\tau=0.5`がordinary decision boundaryですが、reliability requirementでは0.9などを使います。

multiclassではtarget-class probability、class-set probability、top-class transitionなど、boundary definitionを先に選ぶ必要があります。

ordinalでは次を区別します。

```math
\{x:f(x)\ge c_j\},
```

```math
\{x:P(Y\ge g\mid x)\ge\gamma\},
```

```math
\{x:\mathbb E[U(Y)\mid x]\ge u_0\}.
```

---

## 10. Multi-output level set

`\mathbf f(x)=[f_1(x),\ldots,f_m(x)]`に対し

### Intersection

```math
L_\cap
=
\bigcap_{j=1}^{m}
\{x:f_j(x)\ge h_j\}.
```

### Union

```math
L_\cup
=
\bigcup_{j=1}^{m}
\{x:f_j(x)\ge h_j\}.
```

### At-least-r-of-m

```math
L_r
=
\left\{
 x:
\sum_j\mathbf1[f_j(x)\ge h_j]\ge r
\right\}.
```

### Scalarized set

```math
L_s=\{x:s(\mathbf f(x))\ge h\}.
```

output-wise score平均は、これらのjoint set probabilityを自動的に表しません。

correlated outputのintersection probabilityはmultivariate probabilityです。independenceを仮定したproduct approximationと区別します。

---

## 11. Sequential LSE policy

理想的なone-step policyは

```math
x_{t+1}
\in
\arg\max_x
\mathbb E[
\mathcal L_t-\mathcal L_{t+1}
\mid x,\mathcal D_t].
```

直接計算にはfuture outcome、posterior update、domain全体のloss再計算が必要です。そのため実装では次のproxyを使います。

- confidence-bound ambiguity
- Straddle
- boundary-weighted variance
- class entropy
- probability-of-exceedance ambiguity
- integrated contour uncertainty
- joint covariance volume

具体式は16章です。

---

## 12. q-batch、replicate、noise

q-batchの理想値はjoint expected loss reductionです。実用上はlogdet、sequential greedy、same-batch repulsion、pending／observed penaltyを使います。

noisy measurementでは同一点replicateがlatent mean uncertaintyを減らす場合があります。hard duplicate rejectionは常に正しいとは限りません。

replicateが有効なのは、noiseが大きい、noise variance自体を学びたい、1点のsafety confidenceが必要、condition変更よりreplicateが安価な場合です。

---

## 13. Input uncertainty

executionが`\tilde x=x+\delta`なら、robust setには次があります。

### Mean set

```math
\{x:\mathbb E_\delta[f(x+\delta)]\ge h\}.
```

### Chance set

```math
\{x:P_\delta(f(x+\delta)\ge h)\ge\gamma\}.
```

### CVaR set

```math
\{x:\mathrm{CVaR}_\alpha[f(x+\delta)]\ge h\}.
```

perturbationごとのLSE score平均はheuristicであり、必ずしも上記robust setのexpected loss reductionではありません。

---

## 14. Stopping criterion

- unresolved measure：`\nu(U_t)\le\epsilon`
- maximum ambiguity：`\sup_x r_t(x)\le\epsilon`
- integrated risk：`R_t\le\epsilon`
- credible boundary bandが十分狭い
- 必要なsizeのconnected safe regionをcertify

acquisition value thresholdより、set lossやunresolved measureの方が解釈しやすい場合が多いです。

---

## 15. Evaluation protocol

1. ground truthまたはdense reference
2. threshold
3. domain measure／grid
4. observation noise
5. input perturbation
6. initial design
7. budget
8. q
9. set estimator
10. external loss
11. random seed

plotすべきmetricはset error、false-safe／false-unsafe、unresolved measure、Jaccard、Hausdorff、membership calibration、candidate location、duplicate／optimizer failureです。

---

## 16. Constrained BOとの違い

| task | 主目的 |
|---|---|
| constrained BO | feasible region内で高いobjectiveを見つける |
| feasibility search | feasible pointを1つ見つける |
| constraint LSE | feasible／infeasible領域全体をmapする |
| safe BO | unsafe evaluationを抑えながらobjectiveを改善する |

同じconstraint modelを使えても、acquisitionとexternal lossは異なります。

---

## 17. `bochan`実装との対応

```text
src/bochan/acquisition/regression/levelset_estimation/
src/bochan/acquisition/binary/levelset_estimation/
src/bochan/acquisition/multiclass/levelset_estimation/
src/bochan/acquisition/ordinal/levelset_estimation/
src/bochan/acquisition/non_gaussian/levelset_estimation/
```

public nameは`src/bochan/api/registry/acquisition.py`で登録されます。

acquisitionによって消費する空間は異なります。

- regression latent mean／variance
- binary latent GP
- binary probability
- multiclass target probability
- ordinal latent posteriorとcutpoint
- ordinal class probability
- heteroscedastic noise model
- multi-output reduction

現行classごとの式は16章で確認してください。

---

## 18. 新規LSE componentの確認項目

1. target setとthreshold scale
2. latent／predictive／probability／utility space
3. external loss
4. membership estimator
5. local／integrated criterion
6. pointwise／joint q-batch
7. output／boundary reduction
8. noise interpretation
9. replicate policy
10. perturbation definition
11. pending／observed handling
12. stopping criterion
13. Tensor shape
14. exact expected loss reductionかproxyか
