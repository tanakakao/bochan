# 04. 能動学習

能動学習は、予測モデルまたは科学的対象に関する不確かさを効率よく減らすために観測点を選びます。目的関数の最大化ではなく、「何を学びたいか」と「学習品質をどう評価するか」によって定義されます。

境界・集合を学ぶレベル集合推定は05章、具体的LSE実装式は16章で扱います。

---

## 1. 問題設定

```math
\mathcal D_t=\{(x_i,y_i)\}_{i=1}^{n_t}
```

に対し

```math
x_{t+1}
\in
\arg\max_{x\in\mathcal X}
\alpha_{\mathrm{AL}}(x;\mathcal D_t)
```

を選びます。

learning targetの例は次です。

- domain全体のlatent function
- predictive response distribution
- class probability
- model parameter
- selected output
- region of interest
- expected utility
- task covariance
- noise function

高いobjective valueと高いlearning valueは一致しません。

---

## 2. Pool-basedとcontinuous setting

pool-basedでは有限集合`\mathcal P`から

```math
x_{t+1}
\in
\arg\max_{x\in\mathcal P\setminus X_{\mathrm{observed}}}
\alpha(x)
```

を選びます。

continuous settingでは

```math
x_{t+1}\in\arg\max_{x\in\mathcal X}\alpha(x)
```

をacquisition optimizerで解きます。

実験designでは、入力だけでなくsensor、replicate、測定output subset、measurement costもdecisionになる場合があります。

---

## 3. Local uncertainty sampling

Gaussian latent posterior

```math
f(x)\mid\mathcal D
\sim
\mathcal N(\mu(x),\sigma_f^2(x))
```

に対し

```math
\alpha_{\mathrm{var}}(x)=\sigma_f^2(x)
```

とする最も単純なcriterionです。

安価なbaselineですが、domain boundary、irrelevant region、heteroscedastic noise、clustered q-batchに弱い場合があります。

---

## 4. Predictive entropy

```math
H(Y\mid x,\mathcal D)
=-\mathbb E[\log p(Y\mid x,\mathcal D)].
```

### Gaussian response

```math
H(Y\mid x,\mathcal D)
=
\frac12\log(2\pi e\sigma_Y^2(x)).
```

Gaussian entropy最大化はpredictive variance最大化と同じです。`\sigma_Y^2`がobservation noiseを含むと、irreducibly noisy regionを選ぶ可能性があります。

### Binary

```math
H(Y)
=-p\log p-(1-p)\log(1-p).
```

最大は`p=0.5`です。

### Multiclass／ordinal

```math
H(Y)=-\sum_{k=0}^{K-1}p_k\log p_k.
```

ordinalでcategorical entropyを使うと、隣接gradeと遠いgradeの誤りを同じように扱います。

---

## 5. Aleatoricとepistemic

Gaussian regressionでは

```math
\mathrm{Var}(Y\mid x,\mathcal D)
=
\underbrace{\mathrm{Var}(f(x)\mid\mathcal D)}_{\text{epistemic}}
+
\underbrace{\sigma_n^2(x)}_{\text{aleatoric}}.
```

classificationの`p(1-p)`はfuture labelの観測分散であり、probability functionのepistemic uncertaintyそのものではありません。

criterionがtotal predictive uncertainty、latent uncertainty、probability uncertainty、noise parameter uncertaintyのどれを学ぶかを明示します。

---

## 6. BALD

Bayesian Active Learning by Disagreementは

```math
\mathrm{BALD}(x)
=
I(Y;\Theta\mid x,\mathcal D)
```

です。entropy identityから

```math
I(Y;\Theta\mid x,\mathcal D)
=
H(Y\mid x,\mathcal D)
-
\mathbb E_{\Theta\mid\mathcal D}
[H(Y\mid x,\Theta)].
```

第1項はtotal predictive uncertainty、第2項はlatent stateが既知でも残るaleatoric uncertaintyです。

binary MC estimatorはlatent sample`f^{(s)}`をprobability`p_s`へ変換し、

```math
\widehat{\mathrm{BALD}}
=
H\left(\frac1S\sum_s p_s\right)
-
\frac1S\sum_s H(p_s).
```

multiclassでは`p_s`をclass-probability vectorへ置き換えます。

---

## 7. Marginとprobability variance

binary marginは

```math
\alpha(x)=1-|2p(x)-1|.
```

multiclassではtop two probability`p_{(1)},p_{(2)}`に対し

```math
\alpha(x)=1-[p_{(1)}-p_{(2)}].
```

probability posterior sample`p^{(s)}(x)`がある場合、

```math
\mathrm{Var}_s[p^{(s)}(x)]
```

はprobability function uncertaintyです。Bernoulli observation variance`\bar p(1-\bar p)`と区別します。

---

## 8. Integrated Posterior Variance

reference measure`\nu`上のintegrated varianceは

```math
\mathrm{IPV}_t
=
\int_{\mathcal X}
\sigma_t^2(z)\,d\nu(z).
```

観測`x`による理想的なreductionは

```math
\Delta\mathrm{IPV}(x)
=
\mathrm{IPV}_t
-
\mathbb E_{y_x}
[\mathrm{IPV}_{t+1}\mid x,y_x].
```

exact Gaussian GPではcovariance updateから

```math
\sigma_{t+1}^2(z)
=
\sigma_t^2(z)
-
\frac{k_t(z,x)^2}
{k_t(x,x)+\sigma_n^2}
```

となります。

`qNegIntegratedPosteriorVariance`のように、future IPVのnegativeをmaximizeするinterfaceもあります。

---

## 9. 回帰能動学習

代表的criterionは次です。

- latent posterior variance
- predictive entropy
- Gaussian mutual information
- integrated variance reduction
- region-weighted IPV

Gaussian noise下で同一点のlatent valueとobservationのmutual informationは

```math
I(Y;f(x)\mid\mathcal D)
=
\frac12\log\left(
1+
\frac{\sigma_f^2(x)}{\sigma_n^2(x)}
\right).
```

これはepistemic varianceをnoiseで割り引きます。

---

## 10. 分類・順序能動学習

### Binary

learning targetはlatent boundary、class probability、future label、feasibility probabilityなどです。variance、entropy、margin、probability variance、BALDを使います。

### Multiclass

class axisを保持し、sum、mean、max、target class selectionなどの意味あるreductionを選びます。critical classのuncertaintyをclass averageで隠さないようにします。

### Ordinal

対象はfull class probability、latent quality、cutpoint、expected utility、minimum-grade probability、boundaryです。

class utility`u_k`に対するconditional utility varianceは

```math
\mathrm{Var}(U\mid x)
=
\sum_kp_k(x)[u_k-\bar u(x)]^2.
```

ただしこれはrealized class utilityの分散であり、expected-utility functionのepistemic varianceとは限りません。

---

## 11. Multi-output

output-wise score`a_j(x)`を

```math
\alpha(x)=\sum_jw_ja_j(x),
```

```math
\alpha(x)=\max_j a_j(x),
```

```math
\alpha(x)=\min_j a_j(x)
```

などでreduceします。

correlated outputではjoint informationはmarginal informationの単純和ではありません。

```math
I(Y_1,\ldots,Y_m;\Theta)
\ne
\sum_j I(Y_j;\Theta).
```

independent ModelListや`HybridPosterior`はfull cross-output informationを持ちません。

---

## 12. q-batch能動学習

joint informationは

```math
I(\mathbf Y_X;\Theta\mid\mathcal D).
```

Gaussian observationでは

```math
I(\mathbf Y_X;\mathbf f_X)
=
\frac12\log\det\left(
I+\sigma_n^{-2}\Sigma_X
\right).
```

logdetはuncertainかつnonredundantなbatchを評価します。

classificationのjoint BALDは`K^q`のlabel combinationを持つため、greedy／MC approximationが必要です。

---

## 13. 異分散能動学習

```math
Y=f(x)+\varepsilon(x),
\qquad
\varepsilon(x)\sim\mathcal N(0,\sigma_n^2(x))
```

では、total predictive variance最大化がhigh-noise regionへ偏る場合があります。

- mean functionを学ぶ：`\sigma_f^2/(\sigma_n^2+\epsilon)`など
- noise functionを学ぶ：noise model uncertainty
- 両方学ぶ：`\alpha_f+\lambda\alpha_g`
- future observationを予測する：total predictive uncertainty

目的を定めてからnoise weightingを選びます。

---

## 14. Input perturbationとcost

nominal point`x`のexecutionが`x+\delta`なら、pointwise informationの平均

```math
\mathbb E_\delta[\alpha_0(x+\delta)]
```

と、robust functional`\rho[f(x+\delta)]`自体に関するinformationは同じではありません。

measurement cost`c(x,S)`がある場合は

```math
\alpha_{\mathrm{cost}}(x,S)
=
\frac{I(Y_S;\Theta\mid x,\mathcal D)}{c(x,S)}.
```

を考えます。

---

## 15. 停止と評価

停止基準は、integrated variance、maximum local uncertainty、validation loss、BALD、calibration、region-of-interest uncertainty、budgetなどです。

評価metricは次です。

- regression：RMSE、NLPD、coverage、IPV
- classification：log loss、Brier、calibration、macro-F1
- ordinal：ranked probability score、class MAE、boundary calibration
- sequential：metric versus experiment count／cost／wall-clock

---

## 16. `bochan`実装との対応

### Regression

```text
src/bochan/acquisition/regression/active_learning/
```

代表class：`qRegressionPredictiveEntropy`、`qRegressionBALD`、`qRegressionPosteriorVariance`、`qRegressionNegIntegratedPosteriorVariance`。

### Binary

```text
src/bochan/acquisition/binary/active_learning/
```

代表class：`qBinaryPredictiveEntropy`、`qBinaryBALD`、`qBinaryJointBALD`、`qBinaryProbabilityVariance`、`qBinaryMarginUncertainty`。

### Multiclass

```text
src/bochan/acquisition/multiclass/active_learning/
```

代表class：`qMulticlassPredictiveEntropy`、`qMulticlassBALD`、`qMulticlassJointBALD`、probability variance、margin、IPV proxy。

### Ordinal

```text
src/bochan/acquisition/ordinal/active_learning/
```

代表class：`qOrdinalPredictiveEntropy`、`qOrdinalBALD`、`qOrdinalUtilityVariance`、`qOrdinalMarginUncertainty`。

public nameの解決は`src/bochan/api/acquisition_registry.py`で行います。class実装がexact、fantasy-based、MC、proxyのどれかを決めます。

---

## 17. 新規componentの確認項目

1. learning target
2. uncertainty decomposition
3. local／global criterion
4. pointwise／joint q-batch
5. posterior space
6. class／output reduction
7. heteroscedastic interpretation
8. perturbation reduction order
9. pending／observed handling
10. cost
11. stopping metric
12. external evaluation loss
13. approximation
