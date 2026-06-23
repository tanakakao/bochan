# 10. 回帰モデルと尤度

本章では、応答変数のsupportと観測分布に基づいて回帰モデルを整理します。GP prior、kernel、Gaussian conditioning、ELBOなどの共通事項は01章で定義済みです。

中心となる問いは次です。

> 観測された応答に対して、どの確率分布を尤度として用いるべきか。また、modelの`posterior()`は何を表しているか。

---

## 1. 回帰問題の定義

入力を

```math
x_i\in\mathcal X
```

応答を

```math
y_i\in\mathcal Y
```

とします。応答のsupportは尤度選択の重要な手掛かりです。

| 応答のsupport | 代表的なmodel |
|---|---|
| $\mathbb R$ | Gaussian回帰 |
| $(0,1)$ | Beta回帰 |
| $(0,\infty)$の連続量 | Gamma回帰 |
| $\{0,1,2,\ldots\}$ | PoissonまたはNegative Binomial回帰 |
| 有界区間$(a,b)$ | 変換後Betaまたは有界尤度 |
| heavy-tailedな実数応答 | Student-tまたはrobust likelihood |

適切な変換後にGaussian likelihoodを使う方法もありますが、変換と逆変換の意味を明示する必要があります。

---

## 2. 一般化された潜在関数回帰

潜在関数を

```math
f(x)
```

とします。尤度parameterをlink functionを通して

```math
\eta(x)=f(x),
\qquad
\vartheta(x)=g^{-1}(\eta(x))
```

と定義します。観測modelは

```math
y\mid x
\sim
p(y\mid\vartheta(x))
```

です。

次の3つの空間を区別してください。

1. latent space：$f(x)$
2. likelihood-parameter space：$\vartheta(x)$
3. response space：$Y$

inverse linkが非線形なら

```math
g^{-1}(\mathbb E[f])
\ne
\mathbb E[g^{-1}(f)]
```

です。posterior meanだけをlinkへ代入したplug-in predictionは、正しくmarginalizeしたpredictive meanと一致しない場合があります。

---

## 3. Gaussian回帰

### 3.1 Homoscedastic model

標準modelは

```math
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma_n^2)
```

です。等価に

```math
y_i\mid f_i
\sim
\mathcal N(f_i,\sigma_n^2)
```

と書けます。

観測共分散は

```math
K_y=K_f+\sigma_n^2I.
```

通常の`SingleTaskGP`では1つのglobal noise varianceを学習します。fixed noiseや異分散modelを使わない限り、入力ごとにnoiseは変化しません。

### 3.2 既知の観測分散

測定分散$s_i^2$が既知なら

```math
y_i\mid f_i
\sim
\mathcal N(f_i,s_i^2)
```

とし、

```math
K_y
=
K_f+
\mathrm{diag}(s_1^2,\ldots,s_n^2)
```

を使います。

実装上の`train_Yvar`にはstandard deviationではなくvarianceを渡します。outcome transformを使う場合、`train_Yvar`も変換後のscaleに整合させる必要があります。

既知のper-observation varianceと、noise functionを別modelで学習することは別問題です。後者は13章で扱います。

### 3.3 Latent predictionとnoisy prediction

latent posteriorはunderlying process $f(x)$を予測します。future observationのpredictive varianceは

```math
\mathrm{Var}(Y_*\mid\mathcal D)
=
\mathrm{Var}(f_*\mid\mathcal D)
+\sigma_n^2
```

です。

用途は次のように分かれます。

- 基礎processの最適化：latent posterior
- future measurementのprediction interval：noise込みposterior
- specificationを満たすfuture observationの確率：noise込みposterior
- model uncertaintyを減らす能動学習：通常はlatent variance

`observation_noise`引数が利用可能なmodelでは、この区別を明示します。

---

## 4. Gaussian posteriorから得られる意思決定量

scalar posteriorを

```math
f(x)\mid\mathcal D
\sim
\mathcal N(\mu(x),\sigma^2(x))
```

とします。

### Mean

```math
\mathbb E[f(x)\mid\mathcal D]=\mu(x).
```

### Quantile

```math
q_\alpha(x)
=
\mu(x)+\Phi^{-1}(\alpha)\sigma(x).
```

### Exceedance probability

```math
P(f(x)\ge h\mid\mathcal D)
=
\Phi\left(
\frac{\mu(x)-h}{\sigma(x)}
\right).
```

### Joint posterior sample

```math
f^{(s)}(X)
\sim
p(f(X)\mid\mathcal D).
```

これらを計算する前に、latent posteriorかfuture observation posteriorかを決めます。

---

## 5. 独立多出力回帰

複数出力を

```math
\mathbf y(x)
=
[y_1(x),\ldots,y_m(x)]
```

とします。独立modelは

```math
p(f_1,\ldots,f_m\mid\mathcal D)
=
\prod_{j=1}^{m}
p(f_j\mid\mathcal D_j)
```

を仮定します。

### 長所

- outputごとに異なるkernelとnoiseを設定できる
- outputごとに異なるtraining inputを持てる
- missing outputを自然に扱える
- output間関係が弱いときにnegative transferを避けられる

### 短所

- output間で情報共有しない
- cross-output posterior covarianceがない
- joint constraintやscalarization varianceで独立近似になる

BoTorchのModelList-style構成はこのpatternです。

---

## 6. 相関多出力回帰

separable multitask covarianceは

```math
\mathrm{Cov}[f_a(x),f_b(x')]
=
B_{ab}k_X(x,x')
```

です。$B$はtask covariance matrixです。

全taskが同じinput gridで観測される場合、

```math
K_{\mathrm{full}}
=
B\otimes K_X
```

というKronecker構造を利用できます。

### 解釈

学習された$B$は、input kernel、transform、noiseを考慮したlatent associationです。raw target列のPearson correlationそのものではありません。

### 利点

- data-poor taskへのinformation transfer
- coherentなjoint posterior sample
- joint constraintやmulti-objective uncertaintyの表現

### 注意

- task relationが誤っているとnegative transfer
- task scaleとinput scaleのidentifiability
- aligned dataへの強い仮定
- covariance computationの複雑化

`KroneckerMultiTaskGP`系は、共通inputで複数出力を観測する場合に自然です。

---

## 7. 混合連続・カテゴリ入力

入力を

```math
x=(x_c,x_g)
```

とします。$x_c$はcontinuous、$x_g$はcategoricalです。

mixed kernelの概念的な形は

```math
k(x,x')
=
k_c(x_c,x'_c)
+k_g(x_g,x'_g)
+k_{cg}(x,x')
```

です。interactionとして

```math
k_{cg}=k'_ck'_g
```

を使う場合があります。

重要な規則は次です。

- category codeはidentifierであり連続量ではない
- continuous columnのみnormalizeする
- categorical kernelまたはfixed-feature enumerationを使う
- input perturbationでcategoryを変更する場合は遷移確率を定義する
- acquisition optimizerはdiscrete search spaceを尊重する

---

## 8. Beta回帰

Beta回帰は

```math
y\in(0,1)
```

の連続応答に適します。

```math
y\sim\mathrm{Beta}(\alpha,\beta).
```

mean-precision parameterizationでは

```math
\mu
=
\frac{\alpha}{\alpha+\beta},
\qquad
\phi=\alpha+\beta
```

```math
\alpha=\mu\phi,
\qquad
\beta=(1-\mu)\phi.
```

conditional varianceは

```math
\mathrm{Var}(Y\mid x)
=
\frac{\mu(x)[1-\mu(x)]}{\phi(x)+1}.
```

mean linkは

```math
\mu(x)=\sigma(f_\mu(x))
```

precisionには

```math
\phi(x)
=
\mathrm{softplus}(f_\phi(x))+\epsilon
```

を使えます。

構成には次があります。

- mean用GPとconstant precision
- mean用GPとglobal learnable precision
- meanとinput-dependent precisionの2 latent function

最後の構成だけが完全なinput-dependent dispersionを表します。

### 0と1の扱い

ordinary Beta distributionは0と1をsupportに含みません。選択肢は次です。

- zero/one-inflated Beta
- 物理的に妥当なclip
- finite-sample correction
- 別のlikelihood

boundary処理はtail predictionへ影響します。

---

## 9. Gamma回帰

Gamma回帰は正の連続応答

```math
y>0
```

に適します。

shape $a$、rate $b$なら

```math
y\sim\mathrm{Gamma}(a,b),
```

```math
\mathbb E[Y]=\frac ab,
\qquad
\mathrm{Var}(Y)=\frac{a}{b^2}.
```

shapeとscale $\theta$なら

```math
\mathbb E[Y]=a\theta,
\qquad
\mathrm{Var}(Y)=a\theta^2.
```

第2parameterがrateかscaleかを必ず明記します。

positive parameterには`exp`または`softplus + epsilon`を使えます。

適用例はduration、positive intensity、右に歪んだ連続量、meanとともにvarianceが増える応答です。0を含む場合はhurdle／zero-inflated modelが必要です。

---

## 10. Poisson回帰

count responseに対して

```math
y\mid x
\sim
\mathrm{Poisson}(\lambda(x)),
\qquad
\lambda(x)>0
```

とします。

log linkは

```math
\lambda(x)=\exp(f(x)).
```

です。

```math
\mathbb E[Y\mid x]=\lambda(x),
\qquad
\mathrm{Var}(Y\mid x)=\lambda(x).
```

mean=varianceという仮定は強い制約です。

exposure $e(x)$が異なる場合は

```math
\lambda(x)=e(x)\exp(f(x))
```

としてoffsetを入れます。exposure差を無視すると、入力効果を誤認する可能性があります。

---

## 11. Negative Binomial回帰

Negative Binomialはoverdispersed countを扱います。mean-dispersion parameterizationの例は

```math
\mathbb E[Y\mid x]=\mu(x)
```

```math
\mathrm{Var}(Y\mid x)
=
\mu(x)+\frac{\mu(x)^2}{r(x)}.
```

$r$が大きいほどPoissonへ近づきます。

```math
r\rightarrow\infty
\quad\Longrightarrow\quad
\mathrm{Var}(Y)\rightarrow\mu.
```

構成として、mean GP + constant dispersion、mean GP + global dispersion、meanとdispersionの2 latent functionがあります。

count varianceがmeanを大きく上回り、その差がmissing covariateやzero inflationで説明できない場合に候補になります。

---

## 12. PoissonとNegative Binomialの比較

| 性質 | Poisson | Negative Binomial |
|---|---|---|
| support | nonnegative integer | nonnegative integer |
| mean | $\lambda$ | $\mu$ |
| variance | $\lambda$ | $\mu+\mu^2/r$ |
| overdispersion | 表現しない | 表現する |
| parameter数 | 少ない | 多い |
| dispersion identifiability | 不要 | small dataでは弱い場合がある |

RMSEだけでなくpredictive log likelihood、zero frequency、upper-tail calibrationを比較します。

---

## 13. 応答変換とdirect non-Gaussian likelihood

正の応答を

```math
z=\log y
```

へ変換してGaussian回帰する方法があります。これはGamma回帰とは異なるmodelです。

log-Gaussian modelでは

```math
\log Y=f(x)+\varepsilon.
```

$\log Y$がGaussianなら、original spaceのmeanは

```math
\mathbb E[Y]
=
\exp\left(\mu+\frac12\sigma^2\right)
```

であり、単純な$\exp(\mu)$ではありません。

変換modelはsimpleで安定しやすく、direct likelihoodはsupportとmean-variance relationを直接表現できます。predictive distributionで比較してください。

---

## 14. Model familyごとのfitting

### Gaussian exact GP

data sizeとcovariance structureが許せばexact marginal likelihoodを使います。

### Non-Gaussian GP

variational ELBOなどのapproximate inferenceを使います。

### Deep non-Gaussian GP

DeepApproximateMLLとlikelihood samplingを使います。

### Heteroscedastic approximation

mean modelとnoise／dispersion modelを別stageまたはjoint optimizerでfitします。

fit functionはmodelに合わせる必要があります。VAE-GPやDeepGPへgeneric exact-GP fittingを適用してはいけません。

---

## 15. Posterior-space requirement

回帰wrapperは`posterior()`が何を返すかを明記します。

1. latent GP value
2. likelihood mean
3. predictive observation
4. transformed response
5. non-Gaussian responseへのnormal proxy

non-Gaussian modelの`posterior.mean`は、latent mean、response parameter mean、response expectation、Monte Carlo approximationのいずれかです。

BO threshold、`best_f`、LSE thresholdは同じ空間に置きます。

---

## 16. Acquisitionへの影響

### Gaussian回帰

standard EI、NEI、UCB、KG、LSEを直接適用できます。

### Beta回帰

bounded response spaceまたはutility spaceでobjectiveを定義します。latent Gaussian thresholdはresponse thresholdと一致しません。

### Count回帰

expected count、count exceedance probability、count utilityを使います。predictive distributionはdiscreteかつasymmetricです。

### Gamma回帰

response-space probabilityまたはsampleを使います。

最終応答がnon-Gaussianなのにnormal acquisitionを使う場合は、proxy approximationであることを明記します。

---

## 17. Model checking

### Gaussian residual

- residual mean
- residual versus fitted
- heteroscedasticity
- heavy tail
- outlier
- omitted-variable pattern

### Probability integral transform

continuous predictive CDF $F_i$に対して

```math
u_i=F_i(y_i)
```

を計算し、calibrated modelならuniformに近いことを確認します。

### Count calibration

- zero frequency
- region別mean／variance
- upper-tail count
- predictive interval
- deviance residual

### Bounded response

boundary近傍のmassとpredictive quantileを確認します。

---

## 18. 評価指標

### Point prediction

- RMSE
- MAE
- Poisson／Gamma deviance

### Probabilistic prediction

- negative log predictive density
- continuous ranked probability score
- interval coverage
- tail exceedance calibration

### Sequential decision

- regret
- target achievement probability
- feasible regret
- LSE set error
- repeated execution下のrobust performance

少しRMSEが悪いmodelでも、uncertainty calibrationが良ければBO decisionは改善する場合があります。

---

## 19. `bochan`実装との対応

### Gaussian regression

`src/bochan/api/model_registry.py`は、基本構成を次へ解決します。

| 構成 | model |
|---|---|
| continuous base | BoTorch `SingleTaskGP` |
| mixed base | BoTorch `MixedSingleTaskGP` |
| mixed Kronecker multitask | `MixedKroneckerMultiTaskGP` |
| high-dimensional | SAAS、PCA、REMBO、VAE wrapper |
| deep | DeepGP、Deep Kernel wrapper |
| robust | heteroscedastic、relevance-pursuit wrapper |

Gaussian source tree：

```text
src/bochan/models/regression/gaussian/base/
src/bochan/models/regression/gaussian/deep/
src/bochan/models/regression/gaussian/high_dim/
src/bochan/models/regression/gaussian/robust/
```

Non-Gaussian source tree：

```text
src/bochan/models/regression/non_gaussian/beta/
src/bochan/models/regression/non_gaussian/gamma/
src/bochan/models/regression/non_gaussian/poisson/
src/bochan/models/regression/non_gaussian/negative_binomial/
```

fit function：

```text
src/bochan/fit/
```

acquisition：

```text
src/bochan/acquisition/regression/
src/bochan/acquisition/non_gaussian/
```

model familyが返すposterior spaceとacquisitionの仮定を一致させます。

---

## 20. Model選択チェックリスト

1. 応答のsupportは何か
2. 0やboundary valueを含むか
3. continuousかcountか
4. varianceはmeanまたはinputで変化するか
5. overdispersionがあるか
6. outlierやheavy tailがあるか
7. output間相関が必要か
8. 全outputが共通inputで観測されるか
9. response transformは科学的に解釈可能か
10. wrapper posteriorはlatentかresponseか
11. 必要なfit functionは何か
12. acquisitionの仮定は成立するか
13. predictiveとsequentialの両方で何を比較するか

---

## 21. 参考文献

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- McCullagh and Nelder, *Generalized Linear Models*.
- Ferrari and Cribari-Neto, *Beta Regression for Modelling Rates and Proportions*, 2004.
- Hilbe, *Negative Binomial Regression*.
