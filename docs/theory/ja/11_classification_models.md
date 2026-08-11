# 11. 分類モデル

ガウス過程分類は、1つまたは複数のlatent Gaussian processとnon-Gaussian likelihoodを用いて離散labelをモデル化します。本章ではbinary／multiclass classificationのlikelihood、variational inference、probability marginalization、calibration、および現在の`bochan` posterior contractを整理します。

意思決定基準は別章で扱います。

- 04章：entropy、BALD、marginなどの能動学習
- 06章：分類出力をBO objective／constraintとして利用する方法
- 16章：分類LSEの実装式

本章で最も重要な区別は次です。

> latent-function uncertainty、class-probability functionのposterior uncertainty、future class labelのrandomnessは別の量である。

---

## 1. Binary classification

labelを

```math
y_i\in\{0,1\}
```

とします。scalar latent functionに

```math
f\sim\mathcal{GP}(m,k)
```

を置き、inverse link $\pi$を通して

```math
P(Y=1\mid f)=\pi(f),
\qquad
P(Y=0\mid f)=1-\pi(f)
```

とします。

### Logistic link

```math
\pi(f)
=
\sigma(f)
=
\frac{1}{1+\exp(-f)}.
```

### Probit link

```math
\pi(f)=\Phi(f).
```

GPyTorch標準の`BernoulliLikelihood`はprobit-style constructionを使います。binary modelのprobabilityが必ずlogistic sigmoidから生成されたと仮定してはいけません。

---

## 2. Latent scaleの解釈

logistic linkでは

```math
\log
\frac{P(Y=1\mid f)}{P(Y=0\mid f)}
=f
```

なので、latent valueはlog oddsです。

probit linkでは、latent continuous variable

```math
z=f+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,1)
```

を導入し、

```math
Y=\mathbf1[z>0]
```

とすると

```math
P(Y=1\mid f)=\Phi(f)
```

を得ます。

linkが異なるmodel間でlatent scaleを直接比較してはいけません。同じclass probabilityでもlatent valueは異なります。

---

## 3. Non-conjugate posterior

binary likelihoodは

```math
p(\mathbf y\mid\mathbf f)
=
\prod_{i=1}^{n}
\pi(f_i)^{y_i}
[1-\pi(f_i)]^{1-y_i}
```

です。

posteriorは

```math
p(\mathbf f\mid\mathbf y)
\propto
p(\mathbf y\mid\mathbf f)
\mathcal N(\mathbf f;\mathbf m,K)
```

ですが、Bernoulli likelihoodはGaussianではないため、posteriorもevidenceもclosed formではありません。

代表的な近似はLaplace approximation、expectation propagation、variational inference、MCMCです。現在の`bochan` base classification modelはsparse variational GPを使います。

---

## 4. Sparse variational binary GP

inducing inputを

```math
Z=(z_1,\ldots,z_M)
```

inducing variableを

```math
\mathbf u=f(Z)
```

とします。

```math
q(\mathbf u)
=
\mathcal N(\mathbf m_u,S_u)
```

を導入し、

```math
q(\mathbf f)
=
\int
p(\mathbf f\mid\mathbf u)
q(\mathbf u)
\,d\mathbf u
```

を得ます。

ELBOは

```math
\mathcal L_{\mathrm{ELBO}}
=
\sum_{i=1}^{n}
\mathbb E_{q(f_i)}
[\log p(y_i\mid f_i)]
-
\mathrm{KL}[q(\mathbf u)\|p(\mathbf u)].
```

最適化対象はvariational mean／covariance、kernel parameter、mean parameter、learnable inducing locationなどです。

重要な設定は次です。

- inducing point数と初期位置
- inducing locationをlearnするか
- kernelとARD次元
- optimizer learning rate
- epoch数、minibatch size
- class imbalanceの扱い

---

## 5. Predictive class probability

test inputでのlatent posteriorを

```math
q(f_*)
=
\mathcal N(\mu_f,\sigma_f^2)
```

とします。predictive probabilityは

```math
p_*(x)
=
P(Y=1\mid x,\mathcal D)
=
\int
\pi(f_*)q(f_*)\,df_*
```

です。

一般に

```math
p_*(x)\ne\pi(\mu_f)
```

です。plug-in probabilityはlatent uncertaintyを無視します。

probit linkとGaussian latent posteriorなら

```math
\int
\Phi(f)
\mathcal N(f;\mu,\sigma^2)
\,df
=
\Phi\left(
\frac{\mu}{\sqrt{1+\sigma^2}}
\right).
```

latent uncertaintyが大きいほどprobabilityは0.5方向へ寄ります。

---

## 6. Binary classificationの3種類の不確かさ

### 6.1 Latent posterior variance

```math
V_f(x)
=
\mathrm{Var}[f(x)\mid\mathcal D].
```

latent decision functionそのもののposterior uncertaintyです。

### 6.2 Probability-function variance

latent posterior sampleから

```math
p^{(s)}(x)=\pi(f^{(s)}(x))
```

を生成すると、

```math
V_p(x)
=
\mathrm{Var}_s[p^{(s)}(x)]
```

はclass probability functionのposterior uncertaintyです。

### 6.3 Bernoulli observation variance

fixed probability $p$に対して

```math
V_Y(x)=p(x)[1-p(x)]
```

はfuture binary labelのrandomnessです。$p$を完全に知っていても$p=0.5$なら最大になります。

これらをすべてclassification varianceと呼ぶと、能動学習やUCBの意味を誤ります。

---

## 7. `bochan`のbinary posterior contract

主なclassは

```text
BinaryClassificationGPModel
BinaryClassificationMixedGPModel
```

です。

source：

```text
src/bochan/models/classification/binary/base/models.py
```

public accessorは

```python
probability_posterior = model.posterior(X)
latent_posterior = model.latent_posterior(X)
```

です。

### `posterior(X)`

1. input transformを適用
2. latent variational GPを評価
3. Bernoulli likelihoodを適用
4. `SimpleBernoulliPosterior`を構築

meanは

```math
P(Y=1\mid x,\mathcal D)
```

です。varianceはpredictive Bernoulli distributionに基づき、robust wrapperによっては補助noiseが加算されます。

### `latent_posterior(X)`

likelihoodを通さず、latent $f(x)$の`GPyTorchPosterior`を返します。

binary LSEのlatent boundaryと、BOのprobability objectiveでaccessorが異なる点に注意してください。

---

## 8. `SimpleBernoulliPosterior`

custom posteriorはBoTorch-style interfaceを持ちます。

```text
mean:     batch_shape x q x 1
variance: batch_shape x q x 1
```

source：

```text
src/bochan/models/classification/binary/base/posterior.py
```

Bernoulli random variableはdiscreteですが、多くのBoTorch MC acquisitionはdifferentiable reparameterized sampleを期待します。continuous probability sampleやnormal proxyを使う場合、それはexact label samplingではなく、実装上のposterior approximationとして解釈します。

---

## 9. Binary decision boundary

symmetric monotone linkなら

```math
f(x)=0
```

はlink-levelで

```math
P(Y=1\mid f)=0.5
```

に対応します。

probability threshold $\tau_p$のlatent thresholdは

```math
\tau_f=\pi^{-1}(\tau_p)
```

です。

ただしmarginal predictive probability

```math
P(Y=1\mid x,\mathcal D)
=
\int\pi(f)q(f)\,df
```

はlatent varianceにも依存します。そのためposterior mean $\mu_f=\tau_f$とmarginal probability contour $p=\tau_p$は必ずしも一致しません。

---

## 10. Class imbalance

class proportionが大きく偏ると、overall accuracyは高くてもminority classを捉えられないmodelになります。

対策例：

- stratified initial design
- weighted likelihoodまたはresampling
- calibrated decision threshold
- utility-sensitive objective
- target classを重視した能動学習
- class-specific metric

decision threshold変更はprobability modelの再学習ではありません。weighted trainingはcalibration自体を変える可能性があるため別途検証します。

---

## 11. Binary calibration

calibrated modelは概念的に

```math
P(Y=1\mid p(X)=r)=r
```

を満たします。

### Brier score

```math
\mathrm{BS}
=
\frac1n
\sum_i(p_i-y_i)^2.
```

### Log loss

```math
-\frac1n
\sum_i
[y_i\log p_i+(1-y_i)\log(1-p_i)].
```

### Reliability diagram

predictionをbinに分け、mean predicted probabilityとobserved frequencyを比較します。

BOや能動学習ではinput distributionがadaptiveに変化します。random validation dataでのcalibrationがacquisition-selected regionでも成立するとは限りません。

---

## 12. Multiclass classification

labelを

```math
y\in\{0,1,\ldots,K-1\}
```

とします。class-wise latent functionを

```math
f_k(x),
\qquad
k=0,\ldots,K-1
```

とし、

```math
\mathbf f(x)
=
[f_0(x),\ldots,f_{K-1}(x)]
```

を構成します。

categorical likelihoodは

```math
p_k(x)
=
P(Y=k\mid\mathbf f(x))
```

を返します。

---

## 13. Softmax likelihood

```math
p_k
=
\frac{\exp(f_k/T)}
{\sum_{j=0}^{K-1}\exp(f_j/T)}
```

です。$T>0$はtemperatureです。

### Shift invariance

```math
\mathrm{softmax}(\mathbf f+a\mathbf1)
=
\mathrm{softmax}(\mathbf f).
```

absolute logitはidentifiableではなく、class間の差のみが意味を持ちます。

### Temperature

- $T<1$：sharper probability
- $T>1$：flatter probability

post-hoc temperature scalingはcalibrationを改善できますが、別のcalibration dataでfitする必要があります。

---

## 14. Class-batched variational GP

現在のmulticlass base modelはclass-wise batched latent SVGPを使います。概念的には

```math
f_k\sim\mathcal{GP}(m_k,k_k)
```

です。

inducing pointとkernelのbatch shapeは

```text
[num_classes]
```

です。variational objectiveは

```math
\mathcal L
=
\sum_i
\mathbb E_{q(\mathbf f_i)}
[\log P(y_i\mid\mathbf f_i)]
-
\sum_k
\mathrm{KL}[q(\mathbf u_k)\|p(\mathbf u_k)].
```

latent GPはclass-wise batchとして表現され、class probability間のdependenceはsoftmax normalizationを通じて生じます。

---

## 15. Multiclass shape

acquisition inputが

```text
batch_shape x q x d
```

の場合、wrapperはclass batch axisを追加し

```text
batch_shape x 1 x q x d
```

とします。内部で

```text
batch_shape x K x q x d
```

へbroadcastされます。

probability posteriorは

```text
batch_shape x q x K
```

を返します。

class batch axisをDeepGP sample axisやt-batchとしてaverageしてはいけません。

---

## 16. `bochan`のmulticlass posterior contract

主なclassは

```text
MulticlassClassificationGPModel
MulticlassClassificationMixedGPModel
```

です。

source：

```text
src/bochan/models/classification/multiclass/base/models.py
```

public accessor：

```python
latent = model.latent_posterior(X)
probability = model.posterior(X)
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
```

### `latent_posterior(X)`

class-batched latent logitの`GPyTorchPosterior`を返します。

### `posterior(X)`

latent posteriorを`MulticlassProbsPosterior`へ包みます。

### `class_probs(X)`

```text
batch_shape x q x K
```

のprobability meanを返します。

posterior helper：

```text
src/bochan/models/classification/multiclass/_components.py
```

---

## 17. Multiclass predictive probability

predictive probabilityは

```math
p_k(x)
=
\mathbb E_{q(\mathbf f(x))}
\left[
\frac{\exp(f_k/T)}
{\sum_j\exp(f_j/T)}
\right]
```

です。

一般に

```math
p_k(x)
\ne
\frac{\exp(\mathbb E[f_k]/T)}
{\sum_j\exp(\mathbb E[f_j]/T)}.
```

`MulticlassProbsPosterior`はlatent posteriorから、現行acquisitionに必要なprobability momentsやsample approximationを構築します。

---

## 18. Multiclass uncertainty

### Predictive entropy

```math
H(Y\mid x,\mathcal D)
=
-\sum_kp_k\log p_k.
```

### Top-two margin

```math
p_{(1)}-p_{(2)}.
```

marginが小さいほどclass decisionが曖昧です。

### Probability covariance

posterior sampleから

```math
\mathrm{Cov}[\mathbf p(x)]
```

を計算できます。probabilityはsum-to-one constraintを持つため、full $K$-dimensional covarianceはsingularです。

### Categorical observation covariance

one-hot label $e_Y$に対して

```math
\mathrm{Cov}(e_Y\mid\mathbf p)
=
\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top.
```

これはfuture label randomnessであり、probability functionのposterior covarianceではありません。

---

## 19. Target classとclass set

single target class：

```math
p_{k^*}(x).
```

acceptable class set $A$のunion probability：

```math
P(Y\in A\mid x)
=
\sum_{k\in A}p_k(x).
```

classはmutually exclusiveなので、sumは直接的なprobabilityです。mean、max、minは別のscoreであり、同じ確率解釈を持ちません。

---

## 20. Mixed-input classification

continuous dimension $C$とcategorical dimension $G$に対して、mixed kernelは概念的に

```math
k(x,x')
=
k_C+k_G+k'_Ck'_G
```

です。

現行実装は次を行います。

- continuous columnのみnormalize
- category columnへ`CategoricalKernel`
- `input_transform`がcategoryを変更しないことを検証
- multiclass class batchに対応するkernel batch
- `InputPerturbation`でqが展開されてもcategoryを保持

---

## 21. `condition_on_observations`

variational classificationではexact Gaussian rank updateを使えません。

binary／multiclass wrapperは、old dataとnew dataを結合したmodelを再構築し、learned stateをcopyする形で

```python
condition_on_observations(X, Y)
```

を近似的に提供できます。

full variational refitを行わない場合、look-ahead fantasyはapproximationです。fantasy dimensionやcondition semanticsをacquisitionごとに検証します。

---

## 22. 異分散classificationの意味

classification likelihood自体がstochasticです。追加noise modelを使う場合、そのgenerative meaningを定義します。

### Label corruption

```math
P(Y_{\mathrm{obs}}=1)
=
[1-\rho(x)]p(x)+\rho(x)[1-p(x)].
```

### Input-dependent temperature

```math
p_k
=
\mathrm{softmax}
\left(
\frac{f_k}{T(x)}
\right).
```

### Probability-estimation uncertainty

training targetがindividual labelではなくestimated class proportionの場合、外部推定誤差を持つ場合があります。

現行robust wrapperがauxiliary noiseをposterior varianceやacquisition scoreへ加えるだけなら、fully specified noisy-label modelではなくnoise-aware engineering conventionとして説明します。詳細は13章です。

---

## 23. Deep・高次元variant

classification model familyには、DeepGP、Deep Kernel GP、Deep Kernel DeepGP、SAAS、embedding／decomposition、heteroscedastic／robust variantがあります。

deep／high-dimensional componentはlatent representationを変えますが、class labelのsemantic meaningやlikelihoodの区別は変わりません。14章を参照してください。

---

## 24. 評価

### Binary

- log loss
- Brier score
- ROC-AUC、PR-AUC
- operational thresholdでのsensitivity／specificity
- calibration
- minority class error

### Multiclass

- categorical log loss
- macro／weighted F1
- class-wise recall
- confusion matrix
- multiclass Brier score
- class-wise calibration

### Sequential decision

predictive metricだけでなく、BO regret、target achievement、Active Learning loss、LSE boundary errorをsample countに対して評価します。

---

## 25. Source map

| component | source |
|---|---|
| binary base model | `src/bochan/models/classification/binary/base/models.py` |
| binary posterior | `src/bochan/models/classification/binary/base/posterior.py` |
| binary robust | `src/bochan/models/classification/binary/robust/` |
| binary deep | `src/bochan/models/classification/binary/deep/` |
| binary high-dimensional | `src/bochan/models/classification/binary/high_dim/` |
| multiclass base | `src/bochan/models/classification/multiclass/base/models.py` |
| multiclass posterior helper | `src/bochan/models/classification/multiclass/_components.py` |
| multiclass robust | `src/bochan/models/classification/multiclass/robust/` |
| multiclass deep | `src/bochan/models/classification/multiclass/deep/` |
| multiclass high-dimensional | `src/bochan/models/classification/multiclass/high_dim/` |
| posterior transform | `src/bochan/models/transforms/posterior/classification.py` |
| high-level registry | `src/bochan/api/model_registry.py` |

---

## 26. Model checklist

1. binaryかmulticlassか
2. linkとlikelihoodは何か
3. latent GP kernelは何か
4. inducing point数・初期化は何か
5. class imbalanceをどう扱うか
6. `posterior()`はlatentかprobabilityか
7. probabilityをmarginalizeするかplug-inか
8. varianceは何を表すか
9. calibration手順は何か
10. mixed input categoryをどう扱うか
11. conditioning／fantasyをsupportするか
12. deep／high-dimensional representationは何か
13. predictive metricは何か
14. decision objective／constraintへどう変換するか

---

## 27. 参考文献

- Rasmussen and Williams, *Gaussian Processes for Machine Learning*, 2006.
- Nickisch and Rasmussen, *Approximations for Binary Gaussian Process Classification*, 2008.
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, 2009.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017.
