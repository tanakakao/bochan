# 14. 深層・高次元ガウス過程モデル

本章では、標準的なstationary GPを拡張するDeep Kernel Learning（DKL）、Deep Gaussian Process（DeepGP）、SAAS、PCA、REMBO、VAE-GPを整理します。

これらはすべて「高次元に強いmodel」と呼ばれることがありますが、仮定する低次元構造が異なります。

- DKL：deterministic neural feature mapの中でGPを構築
- DeepGP：複数のstochastic GP layerをcomposition
- SAAS：元の入力次元の大部分がirrelevantというsparsity prior
- PCA：data varianceが低次元linear subspaceへ集中
- REMBO：objectiveが未知の低次元linear subspaceに依存
- VAE-GP：data manifoldがnonlinear latent spaceで表現可能

---

## 1. 高次元GPが難しい理由

入力次元$d$が増えると、限られた観測数$n$では次の問題が生じます。

- length scaleの識別が難しい
- distance concentrationによってstationary kernelの差が小さくなる
- acquisition optimizationが困難になる
- irrelevant dimensionがexplorationを分散させる
- categorical／mixed variableとの組合せが増える
- local optimumやoptimizer failureが増える

高次元modelは、dataだけでなく「どのような低次元構造を仮定するか」を選択しています。

---

## 2. Deep Kernel Learning

DKLではdeterministic neural feature map

```math
\phi_\psi:\mathbb R^d\rightarrow\mathbb R^p
```

を学習し、latent feature上でGPを定義します。

```math
f(x)=g(\phi_\psi(x)),
\qquad
 g\sim\mathcal{GP}(m,k).
```

resulting kernelは

```math
k_{\mathrm{DKL}}(x,x')
=
k(\phi_\psi(x),\phi_\psi(x')).
```

です。

DKLはdeterministic deep transform + GPであり、DeepGPとは異なります。small dataではfeature extractorのoverfit、representation uncertaintyの欠落、feature collapseに注意します。

---

## 3. DKLの学習目的

Gaussian exact GPなら

```math
\max_{\psi,\theta}
\log p(\mathbf y\mid\phi_\psi(X),\theta)
```

variational likelihoodなら

```math
\max_{\psi,\theta,\lambda}
\mathcal L_{\mathrm{ELBO}}
```

を使います。

feature dimension、weight decay、early stopping、feature normalization、frozen encoder baselineなどを検討します。predictionだけでなくcalibrationを確認します。

---

## 4. Deep Gaussian Process

DeepGPはGPをcompositionします。

```math
\mathbf h_1(x)=f_1(x),
```

```math
\mathbf h_2(x)=f_2(\mathbf h_1(x)),
```

```math
y=f_L(\mathbf h_{L-1}(x))+\varepsilon.
```

各layerがrandom functionなので、hidden representationもstochasticです。

| 項目 | DKL | DeepGP |
|---|---|---|
| hidden mapping | deterministic NN | stochastic GP |
| uncertainty | final GP中心 | hidden layerにも存在 |
| inference | exact／variational | deep variational |
| runtime | 比較的軽い | 重い |

---

## 5. DeepGPのvariational objective

各layerに

```math
q(\mathbf u_l)=\mathcal N(\mathbf m_l,S_l)
```

を置きます。概念的なELBOは

```math
\mathcal L
=
\mathbb E_{q(f_1,\ldots,f_L)}
[\log p(\mathbf y\mid f_L)]
-
\sum_{l=1}^{L}
\mathrm{KL}[q(\mathbf u_l)\|p(\mathbf u_l)].
```

GPyTorchでは`DeepApproximateMLL`がELBOを包みます。

forwardに

```text
L x batch_shape x q x m
```

のsample axisが追加される場合があります。class axisやt-batchと混同しないようにします。

---

## 6. Predictive moment reduction

sample$s$ごとのmean／varianceを$\mu_s,v_s$とすると

```math
\bar\mu=rac1S\sum_s\mu_s
```

```math
\bar v
=
\frac1S\sum_s(v_s+\mu_s^2)-\bar\mu^2.
```

varianceだけをaverageするとbetween-sample mean variationを失います。wrapperがDeepGP sample axisをcollapseする方法を文書化します。

---

## 7. SAAS

inverse length scaleを

```math
\rho_j=\frac1{\ell_j}
```

とし、global shrinkage parameter$\tau$の下でstrong shrinkage priorを置きます。

```math
\rho_j\mid\tau
\sim
\mathrm{HalfCauchy}(\tau),
```

```math
\tau\sim\mathrm{HalfCauchy}(\tau_0).
```

多くのdimensionはlow sensitivityへ縮み、少数axisだけがrelevantになります。

適する仮定は「relevant original dimensionが少数」です。rotated subspaceやnonlinear manifoldには直接対応しません。

fully Bayesian SAASはNUTS sampleによるmodel-batch axisを持ちます。

---

## 8. MAP-SAAS

```math
\widehat\theta_{\mathrm{MAP}}
=
\arg\max_\theta
[
\log p(\mathbf y\mid\theta)
+
\log p(\theta)
].
```

fully Bayesian SAASより高速ですが、hyperparameter uncertaintyを積分せず、local optimumに依存します。

---

## 9. PCA

centered input

```math
X_c=X-\bar X
```

へSVD

```math
X_c=U\Sigma V^\top
```

を行い、

```math
z=V_p^\top(x-\bar x)
```

へprojectionします。

input varianceの大きい方向がtargetにも重要という仮定です。target relevanceを使わず、low-variance important directionを失う可能性があります。

---

## 10. REMBO

random matrix

```math
A\in\mathbb R^{d\times p},
\qquad p\ll d
```

を使い

```math
x=Az
```

または

```math
x=\Pi_{\mathcal X}(Az)
```

とします。

rotated low-dimensional subspaceへ対応できますが、random embedding variability、projectionのmany-to-one mapping、latent bounds、mixed inputとの整合が課題です。

---

## 11. VAE-GP

VAE objectiveは

```math
\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{q_\phi(z\mid x)}
[\log p_\theta(x\mid z)]
-
\mathrm{KL}[q_\phi(z\mid x)\|p(z)].
```

latent representation

```math
z=E_\phi(x)
```

上でGPをfitします。

nonlinear manifoldとdecoderを利用できますが、VAE geometryがobjective-relevantとは限らず、decoded candidateがdomain constraintを満たさない場合があります。

mixed inputではcontinuous reconstructionとcategorical cross entropyを分けます。category codeをcontinuous MSEで再構成してはいけません。

---

## 12. Raw-spaceとlatent-space optimization

### Raw-space optimization

raw candidate$x$をoptimizationし、model内部で$z=T(x)$へ変換します。raw constraintを直接扱えます。

### Latent-space optimization

$z$をoptimizationし、

```math
x=D(z)
```

でdecodeします。dimensionは低くなりますが、decoded validity、repair、many-to-one mappingが課題です。

---

## 13. Model比較

| model | 仮定する構造 | 主な利点 | 主なリスク |
|---|---|---|---|
| ARD GP | axisごとのsmoothness | simple | high-d small-nで不安定 |
| SAAS | relevant original axisが少数 | sparse high-d | rotated structureに弱い |
| PCA-GP | input varianceがlow-rank linear | fast | target relevanceなし |
| REMBO | objectiveがlow-d linear subspace | latent BO | random embedding |
| DKL | deterministic nonlinear feature | expressive | representation uncertainty不足 |
| DeepGP | stochastic hierarchy | rich uncertainty | heavy computation |
| VAE-GP | nonlinear generative manifold | decoder利用 | invalid decode、unsupervised mismatch |

---

## 14. Validation

- predictive RMSE／log loss／NLPD
- calibration／coverage
- latent dimension sensitivity
- reconstruction error
- decoded validity rate
- category reconstruction accuracy
- regret／hypervolume regret
- optimizer success／duplicate rate
- wall-clock
- random seed／embedding sensitivity

DeepGP sample reduction、SAAS posterior convergence、DKL feature collapse、VAE uncertainty propagationも確認します。

---

## 15. `bochan`実装との対応

```text
src/bochan/models/regression/gaussian/deep/
src/bochan/models/regression/gaussian/high_dim/
src/bochan/models/classification/binary/deep/
src/bochan/models/classification/binary/high_dim/
src/bochan/models/classification/multiclass/deep/
src/bochan/models/classification/multiclass/high_dim/
src/bochan/models/ordinal/deep/
src/bochan/models/ordinal/high_dim/
src/bochan/fit/
src/bochan/api/registry/model.py
src/bochan/api/factory.py
```

確認事項はraw／internal dimension、`train_inputs_raw`、transform位置、posterior extra axis、conditioning support、optimizerのsearch spaceです。

---

## 16. 選択checklist

1. irrelevant axis、linear manifold、nonlinear manifoldのどれか
2. original-axis sparsityを仮定できるか
3. representation学習に十分なdataがあるか
4. decoderが必要か
5. category／constraintをどう保持するか
6. fully Bayesian uncertaintyが必要か
7. extra sample axisをどう扱うか
8. calibrationを評価したか
9. optimization costを許容できるか
10. simple baselineと比較したか
