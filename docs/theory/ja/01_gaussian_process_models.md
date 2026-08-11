# 01. ガウス過程の基礎

本章では、`bochan`の各モデル族に共通するガウス過程（Gaussian Process; GP）の確率論と線形代数を整理します。GP prior、kernel、条件付き分布、周辺尤度、変分推論、posterior sample、BoTorch互換interfaceを対象とします。

応答型ごとの詳細は10〜14章を参照してください。

---

## 1. ガウス確率変数からガウス過程へ

```math
Z\sim\mathcal N(\mu,\sigma^2)
```

```math
\mathbf z\sim\mathcal N(\boldsymbol\mu,\Sigma)
```

とします。共分散行列$\Sigma$はpositive semidefiniteであり、任意の$\mathbf a$について

```math
\mathbf a^\top\Sigma\mathbf a\ge0
```

です。

ガウス過程

```math
f\sim\mathcal{GP}(m,k)
```

とは、任意の有限入力集合$X=(x_1,\ldots,x_n)$に対し

```math
\mathbf f_X
=[f(x_1),\ldots,f(x_n)]^\top
\sim
\mathcal N(\mathbf m_X,K_{XX})
```

となることです。

```math
[\mathbf m_X]_i=m(x_i),
\qquad
[K_{XX}]_{ij}=k(x_i,x_j).
```

GPは1本の曲線ではなく関数全体に対する分布です。

---

## 2. Kernel

共分散関数

```math
k(x,x')=\mathrm{Cov}[f(x),f(x')]
```

はsmoothness、amplitude、length scale、periodicity、interaction、task correlationを定義します。

### RBF

```math
k(x,x')
=
\sigma_f^2
\exp\left[-\frac{\|x-x'\|^2}{2\ell^2}\right].
```

### Matérn-5/2

```math
k(r)
=
\sigma_f^2
\left(1+\sqrt5r+\frac53r^2\right)
\exp(-\sqrt5r)
```

```math
r^2
=
\sum_j\frac{(x_j-x'_j)^2}{\ell_j^2}.
```

ARDでは各dimensionにlength scale$\ell_j$を持たせます。large length scaleはlow sensitivityを示しますが、causal importanceではありません。

additive kernelはfunction decomposition、product kernelはinteractionを表します。

---

## 3. Gaussian conditioning

training latent valueとtest latent valueをjoint Gaussianとし、観測modelを

```math
\mathbf y=\mathbf f+\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon\sim\mathcal N(0,\Sigma_n)
```

とします。

posterior meanは

```math
\boldsymbol\mu_*
=
\mathbf m_*
+
K_{X_*X}(K_{XX}+\Sigma_n)^{-1}
(\mathbf y-\mathbf m)
```

posterior covarianceは

```math
\Sigma_*
=
K_{X_*X_*}
-
K_{X_*X}(K_{XX}+\Sigma_n)^{-1}K_{XX_*}
```

です。実装ではinverseを直接作らずCholesky decompositionやlinear solveを使います。

---

## 4. Latentとpredictive distribution

latent posteriorは

```math
p(f_*\mid\mathcal D)
```

future observationのpredictive distributionは

```math
p(y_*\mid\mathcal D)
=
\int p(y_*\mid f_*)p(f_*\mid\mathcal D)\,df_*.
```

Gaussian noiseなら

```math
\mathrm{Var}(y_*\mid\mathcal D)
=
\mathrm{Var}(f_*\mid\mathcal D)+\sigma_n^2.
```

underlying functionの学習ではlatent variance、future measurement予測ではnoise込みvarianceを使います。

---

## 5. 周辺尤度

```math
K_y=K_{XX}+\Sigma_n
```

に対して

```math
\log p(\mathbf y\mid X,\theta)
=
-\frac12(\mathbf y-\mathbf m)^\top K_y^{-1}(\mathbf y-\mathbf m)
-\frac12\log|K_y|
-\frac n2\log(2\pi).
```

各項はdata fit、complexity penalty、normalizationです。training error最小化とは異なります。

---

## 6. Non-Gaussian likelihoodと変分推論

Bernoulli、Categorical、ordered-logit、Poissonなどではposteriorがclosed formになりません。

inducing variableを

```math
\mathbf u=f(Z),
\qquad
q(\mathbf u)=\mathcal N(\mathbf m_u,S_u)
```

とし、ELBO

```math
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q(\mathbf f)}[\log p(\mathbf y\mid\mathbf f)]
-
\mathrm{KL}[q(\mathbf u)\|p(\mathbf u)]
```

を最大化します。

---

## 7. Fully Bayesian GPと多出力

hyperparameter uncertaintyを積分すると

```math
p(f_*\mid\mathcal D)
=
\int p(f_*\mid\mathcal D,\theta)
p(\theta\mid\mathcal D)\,d\theta.
```

posterior tensorにはmodel-batch axisが追加されます。

multi-output covarianceの一例は

```math
\mathrm{Cov}[f_a(x),f_b(x')]
=B_{ab}k_X(x,x').
```

共通input gridでは$B\otimes K_X$というKronecker構造を持つ場合があります。task covarianceはraw target correlationとは異なります。

---

## 8. Transform

input normalization：

```math
\tilde x_j=\frac{x_j-l_j}{u_j-l_j}.
```

outcome standardization：

```math
\tilde y=\frac{y-\bar y}{s_y}.
```

mixed inputではcontinuous columnだけをnormalizeし、category valueを保持します。

`InputPerturbation`はevaluation時に

```text
batch_shape x q x d
    -> batch_shape x (q * n_w) x d
```

へ展開します。training時に無条件で適用してはいけません。

---

## 9. Posterior sample

```math
f^{(s)}=\mu+L\epsilon^{(s)},
\qquad
\epsilon^{(s)}\sim\mathcal N(0,I),
\qquad
LL^\top=\Sigma.
```

reparameterizationによりcandidate$X$へgradientを流せます。

重要なinterface：

```python
posterior.mean
posterior.variance
posterior.rsample(sample_shape)
posterior.rsample_from_base_samples(...)
```

custom posteriorはexact multivariate Gaussianとは限らず、probability／hybrid posteriorはproxy samplingの場合があります。

---

## 10. 数値安定性と評価

- jitterは数値安定化項でありscientific noiseではない
- duplicate、extreme length scale、small noise、bad scalingはcondition numberを悪化させる
- Cholesky failureではdouble precision、normalization、jitter、prior、model simplificationを検討する
- constant posteriorや全点zero varianceはtransform／shape／fitを確認する

評価はRMSE／MAEだけでなく、NLPD、coverage、calibration、BO regret、Active Learning loss、LSE set errorを使います。

---

## 11. `bochan`実装との対応

| 数学的対象 | 実装概念 |
|---|---|
| GP prior | GPyTorch modelの`forward()`が返す`MultivariateNormal` |
| exact MLL | `ExactMarginalLogLikelihood` |
| variational posterior | `VariationalStrategy` |
| variational objective | `VariationalELBO` |
| deep objective | `DeepApproximateMLL` |
| predictive posterior | `GPyTorchPosterior`またはcustom posterior |
| normalization | BoTorch `Normalize` |
| standardization | BoTorch `Standardize` |

source：

```text
src/bochan/models/
src/bochan/models/components/
src/bochan/models/transforms/
src/bochan/models/regression/gaussian/likelihood.py and family-local likelihood modules
src/bochan/fit/
```

posterior contract：

- regression：`posterior(X, observation_noise=...)`
- binary：probabilityは`posterior(X)`、latentは`latent_posterior(X)`
- multiclass：probabilityは`posterior(X)`、latentは`latent_posterior(X)`
- ordinal：latentは`posterior(X)`、probabilityは`class_probs(X)`
- hybrid：mode-specific `HybridPosterior`
