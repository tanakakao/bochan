# 16. レベル集合獲得関数の数式と実装対応

05章ではレベル集合推定（LSE）の問題、loss、confidence set、評価を定義しました。本章では、現在の`bochan` LSE classが実際に計算するscore、posterior space、reduction、source pathを整理します。

一般名に複数の定義がある場合、classの`forward()`で実装されている式を優先します。

---

## 1. 共通記号

candidate batchを

```math
X=[x_1,\ldots,x_q]
```

とし、

```math
\mu_i=\mu(x_i),
\qquad
v_i=\mathrm{Var}[f(x_i)],
\qquad
\sigma_i=\sqrt{v_i}
```

とします。thresholdを$h$、classificationのtarget probabilityを$p_i$、ordinal cutpointを$c_j$とします。

pointwise acquisitionの典型的処理順は次です。

```text
posterior / probability
    -> pointwise score
    -> duplicate / pending / observed penalty
    -> optional risk objective
    -> q*n_w to q reduction
    -> q reduction
    -> batch_shape
```

joint acquisitionはq-dimensional covarianceからbatch valueを直接構成します。

---

## 2. Regression Straddle

source：

```text
src/bochan/acquisition/regression/levelset_estimation/single_output.py
```

`qRegressionStraddle`のpointwise scoreは

```math
s_i
=
\beta\sigma_i-|\mu_i-h|.
```

- $\beta\sigma_i$：uncertainty reward
- $-|\mu_i-h|$：boundary proximity

current parameter`beta`はstandard deviationへ直接掛かります。

---

## 3. Regression Joint Straddle

q-batch covarianceを$\Sigma_X$とし、

```math
D(X)
=
\frac1q\sum_{i=1}^{q}|\mu_i-h|
```

とします。

```math
s(X)
=
-D(X)+\beta U(\Sigma_X).
```

uncertainty modeは次です。

### Trace

```math
U(\Sigma)=\mathrm{tr}(\Sigma).
```

### Log determinant

```math
U(\Sigma)
=
\log\det(\Sigma+\epsilon I).
```

### Log determinant of identity plus covariance

```math
U(\Sigma)
=
\log\det(I+\Sigma+\epsilon I).
```

logdetはjoint uncertainty volumeを評価し、correlated candidateのredundancyを抑えます。

---

## 4. Regression ICU・Boundary Variance

current ICU-style local scoreは

```math
s_i
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-h}{b_i}
\right)^2
\right]
\sigma_i.
```

`bandwidth`を指定しない場合は$b_i=\sigma_i$です。これはlocal contour-weighted uncertaintyであり、global contour lossのexact integrated reductionではありません。

Boundary Varianceは

```math
w_i
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-h}{\tau}
\right)^2
\right]
```

```math
s_i=v_iw_i
```

です。

---

## 5. Regression Probability of Exceedance

Gaussian posteriorで`mode="above"`なら

```math
s_i
=
P(f_i\ge h)
=
\Phi\left(
\frac{\mu_i-h}{\sigma_i}
\right).
```

`mode="below"`なら

```math
s_i
=
\Phi\left(
\frac{h-\mu_i}{\sigma_i}
\right).
```

interval$[l,u]$なら

```math
s_i
=
\Phi\left(
\frac{u-\mu_i}{\sigma_i}
\right)
-
\Phi\left(
\frac{l-\mu_i}{\sigma_i}
\right).
```

`temperature`を指定したpathではmean-space sigmoid approximationを使います。Probability of Exceedanceはmembership scoreであり、必ずしもboundary learning scoreではありません。

---

## 6. Regression risk objective

`RegressionLevelSetScoreObjective`はexpanded score

```text
... x (q * n_w)
```

を

```text
... x q x n_w
```

へreshapeします。

### Mean

```math
\bar s_i
=
\frac1{n_w}\sum_{r=1}^{n_w}s_{ir}.
```

### VaR／CVaR

`maximize` directionに従ってsortし、tail size

```math
k=\lceil\alpha n_w\rceil
```

を使います。これは計算済みLSE scoreへのrisk aggregationであり、robust latent response自体のLSEとは限りません。

---

## 7. Binary Latent Straddle

source：

```text
src/bochan/acquisition/binary/levelset_estimation/single_output.py
```

binary baseは`latent_posterior()`を使います。

```math
s_i
=
\beta\sigma_i
-
\sqrt{(\mu_i-h_f)^2+10^{-8}}.
```

default latent thresholdは$h_f=0$です。probability thresholdをtargetにする場合はlinkとの対応を確認します。

---

## 8. Binary Joint Latent Straddle

```math
s(X)
=
\beta U(\Sigma_f)
-D(\boldsymbol\mu_f,h_f).
```

uncertainty modeには`logdet1p`、`logdet`、`sqrt_trace`があります。

例：

```math
U_{\mathrm{logdet1p}}(\Sigma)
=
\frac12\log\det\left(
I+\frac{\Sigma}{\tau^2}
\right).
```

boundary distanceにはmean absolute、root mean square、maximum absoluteがあります。

`marginalize_pending=True`ではpendingを含むjoint scoreからpendingだけのscoreを引くincremental approximationを使います。full fantasy conditioningではありません。

---

## 9. Binary ICU・entropy

binary ICUはprobability posterior mean$p_i$に対して

```math
s_i=4p_i(1-p_i)
```

を使います。$p=0.5$で1、0または1で0です。

class entropyは

```math
s_i
=-p_i\log p_i
-(1-p_i)\log(1-p_i).
```

です。いずれもpredictive class ambiguityであり、BALDやlatent varianceとは異なります。

Binary Boundary Varianceはlatent varianceへthreshold-centered Gaussian weightを掛けます。

---

## 10. Multiclass target probability

source：

```text
src/bochan/acquisition/multiclass/levelset_estimation/single_output.py
```

selected class set$T$から

```math
p_T(x)
=
\mathrm{class\_reduce}
\{p_k(x):k\in T\}
```

を作ります。mutually exclusiveなacceptable classのunion probabilityならsum reductionが確率的に自然です。

current class名`qMulticlassLatentStraddleAcquisition`には`Latent`が含まれますが、single-output実装はtarget probability spaceで動作します。

---

## 11. Multiclass uncertainty mode

posterior target-probability sampleを$p_i^{(s)}$とします。

### Bernoulli mode

```math
u_i=\sqrt{p_i(1-p_i)}.
```

### Posterior mode

```math
u_i=\mathrm{Std}_s[p_i^{(s)}].
```

### Combined mode

```math
u_i
=
\sqrt{
\mathrm{Var}_s[p_i^{(s)}]
+p_i(1-p_i)
}.
```

posterior uncertaintyとtarget-set membership randomnessを区別します。

---

## 12. Multiclass Straddle・joint score

pointwise target-probability Straddleは

```math
s_i
=
\beta u_i-|p_i-h_p|.
```

joint classではsample covariance

```math
\widehat\Sigma_p
=
\frac1{S-1}
\sum_s
(\mathbf p^{(s)}-\bar{\mathbf p})
(\mathbf p^{(s)}-\bar{\mathbf p})^\top
+\epsilon I
```

を使い、

```math
s(X)
=
\beta U(\widehat\Sigma_p)
-D(\bar{\mathbf p},h_p)
```

とします。

---

## 13. Multiclass ICU・Boundary Variance・entropy

ICUはGaussian contour weight

```math
w_i
=
\exp\left[
-\frac12
\left(
\frac{p_i-h_p}{b}
\right)^2
\right]
```

を使い、

```math
s_i=u_i^2w_i
```

とします。

Boundary Varianceは

```math
w_i
=
\exp\left(
-\frac{|p_i-h_p|}{b}
\right)
```

を使います。

full class entropyは

```math
s_i
=-\sum_{k=0}^{K-1}p_{ik}\log p_{ik}.
```

selected-class pathでは実装上$-p_T\log p_T$のみを使う場合があり、binary entropyのcomplement termを含まない点に注意します。

Probability of Exceedanceは

```math
s_i
=
\sigma\left(
\frac{p_i-h_p}{\tau}
\right)
```

というsmooth membership scoreです。

---

## 14. Ordinal boundary indexing

source：

```text
src/bochan/acquisition/ordinal/levelset_estimation/single_output.py
```

K classにはK-1 cutpointがあります。boundary index$j$はclass$j$とclass$j+1$の境界です。

```text
K = 3
target_boundary_idx = 0 -> class 0 / 1
target_boundary_idx = 1 -> class 1 / 2
```

boundary-wise score shapeは

```text
batch_shape x q_like x (K - 1)
```

です。

---

## 15. Ordinal Latent Straddle

boundary$j$へのdistanceは

```math
d_{ij}=|\mu_i-c_j|.
```

scoreは

```math
s_{ij}
=
\beta\sigma_i-d_{ij}.
```

`target_boundary_idx`で1つを選ぶか、weight、mean、sum、max、minでboundary reductionします。

joint classでは

```math
s(X)
=
\beta U(\Sigma)
+
\frac1q\sum_i
\mathrm{boundary\_reduce}_j[-|\mu_i-c_j|]
-P(X).
```

$P(X)$はsame-batch、pending、observed penaltyです。

---

## 16. Ordinal ICU・Boundary Variance・entropy

boundary$j$のcumulative upper probabilityを

```math
g_{ij}
=P(Y_i\ge j+1)
=
\sum_{k=j+1}^{K-1}p_{ik}
```

とします。

ordinal ICUは

```math
u_{ij}=4g_{ij}(1-g_{ij})
```

を使います。

Boundary Varianceは

```math
w_{ij}
=
\exp\left[
-\frac12
\left(
\frac{\mu_i-c_j}{\tau}
\right)^2
\right]
```

```math
s_{ij}=v_iw_{ij}
```

です。

class entropyは

```math
s_i
=-\sum_{k=0}^{K-1}p_{ik}\log p_{ik}.
```

です。

---

## 17. Multi-output・異分散wrapper

registered familyには次があります。

```text
qMultiOutputRegressionStraddle
qMultiOutputBinaryLatentStraddleAcquisition
qMultiOutputMulticlassLatentStraddleAcquisition
qMultiOutputOrdinalLatentStraddleAcquisition
qHeteroRegressionStraddle
qHeteroBinaryLatentStraddleAcquisition
qHeteroMulticlassICUAcquisition
qHeteroOrdinalBoundaryVarianceAcquisition
```

multi-output wrapperは通常、output-wise scoreを`... x q x m`で保持し、configured reductionを適用します。score mean／sum／maxはjoint intersection probabilityではありません。

異分散wrapperはbase scoreとpredicted noise／reliabilityをcombineします。noise penalty、noise-learning value、likelihood noise modelは別概念です。

---

## 18. Perturbation covariance reduction

expanded covariance

```text
... x (q*n_w) x (q*n_w)
```

を

```text
... x q x n_w x q x n_w
```

へreshapeし、両perturbation axisをaverageする`block_mean`があります。

`diagonal_mean`はgroup内marginal varianceだけをaverageし、q covarianceをdiagonalにするためcross-candidate covarianceを捨てます。

---

## 19. Penaltyとq reduction

same-batch penaltyはtransformed distanceに基づくsoft repulsionです。pending／observed penaltyもnearest reference distanceを使います。

input perturbationでqが展開されている場合、同じnominal candidateに属するperturbation replica同士はrepulsion対象から除外します。

pointwise classのq reductionにはmean、sum、max、minなどがあります。covariance-aware batch valueが必要ならjoint classを使います。

---

## 20. Formula-to-class対応

| class | space | core score |
|---|---|---|
| `qRegressionStraddle` | regression posterior | $\beta\sigma-|\mu-h|$ |
| `qRegressionJointStraddle` | joint covariance | $-\mathrm{mean}|\mu-h|+\beta U(\Sigma)$ |
| `qRegressionICU` | regression posterior | contour weight $\times\sigma$ |
| `qRegressionBoundaryVariance` | regression posterior | boundary weight $\times v$ |
| `qBinaryLatentStraddleAcquisition` | binary latent | $\beta\sigma-|\mu-h_f|$ |
| `qBinaryICUAcquisition` | binary probability | $4p(1-p)$ |
| `qBinaryClassEntropyAcquisition` | binary probability | Bernoulli entropy |
| `qMulticlassLatentStraddleAcquisition` | target probability | $\beta u-|p_T-h_p|$ |
| `qMulticlassICUAcquisition` | target probability | $u^2\times$ contour weight |
| `qOrdinalLatentStraddleAcquisition` | ordinal latent | $\beta\sigma-|\mu-c_j|$ |
| `qOrdinalICUAcquisition` | cumulative probability | $4g_j(1-g_j)$ |
| `qOrdinalBoundaryVarianceAcquisition` | ordinal latent | variance $\times$ cutpoint weight |

---

## 21. Source map

```text
src/bochan/acquisition/regression/levelset_estimation/
src/bochan/acquisition/binary/levelset_estimation/
src/bochan/acquisition/multiclass/levelset_estimation/
src/bochan/acquisition/ordinal/levelset_estimation/
src/bochan/acquisition/non_gaussian/levelset_estimation/
src/bochan/api/acquisition_registry.py
```

---

## 22. 検証checklist

1. posterior accessorとthreshold space
2. classの`forward()`と数式の一致
3. `q=1`／`q>1`
4. t-batch output shape
5. single／multi-output
6. class／boundary index
7. pending／observed update
8. exact duplicate
9. mixed input transform
10. `InputPerturbation`と`n_w`
11. DeepGP／ensemble extra axis
12. pointwise／joint batch behavior
13. 05章のexternal LSE lossによる評価
14. published criterionかimplementation proxyか

---

## 23. 解釈上の重要事項

- current ICU classはfamily-specific contour-uncertainty proxyであり、必ずしもexact integrated contour-loss reductionではない
- multiclassの`LatentStraddle`というclass名は、current single-output実装ではtarget probability spaceを使う
- Bernoulli varianceとclass entropyはobservation ambiguityを含む
- posterior probability varianceは別のuncertainty
- output reductionはjoint eventを明示的に計算しない限りscore aggregation
- distance penaltyはBayesian conditioningではない
- risk score objectiveは、classがrobust latent targetを先に構成しない限りscore-level aggregation

## Non-Gaussian response and observation level sets

Response-mean Straddle uses $\beta\sigma_\mu-|\bar\mu-t|$ and excludes
observation noise. JointStraddle replaces pointwise uncertainty by a covariance
trace or log determinant. BoundaryVariance and ICUProxy are local contour
scores; ICUProxy is not fantasy-based integrated contour reduction. Response
PoE uses fixed MC samples (smooth MC by default), whereas ObservationPoE
integrates a family CDF and includes heteroscedastic variance through moment
matching where it cannot remain in the original family. For counts,
$P(Y\ge k)=1-P(Y\le k-1)$ and $P(Y\le t)=F(\lfloor t\rfloor)$.
LevelSetUncertainty scores Bernoulli variance, binary entropy, or margin and is
maximal at exceedance probability one half.
