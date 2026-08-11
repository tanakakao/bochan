# 13. 異分散・ロバストモデル

本章では、「ノイズ」という1語でまとめられがちな複数の不確かさを分離します。既知の測定分散、入力依存観測ノイズ、外れ値、label ambiguity、入力摂動、model uncertaintyは異なる生成過程であり、同じvariance加算だけで処理してよいとは限りません。

---

## 1. 不確かさの分解

連続観測

```math
Y=f(X)+\varepsilon
```

に対して、predictive varianceは概念的に

```math
\mathrm{Var}(Y_*\mid\mathcal D)
=
\underbrace{\mathrm{Var}(f_*\mid\mathcal D)}_{\text{epistemic}}
+
\underbrace{\mathbb E[\sigma_n^2(X_*)\mid\mathcal D]}_{\text{aleatoric}}
+
\text{approximation terms}
```

と分けられます。

入力摂動

```math
\widetilde X=X+\Delta
```

はfunction evaluation前のuncertaintyであり、observation noiseとは別です。

---

## 2. 既知の異分散観測分散

各実験のvariance $s_i^2$が既知なら

```math
y_i\mid f_i
\sim
\mathcal N(f_i,s_i^2)
```

```math
K_y
=
K_f+
\mathrm{diag}(s_1^2,\ldots,s_n^2)
```

を使います。

`train_Yvar`にはstandard deviationではなくvarianceを渡します。repeated measurement、instrument calibration、error propagationから得たvarianceに適します。

---

## 3. Learned heteroscedastic Gaussian regression

mean processとlog-noise processを

```math
f\sim\mathcal{GP}(m_f,k_f),
\qquad
g\sim\mathcal{GP}(m_g,k_g)
```

とし、

```math
\sigma^2(x)
=
\exp(g(x))
```

または

```math
\sigma^2(x)
=
\mathrm{softplus}(g(x))+\epsilon
```

でpositive varianceを作ります。

full posterior

```math
p(f,g\mid\mathbf y)
\propto
p(\mathbf y\mid f,g)p(f)p(g)
```

はnon-conjugateです。

実用上のresidual approximationでは、mean GPをfitし、

```math
r_i=y_i-\widehat f(x_i),
\qquad
z_i=\log(r_i^2+\epsilon)
```

をsecond GPへfitします。これはjoint Bayesian inferenceではなくengineering approximationです。

shared helper：

```text
src/bochan/models/components/heteroscedastic.py
```

---

## 4. Predictionとacquisition

noise estimateをfixedとして扱うGaussian modelでは

```math
\mathrm{Var}(Y\mid x,\mathcal D)
\approx
\mathrm{Var}(f(x)\mid\mathcal D)
+
\widehat\sigma^2(x).
```

underlying meanを学ぶならlatent variance、future measurementを予測するならtotal varianceを使います。

high-noise pointを避けるscoreは

```math
\alpha(x)
=
\alpha_0(x)w(\sigma_n^2(x))
```

で、例として

```math
w(v)=\frac1{1+\lambda v}
```

や

```math
w(v)=\exp(-\lambda v)
```

があります。

noise function自体を学ぶ場合は

```math
\alpha(x)=\alpha_f(x)+\lambda\alpha_g(x)
```

のように両modelのinformationを評価します。

---

## 5. Replicate

noisyな既観測点ではreplicateによってmean uncertaintyを減らせます。heteroscedastic experimentでduplicateを常に禁止するのは適切ではありません。

- 意図したreplicate
- optimizerが偶然生成したduplicate

を区別します。

---

## 6. Classificationでの追加noise

Bernoulli likelihoodはもともと

```math
Y\mid f\sim\mathrm{Bernoulli}(\pi(f))
```

であり、観測分散は

```math
\mathrm{Var}(Y\mid f)
=
\pi(f)[1-\pi(f)]
```

です。

追加noise modelには明確な生成的意味が必要です。

### Label reliability model

入力依存reliability$\rho(x)$を使う一例は

```math
P(Y_{\mathrm{obs}}=1\mid f,x)
=
[1-\rho(x)]\pi(f)
+
\rho(x)[1-\pi(f)].
```

### Input-dependent temperature

```math
P(Y=1\mid f,x)
=
\sigma\left(
\frac{f(x)}{s(x)}
\right).
```

現行wrapperがauxiliary varianceを`SimpleBernoulliPosterior.variance`へ加えるだけなら、noise-aware interfaceではありますが、likelihood自体が上記modelでない限りfully specified generative modelではありません。

---

## 7. Ordinalでの異分散

input-dependent scale ordered-logitは

```math
P(Y\le j\mid x)
=
\sigma\left(
\frac{c_j-f(x)}{s(x)}
\right),
\qquad s(x)>0.
```

です。large $s(x)$はclass transitionをdiffuseにします。

noise modelをacquisition reweightingにだけ使う場合は、fully heteroscedastic ordered-logit inferenceではなくnoise-aware ordinal acquisitionと解釈します。

関連source：

```text
src/bochan/models/ordinal/robust/heteroscedastic.py
src/bochan/acquisition/ordinal/active_learning/hetero_*.py
src/bochan/acquisition/ordinal/levelset_estimation/hetero_*.py
```

---

## 8. Robust Relevance Pursuit

```math
y_i=f(x_i)+o_i+\varepsilon_i
```

のようにsparse correction$o_i$を導入します。

```math
\|\mathbf o\|_0\ll n
```

またはcontinuous relaxationで、少数観測へ追加flexibilityを与えます。

| 手法 | sparseにする対象 |
|---|---|
| RRP | observation／likelihood term／local correction |
| ARD | kernel sensitivity |
| SAAS | inverse length scaleによるdimension |
| k-sparse repair | candidateのnonzero component |

主なsource：

```text
src/bochan/models/regression/gaussian/robust/relevance_pursuit.py
src/bochan/models/classification/binary/robust/
src/bochan/models/classification/multiclass/robust/
src/bochan/models/ordinal/robust/
src/bochan/fit/robust/
```

---

## 9. Heavy-tailed likelihoodとの違い

Student-t likelihood

```math
y_i\mid f_i
\sim
\mathrm{StudentT}(\nu,f_i,\sigma)
```

はresidual全体をcontinuousにdownweightします。sparse outlier modelは少数のexceptional pointを分離します。

- occasional sensor anomaly：sparse correction
- consistent heavy tail：Student-t
- input-dependent variance：heteroscedastic Gaussian
- uncertain class observation：classification reliability model

---

## 10. Shapeとtransform

shared helperは次を行います。

1. `train_Yvar`を`N x 1`へ整形
2. positive floorでclamp
3. auxiliary noise modelにはnormalization-only transformを使う
4. mixed modelでcategory dimensionをnormalizeしない
5. predicted noise tensorをposterior shapeへalign
6. `q -> q*n_w`時のraw input alignmentを管理

---

## 11. 評価

- mean prediction：RMSE、MAE
- probabilistic prediction：NLPD、coverage
- noise model：replicate varianceとの比較
- robust decision：repeat mean、variance、quantile、CVaR、constraint violation

```math
\mathrm{NLPD}
=
-\frac1n
\sum_i
\log p(y_i\mid x_i,\mathcal D).
```

---

## 12. Source map

| component | source |
|---|---|
| shared helper | `src/bochan/models/components/heteroscedastic.py` |
| Gaussian heteroscedastic | `src/bochan/models/regression/gaussian/robust/heteroscedastic.py` |
| non-Gaussian robust | `src/bochan/models/regression/{beta,gamma}/robust/` および `src/bochan/models/regression/count/*/robust/` |
| binary robust | `src/bochan/models/classification/binary/robust/heteroscedastic.py` |
| multiclass robust | `src/bochan/models/classification/multiclass/robust/heteroscedastic.py` |
| ordinal robust | `src/bochan/models/ordinal/robust/heteroscedastic.py` |
| RRP fitting | `src/bochan/fit/robust/` |

---

## 13. 確認項目

1. varianceはknownかlearnedか
2. noiseはinput dependentか
3. replicateはあるか
4. noise function自体を学ぶか
5. sparse correctionかheavy-tailか
6. classification／ordinal noiseの意味は何か
7. latent varianceとpredictive varianceのどちらを使うか
8. input perturbationとobservation noiseを分離したか
9. calibrationを何で評価するか
