# 12. 順序モデル

順序モデルは、段階評価のように順序を持つクラスを扱います。クラス番号を通常の回帰値として扱うのではなく、1本の潜在関数と複数のcutpointによってclass probabilityを定義します。

BOへの変換は06章、能動学習は04章、LSEの実装式は16章を参照してください。

---

## 1. Ordered-logit model

```math
Y\in\{0,1,\ldots,K-1\},
\qquad
f\sim\mathcal{GP}(m,k)
```

cutpointを

```math
c_0<c_1<\cdots<c_{K-2}
```

とします。ordered-logit likelihoodは

```math
P(Y\le k\mid f)=\sigma(c_k-f)
```

であり、class probabilityは

```math
P(Y=k\mid f)
=
\sigma(c_k-f)-\sigma(c_{k-1}-f).
```

$f$が増加するとhigher classへprobability massが移ります。

---

## 2. Cutpointの順序制約

unconstrained parameterからpositive gapを作ります。

```math
\Delta_j=\mathrm{softplus}(r_j)+\epsilon.
```

first cutpointを固定する場合は

```math
c_0=0,
\qquad
c_j=\sum_{l=1}^{j}\Delta_l.
```

全cutpointをfreeにする場合はcumulative sumをcenterします。いずれもstrict orderを保ちます。

---

## 3. Identifiability

```math
f(x)\mapsto f(x)+a,
\qquad
c_j\mapsto c_j+a
```

としてもclass probabilityは変化しません。そのため、first cutpoint固定、cutpoint centering、latent mean constraintなどのlocation conventionが必要です。

input-dependent scaleを追加する場合は、latent function、cutpoint、scaleの識別条件も必要です。

---

## 4. Variational inference

inducing variableを

```math
\mathbf u=f(Z),
\qquad
q(\mathbf u)=\mathcal N(\mathbf m_u,S_u)
```

とします。ELBOは

```math
\mathcal L
=
\sum_i
\mathbb E_{q(f_i)}
[\log P(y_i\mid f_i,\mathbf c)]
-
\mathrm{KL}[q(\mathbf u)\|p(\mathbf u)].
```

cutpointはlikelihood parameterなので、GP parameterとともにoptimizerへ含めます。1次元latent expectationにはGauss-Hermite quadratureを利用できます。

---

## 5. Marginal class probability

predictionではlatent posteriorを積分します。

```math
p_k(x)
=
\int
P(Y=k\mid f)
q(f\mid x,\mathcal D)\,df.
```

一般に

```math
p_k(x)
\ne
P(Y=k\mid f=\mu_f(x)).
```

です。`OrdinalLogitLikelihood.marginal_class_probs`はquadratureで積分を近似します。

minimum-grade probabilityは

```math
P(Y\ge g\mid x)
=
\sum_{k=g}^{K-1}p_k(x)
```

であり、constraintやLSE targetに利用できます。

---

## 6. Utilityとboundary

class utility$u_k$に対する期待効用は

```math
U(x)=\sum_{k=0}^{K-1}u_kp_k(x).
```

`u_k=k`はclass間隔が等しいという追加仮定です。domain-specific utilityを使えば、特定grade到達の価値を明示できます。

cutpoint$c_j$はlower groupとupper groupを分け、

```math
P(Y\ge j+1\mid f)=\sigma(f-c_j)
```

です。boundary ambiguityには

```math
A_j(x)=4g_j(x)[1-g_j(x)]
```

を使えます。

realized class utility varianceとexpected-utility functionのepistemic varianceは別の量です。

---

## 7. `bochan`のposterior contract

```python
latent = model.posterior(X)
probs = model.class_probs(X)
predicted_class = model.predict_class(X)
utility = model.expected_utility(X, utilities)
```

- `posterior(X)`：scalar latent GP
- `class_probs(X)`：`batch_shape x q x K`
- `expected_utility(X, utilities)`：`batch_shape x q`

binary／multiclassの`posterior()`がprobability-spaceである点と異なります。

---

## 8. Input、mixed model、inducing point

ordinal wrapperは`train_inputs_raw`とtransformed `train_inputs`を区別します。continuous dimensionには通常のkernel、categorical dimensionには`CategoricalKernel`を使い、category columnをnormalizeしません。

training inputへevaluation-mode `InputPerturbation`を再適用するとq軸が増えるため、transform適用段階を明確にします。

inducing pointについてもraw spaceとtransformed spaceを分離します。

---

## 9. Class countとconditioning

labelは原則として0始まりの連続integerにします。initial dataに未観測classがある場合は`num_classes`を明示します。

variational ordinal modelはexact Gaussian rank updateを行えません。`condition_on_observations()`はmodelを再構築し、stateをcopyし、必要に応じて追加optimizationします。look-ahead fantasyはapproximationとして検証します。

---

## 10. Calibrationと評価

- negative log likelihood
- class-wise calibration
- cumulative boundary calibration
- ranked probability score
- mean absolute class error

RPSは

```math
\mathrm{RPS}
=
\sum_{k=0}^{K-2}
[F_k-\mathbf1(y\le k)]^2
```

です。

cutpoint order、gap size、class frequency、missing class、inducing point coverage、initializationへの感度も確認します。

---

## 11. 異分散・deep・multi-output

入力依存scaleをlikelihoodへ入れるmodelと、auxiliary noiseをposterior varianceやacquisition scoreへ組み合わせるwrapperは別です。

DeepGP、Deep Kernel、SAASはlatent representationを変えますが、ordered likelihoodとcutpoint semanticsは維持されます。

相関multi-output ordinalでは

```math
\mathrm{Cov}[f_r(x),f_s(x')]
=B_{rs}k(x,x')
```

を使えますが、$B$はraw label correlationではありません。

---

## 12. 実装対応

| component | source |
|---|---|
| ordered-logit likelihood | `src/bochan/models/ordinal/likelihood.py` |
| base model | `src/bochan/models/ordinal/base/` |
| fitting | `src/bochan/fit/ordinal.py` |
| posterior transform | `src/bochan/models/transforms/posterior/ordinal.py` |
| BO | `src/bochan/acquisition/ordinal/bayesian_optimization/` |
| Active Learning | `src/bochan/acquisition/ordinal/active_learning/` |
| LSE | `src/bochan/acquisition/ordinal/levelset_estimation/` |
| robust | `src/bochan/models/ordinal/robust/` |
| deep／high-dimensional | `src/bochan/models/ordinal/deep/`, `src/bochan/models/ordinal/high_dim/` |

---

## 13. 確認項目

1. labelは本当にorderedか
2. class数とencodingは正しいか
3. linkとcutpoint conventionは何か
4. kernel／inducing pointは何か
5. `posterior()`と`class_probs()`を使い分けているか
6. utility spacingは妥当か
7. calibrationとmissing classを評価したか
8. conditioning／fantasyの近似を理解しているか
9. deep／heteroscedastic extensionの意味を分けたか
