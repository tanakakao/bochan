# 12. 順序モデル

順序モデルは、段階評価のように順序を持つクラスを扱います。クラス番号を通常の回帰値として扱うのではなく、1本の潜在関数と複数のcutpointによってクラス確率を定義します。

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

全cutpointをfreeにする場合はcumulative sumをcenterします。

---

## 3. Identifiability

```math
f(x)\mapsto f(x)+a,
\qquad
c_j\mapsto c_j+a
```

としてもclass probabilityは変化しません。そのため、first cutpoint固定、cutpoint centering、latent mean constraintなどのlocation conventionが必要です。

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

cutpointはlikelihood parameterなので、GP parameterとともにoptimizerへ含めます。

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

一般に、latent meanをlikelihoodへ代入した値とは一致しません。`OrdinalLogitLikelihood.marginal_class_probs`はquadratureで積分を近似します。

---

## 6. Utilityとboundary

class utility$u_k$に対する期待効用は

```math
U(x)=\sum_{k=0}^{K-1}u_kp_k(x).
```

cutpoint$c_j$はlower groupとupper groupを分け、

```math
P(Y\ge j+1\mid f)=\sigma(f-c_j)
```

です。boundary ambiguityには

```math
A_j(x)=4g_j(x)[1-g_j(x)]
```

を使えます。

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

## 8. Inputとmixed model

ordinal wrapperは`train_inputs_raw`とtransformed `train_inputs`を区別します。continuous dimensionには通常のkernel、categorical dimensionには`CategoricalKernel`を使い、category columnをnormalizeしません。

labelは原則として0始まりの連続integerにします。initial dataに未観測classがある場合は`num_classes`を明示します。

---

## 9. Calibrationと評価

- negative log likelihood
- class-wise calibration
- cumulative boundary calibration
- ranked probability score
- mean absolute class error

cutpoint order、gap size、class frequency、missing class、inducing point coverage、initializationへの感度も確認します。

---

## 10. 実装対応

| component | source |
|---|---|
| ordered-logit likelihood | `src/bochan/likelihoods/ordinal.py` |
| base model | `src/bochan/models/ordinal/base/` |
| fitting | `src/bochan/fit/ordinal.py` |
| posterior transform | `src/bochan/models/transforms/posterior/ordinal.py` |
| BO | `src/bochan/acquisition/ordinal/bayesian_optimization/` |
| Active Learning | `src/bochan/acquisition/ordinal/active_learning/` |
| LSE | `src/bochan/acquisition/ordinal/levelset_estimation/` |
| robust | `src/bochan/models/ordinal/robust/` |
| deep／high-dimensional | `src/bochan/models/ordinal/deep/`, `src/bochan/models/ordinal/high_dim/` |
