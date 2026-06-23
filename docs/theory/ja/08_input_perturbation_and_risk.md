# 08. 入力摂動とリスク指向objective

標準的なベイズ最適化はnominal inputを評価しますが、実際の製造条件や実験条件は設定値からずれる場合があります。本章では、入力摂動下の平均性能、variance penalty、quantile、VaR、CVaR、chance constraint、および`bochan`でのTensor実装を整理します。

---

## 1. Nominal inputとexecuted input

選択した入力を$x$、実際に実行される入力を

```math
\widetilde X=T(x,W)
```

とします。additive perturbationなら

```math
\widetilde X=x+W.
```

responseを$Z_x=f(\widetilde X)$とすると、robust optimizationは

```math
x^*\in\arg\max_x\rho(Z_x)
```

を解きます。$\rho$はmean、quantile、CVaRなどのrisk functionalです。

---

## 2. 区別すべきrandomness

- posterior epistemic uncertainty：$f\sim p(f\mid\mathcal D)$
- observation noise：$Y=f(X)+\varepsilon$
- input perturbation：$\widetilde X=T(x,W)$
- environmental variation：$Y=f(x,E)+\varepsilon$
- class outcome randomness：$Y\sim\mathrm{Categorical}(\mathbf p(x))$

risk measureがどのrandomnessについて定義されるかを明示します。

---

## 3. Mean・variance・quantile

平均性能は

```math
R_{\mathrm{mean}}(x)
=
\mathbb E_W[f(T(x,W))]
```

で、MC estimatorは

```math
\widehat R_{\mathrm{mean}}(x)
=
\frac1{n_w}
\sum_{r=1}^{n_w}f(T(x,W_r)).
```

mean-variance objectiveは

```math
R_{\mathrm{MV}}(x)
=
\mathbb E[Z_x]-\lambda\mathrm{Var}(Z_x).
```

posterior uncertaintyとinput perturbationが同時にある場合、

```math
\mathrm{Var}(Z_x\mid\mathcal D)
=
\mathbb E_W[\mathrm{Var}_f(f(T(x,W))\mid W,\mathcal D)]
+
\mathrm{Var}_W[\mathbb E_f(f(T(x,W))\mid W,\mathcal D)].
```

第1項はmodel uncertainty、第2項はexecution sensitivityです。

lower-tail quantileは

```math
q_\alpha(x)=F_x^{-1}(\alpha)
```

です。utilityをmaximizeする場合、small $\alpha$は低い側のperformanceを表します。

---

## 4. VaRとCVaR

utilityに対するlower-tail VaRを

```math
\mathrm{VaR}^{\mathrm{lower}}_\alpha(Z_x)=q_\alpha(x)
```

とします。

CVaRはtail内の平均です。

```math
\mathrm{CVaR}^{\mathrm{lower}}_\alpha(Z_x)
=
\mathbb E[Z_x\mid Z_x\le q_\alpha(x)].
```

sampleではutilityをascending sortし、worst $\lceil\alpha n_w\rceil$個のmeanを使います。

lossをminimizeする場合はupper-tail conventionになるため、次を明示します。

- larger is betterか
- lower／upper tail
- `alpha`がtail massかconfidence levelか
- sorting direction
- `maximize` flag

---

## 5. Target reliabilityとchance constraint

threshold$h$を満たす確率を

```math
R_h(x)=P_W(f(T(x,W))\ge h)
```

とします。chance constraintは

```math
R_h(x)\ge1-\epsilon
```

です。

posterior uncertaintyまで含む場合は$P_{f,W}$、future measurement noiseまで含む場合は$P_{f,W,\varepsilon}$となり、異なるfeasible setを定義します。

---

## 6. Perturbation distribution

代表例は次です。

- additive Gaussian：$W\sim\mathcal N(0,\Sigma_W)$
- uniform tolerance
- multiplicative error：$\widetilde X_j=x_j(1+W_j)$
- correlated execution error
- empirical／bootstrap distribution
- categorical transition matrix

category codeへ連続noiseを加えるのではなく、category transition probabilityを定義します。

composition inputではDirichlet、logistic-normal、mass-transferなどsum constraintを保つperturbationを使います。

---

## 7. Domain boundary handling

perturbed inputがdomain外へ出る場合は、clipping、reflection、rejection／resampling、feasible manifold上での直接sampleなどを使います。

これらは同じdistributionではありません。たとえばclippingはboundaryにprobability massを集めます。policy自体をproblem definitionに含めます。

---

## 8. Common random numbers

acquisition optimization中は同じperturbation sampleを固定すると、sample-average approximationがsmoothになり、gradientとrestart比較が安定します。

iteration間でresampleする場合もseedを記録して再現可能にします。

---

## 9. Tensor expansion

```text
X:       batch_shape x q x d
X_tilde: batch_shape x (q * n_w) x d
```

pointwise scoreは

```text
batch_shape x (q * n_w)
```

から

```text
batch_shape x q x n_w
```

へreshapeし、perturbation axisをreduceします。

posterior sampleは

```text
sample_shape x batch_shape x q x n_w x m
```

となる場合があります。

---

## 10. Reduction order

nonlinear operationは交換できません。

```math
\mathrm{EI}(\mathrm{CVaR}_W[f])
\ne
\mathrm{CVaR}_W(\mathrm{EI}[f]).
```

robust latent utilityをoptimizationする場合は、output transform、perturbation risk reduction、improvement、posterior sample averageの順で処理します。

一方、`bochan`の一部の能動学習・LSE objectiveは、計算済みpointwise scoreをmean／VaR／CVaRでaggregateします。これはscore-level risk aggregationです。

---

## 11. Classification・ordinal・multi-output

binary probability下では、mean success probability、lower quantile、probability requirementを満たす確率を区別します。

ordinal expected utilityは

```math
\mathbb E_W\left[\sum_ku_kp_k(x,W)\right]
```

です。linear expectationはsumと交換できますが、CVaRは一般に交換できません。

multi-outputでは

```math
\rho_W[s(\mathbf f)]
\ne
s(\rho_W[f_1],\ldots,\rho_W[f_m])
```

であり、robust scalarizationとcomponent-wise riskは異なります。

---

## 12. Sample sizeと評価

mean estimatorのstandard errorは概ね$O(n_w^{-1/2})$です。CVaRで使われるtail sample数は約$\alpha n_w$なので、小さい$\alpha$では大きな$n_w$が必要です。

評価項目：

- repeated-execution mean／variance
- lower quantile、CVaR
- target failure probability
- constraint violation rate
- robust regret
- calibration under perturbed input
- runtime versus $n_w$

---

## 13. `bochan`実装との対応

`InputPerturbation`はmodelのinput transformとして利用され、evaluation時に`q -> q*n_w`へ展開します。training dataはtargetと不整合にならないよう展開を抑制します。

score objectiveの典型的な流れは

```text
pointwise score
    -> reshape (..., q, n_w)
    -> mean / VaR / CVaR
    -> q reduction
```

です。

異分散補助modelでは

```text
src/bochan/models/components/heteroscedastic.py
```

がnormalization-only transformを抽出し、noise-model trainingへ`InputPerturbation`を適用しないようにします。

主な設定は`n_w`、`risk_type`、`alpha`、`maximize`、`weight`、`sign`です。

---

## 14. 設定checklist

1. nominal inputとexecuted inputの関係
2. perturbation distributionとcorrelation
3. boundary handling
4. categorical／composition treatment
5. riskに含めるrandomness
6. utility／loss convention
7. risk functionalとtail direction
8. `alpha`の意味
9. `n_w`とsampling method
10. common random number policy
11. reduction order
12. constraint interpretation
13. repeated-execution evaluation
