# 03. ベイズ最適化の獲得関数

獲得関数はposterior distributionを「次にどこを観測する価値が高いか」というscalarへ変換します。本章ではBO用criterionを扱います。能動学習は04章、LSEは05・16章、多目的獲得関数は07章を参照してください。

---

## 1. 一般形

candidate batch`X=[x_1,\ldots,x_q]`に対して

```math
\alpha_t(X)
=
\mathbb E
\left[
V(X,F;\mathcal D_t)
\mid\mathcal D_t
\right]
```

と書きます。optimizerは

```math
X_{t+1}
\in
\arg\max_{X\in\mathcal X^q}
\alpha_t(X)
```

を解きます。

BoTorchでは通常

```text
X:         batch_shape x q x d
acq_value: batch_shape
```

です。sample、output、class、q軸をcriterionの定義に従って削減し、t-batch axisだけを残します。

---

## 2. AnalyticとMonte Carlo

analytic acquisitionはclosed formで高速ですが、single-output、Gaussian posterior、`q=1`などに制約されます。

MC acquisitionはreparameterized sample

```math
f^{(s)}(X)\sim p(f(X)\mid\mathcal D_t)
```

へobjectiveを適用し、

```math
\alpha(X)
\approx
\frac1S\sum_{s=1}^{S}V(u^{(s)}(X))
```

で期待値を近似します。q-batch、非線形objective、constraint、多出力に対応しやすい一方、sample数とbase sample管理が重要です。

---

## 3. Probability of Improvement

thresholdを

```math
\tau=f_{\mathrm{best}}+\xi
```

とすると

```math
\mathrm{PI}(x)
=
P(f(x)\ge\tau\mid\mathcal D).
```

Gaussian posteriorでは

```math
\mathrm{PI}(x)
=
\Phi\left(
\frac{\mu(x)-\tau}{\sigma(x)}
\right).
```

PIは改善量を考慮しないため、小さい改善が高確率で得られる点を選びやすい特徴があります。

---

## 4. Expected Improvement

```math
I(x)=\max(f(x)-f_{\mathrm{best}},0)
```

として

```math
\mathrm{EI}(x)=\mathbb E[I(x)\mid\mathcal D].
```

Gaussian posteriorで

```math
z=\frac{\mu-f_{\mathrm{best}}}{\sigma}
```

なら

```math
\mathrm{EI}(x)
=
(\mu-f_{\mathrm{best}})\Phi(z)
+
\sigma\phi(z).
```

q-batchでは

```math
I(X)
=
\max\left(
\max_i f(x_i)-f_{\mathrm{best}},0
\right)
```

をMCで評価します。

### LogEI

改善確率が非常に小さいとEIはunderflowし、gradientが弱くなります。LogEIはlog-domainで安定に評価するため、現在のBoTorchでは推奨される場面が多いcriterionです。

---

## 5. Noisy Expected Improvement

noisy observationではbaseline latent valueも不確かです。

```math
\mathrm{qNEI}(X)
=
\mathbb E\left[
\max\left(
\max f(X)-\max f(X_{\mathrm{baseline}}),0
\right)
\right].
```

重要なargumentは`X_baseline`、objective、constraint、sampler、`X_pending`、baseline pruningです。

### Log Noisy Expected Improvement

`qLogNoisyExpectedImprovement`はnoisy improvementをlog-domainで数値安定に評価します。`best_f`ではなく`X_baseline`を基準として使い、objective、constraint、sampler、`X_pending`などはqNEI系と同じ考え方で扱います。

bochanではstandard Gaussian regressionに対して`lognei` / `qlognei`で利用できます。通常のqNEIと探索戦略が別物なのではなく、同じnoisy-improvement criterionを安定に最適化するための実装として扱います。

---

## 6. UCB

maximizationでは

```math
\mathrm{UCB}(x)
=
\mu(x)+\sqrt\beta\,\sigma(x).
```

parameterizationによっては`beta`を直接standard deviationへ掛けます。classの定義を確認してください。

large betaはexploration、small betaはexploitationを強めます。output standardizationにより適切なscaleが変わります。

---

## 7. Thompson sampling

posterior function sample

```math
\tilde f\sim p(f\mid\mathcal D)
```

をdrawし、

```math
x_{t+1}\in\arg\max_x\tilde f(x)
```

を選びます。

marginalを独立にsampleするとfunction correlationを失うため、continuous domainではpathwise sampleやrandom feature approximationが必要です。

---

## 8. Knowledge Gradient

```math
\mathrm{KG}(x)
=
\mathbb E_{y_x}
\left[
\max_{x'}\mu_{t+1}(x';y_x)
\right]
-
\max_{x'}\mu_t(x').
```

EIがsampled pointでのdirect improvementを評価するのに対し、KGは観測によってfuture best decisionがどれだけ改善するかを評価します。

---

## 9. Multi-step look-ahead

multi-step policyはfuture observationとfuture optimizationをnestedに扱います。

```math
\alpha_t(x_1)
=
\mathbb E_{y_1}
\left[
\max_{x_2}
\mathbb E_{y_2\mid y_1}[\cdots]
\right].
```

`qMultiStepLookahead`はfantasy sample、stage batch size、stage value functionで表現します。計算量、conditioning support、mixed constraintが主な制約です。

---

## 10. Information-theoretic BO

- Predictive Entropy Search：optimizer`x^*`とのmutual information
- Max-value Entropy Search：optimum value`f^*`とのmutual information
- Joint Entropy Search：`(x^*,f^*)`とのmutual information

sample efficientになり得ますが、optimum distributionの近似が必要です。

---

## 11. Objectiveとposterior transform

posterior sample`Y^{(s)}\in\mathbb R^m`へ

```math
T:\mathbb R^m\rightarrow\mathbb R
```

または

```math
T:\mathbb R^m\rightarrow\mathbb R^{m_{\mathrm{obj}}}
```

を適用します。

- output selection
- minimizationからmaximizationへのsign変換
- weighted scalarization
- class probabilityから期待効用
- input perturbation aggregation
- feasibility constraint

posterior transformはdistributionそのものを変換し、MC objectiveはsampleを変換します。非線形`T`では

```math
T(\mathbb E[Y])\ne\mathbb E[T(Y)].
```

bochanのtask-specific NParEGOとBoTorchの`qLogNParEGO`では、augmented Chebyshev scalarizationとbaseline比較を獲得関数内部で行います。この場合の`objective`はscalarizationではなく、raw posterior sampleを`[..., q, m]`の目的空間へ写す**multi-output preprocessing objective**です。したがって外側から二重scalarizationしてはいけません。またこれらは`X_baseline`を使い、`best_f`を必要としません。

---

## 12. Sample-level constraint

sample constraint`c_j(Y)\le0`に対して

```math
V(Y)
=
I(Y)
\prod_j\mathbf 1[c_j(Y)\le0]
```

とします。hard indicatorはgradientを持たないため、sigmoid approximationを使う場合があります。

```math
\alpha_c(x)=\alpha_0(x)P(\mathrm{feasible}\mid x)
```

というfactorizationは簡単ですが、joint objective-constraint dependenceを正確に扱うとは限りません。

`qLogProbabilityOfFeasibility`はposterior sampleに対するMonte Carlo feasibility probabilityをlog-domainで評価します。bochanではstandard regressionのsample constraintに対して`logpof` / `qlogpof`を公開します。これはbinary / multiclass / ordinalでclass probabilityやutilityを意味するtask-specific PoFとは区別します。

---

## 13. Batch dependenceとpending

q-batch posterior covariance

```math
\mathbf f_X\sim\mathcal N(\boldsymbol\mu_X,\Sigma_X)
```

はexpected maximum、at-least-one improvement、constraint、information gainを決めます。marginal mean／varianceだけではredundant candidateを過大評価します。

`X_pending`はfantasy、joint sample、sequential selection、distance penaltyで扱います。distance penaltyはBayesian conditioningではありません。

---

## 14. Acquisition optimization

標準的なmultistart optimizationは次です。

1. raw sample生成
2. initialization score評価
3. `num_restarts`を選択
4. gradient optimization
5. best result選択

mixed domainではcategory enumeration、`optimize_acqf_mixed`、evolutionary optimizerを使います。

repair後はacquisition value、constraint、category validity、duplicateを再評価します。

---

## 15. 数値上の注意

- local optimization中はbase sampleを固定し、sample-average approximationを安定化する
- tiny improvementやfeasibility probabilityではLog系criterionを検討する
- `best_f`、beta、temperature、reference pointのscaleを確認する
- variance clampが頻発するならcovariance実装を疑う
- replicateが不要ならduplicateを抑制する

bochanではstandard regression向けにLogEI、LogNEI、LogPoF、LogEHVI、LogNEHVI、LogNParEGOへのregistry経路を揃えます。Log版は別のexploration heuristicではなく、対応するcriterionを数値安定に扱う実装として選択します。

---

## 16. 選択の目安

| 状況 | 初期候補 |
|---|---|
| low-noise single-objective | LogEI / qLogEI |
| noisy experiment | qLogNEI（非log baselineとしてqNEI） |
| MC feasibility constraint | qLogPoF |
| simple exploration baseline | qUCB |
| probability target | task-specific EI / PI / UCB |
| future decision価値 | qKG |
| explicit multi-step | qMultiStepLookahead |
| multi-objective low-noise | qLogEHVI / qEHVI |
| multi-objective noisy | qLogNEHVI / qNEHVI |
| scalarized multi-objective | qLogNParEGO / task-specific NParEGO |

model、objective、posterior contractを確認してから適用します。

---

## 17. `bochan`実装との対応

`src/bochan/api/acquisition_registry.py`は代表的に次を解決します。

| alias | class |
|---|---|
| `qei`, `ei` | `qExpectedImprovement` |
| `qlogei`, `logei` | `qLogExpectedImprovement` |
| `qnei`, `nei` | `qNoisyExpectedImprovement` |
| `qlognei`, `lognei` | `qLogNoisyExpectedImprovement` |
| `qlogpof`, `logpof` | `qLogProbabilityOfFeasibility` |
| `qucb`, `ucb` | `qUpperConfidenceBound` |
| `qpi`, `pi` | `qProbabilityOfImprovement` |
| `qkg`, `kg` | `qKnowledgeGradient` |
| `lookahead` | `qMultiStepLookahead` |
| `qehvi`, `ehvi` | `qExpectedHypervolumeImprovement` |
| `qlogehvi`, `logehvi` | `qLogExpectedHypervolumeImprovement` |
| `qnehvi`, `nehvi` | `qNoisyExpectedHypervolumeImprovement` |
| `qlognehvi`, `lognehvi` | `qLogNoisyExpectedHypervolumeImprovement` |
| `qlognparego`, `lognparego` | `qLogNParEGO` |

multi-output regressionで`nparego` / `qnparego`を指定した場合は、qEIへのfallbackではなくbochanの`qMultiOutputRegressionNParEGO`へcontextual routingします。このclassはaugmented Chebyshev scalarizationを内部で実行します。

custom acquisitionはtask別に配置されます。

```text
src/bochan/acquisition/regression/bayesian_optimization/
src/bochan/acquisition/binary/bayesian_optimization/
src/bochan/acquisition/multiclass/bayesian_optimization/
src/bochan/acquisition/ordinal/bayesian_optimization/
src/bochan/acquisition/non_gaussian/bayesian_optimization/
```

今回のLog short aliasはstandard regression / hybrid posteriorの意味論に限定します。classification / ordinalへ暗黙に転送するとclass probabilityやutility spaceの定義が変わるため、task-specific acquisitionとは明示的に分けます。

optimizer backendとrepairは`src/bochan/optim/`にあります。

---

## 18. 新規acquisitionの確認項目

1. optimization targetとdirection
2. 消費するposterior space
3. analytic／MC estimator
4. `best_f`またはbaseline定義
5. objective／constraint
6. q-batch semantics
7. pending handling
8. input perturbation
9. output／class reduction
10. return shape
11. differentiability
12. optimizer compatibility
13. published criterionかcustom proxyか
