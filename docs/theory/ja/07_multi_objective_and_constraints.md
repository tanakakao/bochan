# 07. 多目的最適化と制約

複数のmodel outputが存在しても、すべてがobjectiveとは限りません。constraint、diagnostic、cost、auxiliary predictionを区別する必要があります。本章ではPareto dominance、hypervolume、scalarization、probabilistic constraintを整理します。

異種出力posteriorは15章で扱います。

---

## 1. Multi-output、multi-objective、multitask

multi-output modelは

```math
\mathbf y(x)=[y_1(x),\ldots,y_m(x)]
```

を予測します。

multi-objective problemは複数の変換済みobjective

```math
\mathbf f(x)=[f_1(x),\ldots,f_M(x)]
```

をoptimizationします。`m`と`M`は一致しなくても構いません。

multitask modelはoutput間のstatistical structureを明示的に共有します。independent ModelListはmulti-outputですがcross-output covarianceを持ちません。

---

## 2. Direction normalization

maximization conventionへ揃えるため

```math
f_j(x)=s_jg_j(x),
\qquad
s_j\in\{+1,-1\}
```

とします。target matchingなら

```math
f_j(x)=-|g_j(x)-a_j|
```

などを使います。

Pareto、reference point、constraint、baselineはすべて変換後spaceで定義します。

---

## 3. Pareto dominance

maximizationで`\mathbf a`が`\mathbf b`をweakly dominateするとは

```math
a_j\ge b_j
\quad\forall j
```

です。少なくとも1次元でstrictならstrict dominanceです。

Pareto setは

```math
\mathcal P_X
=
\{x\in\mathcal X:
\nexists x'\text{ such that }\mathbf f(x')\succ\mathbf f(x)
\}.
```

Pareto frontierはそのobjective-space imageです。追加preferenceがなければunique optimumではなくtrade-off setを返します。

---

## 4. Hypervolume

reference point`\mathbf r`をobjective regionより悪い点として、nondominated set`P`のdominated hypervolumeを

```math
\mathrm{HV}(P;\mathbf r)
=
\lambda_M
\left(
\bigcup_{\mathbf y\in P}
[\mathbf r,\mathbf y]
\right)
```

とします。

candidate outcome`\mathbf y`のhypervolume improvementは

```math
\mathrm{HVI}(\mathbf y)
=
\mathrm{HV}(P\cup\{\mathbf y\};\mathbf r)
-
\mathrm{HV}(P;\mathbf r).
```

EHVIはそのposterior expectation、qEHVIはq候補のunionをjointに評価します。

---

## 5. Noisy hypervolume improvement

noisy baselineではlatent Pareto frontier自体が不確かです。

```math
\mathrm{qNEHVI}(X)
=
\mathbb E
\left[
\mathrm{HV}(P(\mathbf f_B)\cup\mathbf f_X;\mathbf r)
-
\mathrm{HV}(P(\mathbf f_B);\mathbf r)
\right].
```

qNEHVIはqEHVIのvarianceを増やしたものではなく、baseline frontierのposterior uncertaintyを積分します。

---

## 6. Reference point

reference pointは、hypervolumeとして価値を持たせたいoutcomeより悪く設定します。

- sign／weight変換後spaceに置く
- probability objectiveは通常`[0,1]`
- ordinal utilityはutility range
- standardized regressionかoriginal unitかを確認

optimisticすぎるreferenceは有効solutionのHVを0にし、pessimisticすぎるreferenceはscaleに支配されます。

benchmarkではfixed reference pointを使う方が比較しやすくなります。

---

## 7. Objective scaling

objective scaleが大きく異なるとhypervolumeやscalarizationが一方へ偏ります。

```math
\tilde f_j
=
\frac{f_j-a_j}{b_j-a_j}
```

などでnormalizationします。domain knowledge、historical bound、pilot dataなどから固定scaleを決めると再現性が高くなります。

---

## 8. Scalarization

### Weighted sum

```math
s_w(\mathbf f)=\sum_jw_jf_j,
\qquad
w_j\ge0,
\quad
\sum_jw_j=1.
```

単純ですがnonconvex frontierの一部を表せず、unitに敏感です。

### Chebyshev

```math
s_w(\mathbf f)
=
\min_j w_j(f_j-z_j)
+
\rho\sum_jw_j(f_j-z_j).
```

nonconvex frontierを探索しやすい特徴があります。実装ごとのsign conventionを確認します。

### NParEGO

毎iterationでrandom weightをsampleし、Chebyshev系scalarizationとnoisy scalar acquisitionを組み合わせます。strongly negative correlated objectiveではsimple weighted sumによるcancelに注意します。

---

## 9. Preference utility

最終的に1つのdecisionが必要なら

```math
U(\mathbf f)
```

を定義できます。monetary value、desirability、target-distance、piecewise specification penalty、risk-adjusted utilityなどです。

utilityはtrade-offを解決しますが、preference assumptionを明示的に導入します。

---

## 10. Deterministic input constraint

```math
A_{\mathrm{eq}}x=b_{\mathrm{eq}},
```

```math
A_{\mathrm{ineq}}x\le b_{\mathrm{ineq}}
```

はknown constraintです。optimizer constraint、reparameterization、repair、rejection、enumerationで扱います。

step constraintは

```math
x_j=a_j+k_js_j,
\qquad k_j\in\mathbb Z.
```

compositionは

```math
\sum_{j\in S}x_j=c
```

です。independent roundingはcoupled constraintを壊すためconstraint-aware repairが必要です。

k-sparsity

```math
\|x\|_0\le k
```

はcandidate constraintであり、SAASのfeature sparsityとは異なります。

---

## 11. Unknown outcome constraint

```math
g_j(x)\le0
```

をsurrogateでmodel化し、

```math
p_j^{\mathrm{feas}}(x)
=
P(g_j(x)\le0\mid\mathcal D)
```

を求めます。

Gaussian posteriorでは

```math
p_j^{\mathrm{feas}}(x)
=
\Phi\left(
\frac{-\mu_j(x)}{\sigma_j(x)}
\right).
```

independent constraintならproductを使えますが、correlated constraintではjoint probabilityまたはposterior sampleが必要です。

classification constraint、multiclass acceptable set、ordinal minimum-grade probabilityも同様に扱えます。

---

## 12. Chance constraint

```math
P(g(x,\omega)\le0)\ge1-\epsilon
```

のrandomness`\omega`が何を表すかを明示します。

- posterior uncertainty
- input perturbation
- future measurement noise
- joint outcome uncertainty

これらは異なるfeasible setを定義します。

---

## 13. Feasibility-weighted acquisition

```math
\alpha_c(x)
=
\alpha_0(x)P(\mathrm{feasible}\mid x)
```

は簡単ですが、joint dependenceを完全には表さず、constraint数が多いとproductが極端に小さくなります。

sample-level constrained acquisitionは

```math
\hat\alpha(X)
=
\frac1S\sum_s
I^{(s)}V(Y^{(s)})
```

としてobjectiveとconstraintのsample dependenceを保てます。

---

## 14. Feasible frontierとfeasibility search

feasible Pareto setはfeasible pointの中でnondominatedな集合です。

feasible observationがまだない場合は、probability of feasibility、expected violation reduction、two-stage acquisition、safe initial pointを使います。

objective optimizationへ移行する条件を明示します。

---

## 15. Output reduction

posterior sample

```text
sample_shape x batch_shape x q x m
```

に対し、次を区別します。

- outputを保持してmulti-objective acquisitionへ渡す
- scalarizeしてsingle-objective acquisitionへ渡す
- objective outputとconstraint outputへ分ける
- 1 outputだけ選ぶ

この選択はmodelではなくobjective layerの責務です。

---

## 16. Heterogeneous output

regression value、binary probability、ordinal utility、multiclass acceptable probabilityを各scalar channelへ変換してからPareto／scalarizationへ渡します。

`HybridMultiOutputModel`はこの共通interfaceを提供しますが、`HybridPosterior`のproxy sampleはoutput間を独立に扱います。詳細は15章です。

---

## 17. q-batch、pending、repair

qEHVI／qNEHVIはq候補のunionを評価します。similar candidateは同じHV regionを改善するため、joint posterior correlationが重要です。

repair`R(X)`後は

```math
\alpha(R(X))\ne\alpha(X)
```

であるため、constraint、acquisition value、duplicate、category validityを再評価します。

---

## 18. 評価

multi-objective metric：hypervolume、HV regret、epsilon indicator、generational distance、frontier coverage。

constraint metric：feasible recommendation rate、violation count／magnitude、false-feasible calibration、feasible regret、first feasibleまでの評価回数。

experiment count、cost、unsafe evaluation、wall-clockに対してplotします。

---

## 19. `bochan`実装との対応

standard acquisitionはregistryからBoTorch classへ解決されます。

| alias | class |
|---|---|
| `qehvi`, `ehvi` | `qExpectedHypervolumeImprovement` |
| `qnehvi`, `nehvi` | `qNoisyExpectedHypervolumeImprovement` |
| `nparego` | randomized scalarizationを使うhigh-level構成 |

関連sourceは次です。

```text
src/bochan/acquisition/objective/
src/bochan/acquisition/feasible/wrapper.py
src/bochan/acquisition/feasible/constraints.py
src/bochan/models/hybrid/
src/bochan/optim/
```

---

## 20. 設定checklist

1. model output
2. objective output
3. constraint output
4. signとscale
5. probability／utility transform
6. dependence assumption
7. reference point
8. baseline
9. scalarization
10. deterministic constraint
11. probabilistic constraint
12. feasibility threshold
13. no-feasible時のbehavior
14. q／pending
15. rounding／repair
16. external metric
