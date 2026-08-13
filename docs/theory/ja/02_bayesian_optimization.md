# 02. ベイズ最適化

ベイズ最適化（Bayesian Optimization; BO）は、評価コストが高い、ノイズを含む、または解析式が得られない目的関数を少数評価で最適化する逐次意思決定手法です。確率的surrogate modelと、現在の性能および情報価値を評価する獲得関数を組み合わせます。

本章ではBOの問題設定とloopを扱います。個別の獲得関数は03章、分類・順序出力は06章、多目的問題は07章を参照してください。

---

## 1. 最適化問題

standard single-objective problemは

```math
x^*\in\arg\max_{x\in\mathcal X}f(x)
```

です。観測がnoisyなら

```math
y(x)=f(x)+\varepsilon(x).
```

設計空間はcontinuous、integer、categorical、linear／nonlinear constraint、composition、sparsityなどを含められます。

---

## 2. Posterior stateとdecision function

反復$t$までのdataを

```math
\mathcal D_t
=
\{(x_i,y_i)\}_{i=1}^{n_t}
```

とします。surrogate posteriorは

```math
p(f\mid\mathcal D_t)
```

です。

non-Gaussian outputでは、意思決定量を

```math
u(x)=T[p(y\mid x,\mathcal D_t)]
```

と明示します。例はposterior mean、success probability、ordinal expected utilityです。

---

## 3. Sequential policy

1点選択では

```math
x_{t+1}
\in
\arg\max_{x\in\mathcal X}
\alpha_t(x;\mathcal D_t)
```

q-batchでは

```math
X_{t+1}
\in
\arg\max_{X\in\mathcal X^q}
\alpha_t(X;\mathcal D_t)
```

とします。新しい観測が加わるたびにposteriorとacquisition surfaceが更新されます。

---

## 4. Exploitationとexploration

pure exploitationは

```math
x_{t+1}
\in
\arg\max_x
\mathbb E[u(x)\mid\mathcal D_t]
```

ですが、未探索領域のmodel errorを無視します。

pure uncertainty samplingはobjective valueの見込みが低い領域へ観測を使う場合があります。EI、UCB、information gain、posterior sampling、look-aheadなどは両者を異なる方法で組み合わせます。

---

## 5. Directionとscale

BoTorch acquisitionは原則maximization conventionです。minimizeしたい$g(x)$は

```math
u(x)=-g(x)
```

へ変換します。target matchingなら

```math
u(x)=-|g(x)-a|
```

などを使います。

posterior sample、`best_f`、constraint、reference point、baselineは同じobjective spaceに置きます。

---

## 6. Regret

instantaneous regret：

```math
r_t=f(x^*)-f(x_t).
```

cumulative regret：

```math
R_T=\sum_{t=1}^{T}r_t.
```

simple regret：

```math
s_T
=f(x^*)-
\max_{1\le t\le T}f(x_t).
```

final recommendation$\widehat x_T$に対するrecommendation regret：

```math
r_{\mathrm{rec}}
=f(x^*)-f(\widehat x_T).
```

実験BOではsimple regretやrecommendation regretが重要です。best observed、best latent、posterior recommendedのどれを使うかを明示します。

---

## 7. Noisy observation

```math
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\sim\mathcal N(0,\sigma_i^2)
```

では、最大observed valueはpositive noiseによりupward biasを持ちます。

Noisy Expected Improvementはbaseline latent valueもrandom variableとして扱います。

```math
\alpha_{\mathrm{NEI}}(X)
=
\mathbb E
\left[
\max\left(
\max\mathbf f_X-\max\mathbf f_B,0
\right)
\right].
```

observation noiseとinput perturbationは別です。後者は08章で扱います。

---

## 8. Initial design

代表例：

- Sobol sequence
- Latin hypercube
- factorial／fractional factorial
- historical data
- expert-selected safe point
- category combinationごとのstratified design

mixed spaceではcontinuous coverageだけでなくcategory coverageも必要です。feasible pointが1つもないconstraint problemではfeasibility searchを先に行う場合があります。

---

## 9. q-batch

joint batch acquisitionは

```math
\alpha(X),
\qquad
X=[x_1,\ldots,x_q]
```

を評価します。一般に

```math
\alpha(X)\ne\sum_i\alpha(x_i).
```

correlated candidateは情報が重複します。

- joint selection：q点を同時optimization
- sequential greedy：1点ずつpendingへ追加
- pointwise top-q：dependenceを無視するためdiversity処理が必要

---

## 10. Pendingとasynchronous experiment

実行中だがoutcome未取得のpointを`X_pending`とします。

- fantasy observation
- acquisition固有の`X_pending`
- sequential conditioning
- local penalization
- duplicate avoidance

distance penaltyはlocation重複を防ぎますが、pending outcomeのinformation valueをBayesianに積分するものではありません。

---

## 11. Constraint

```math
\max_x f(x)
\quad\mathrm{subject\ to}\quad
c_j(x)\le0
```

を考えます。

- known input constraint：optimizer／repairで扱う
- unknown outcome constraint：surrogate probabilityで扱う
- operational post-processing：rounding、composition、k-sparsity、valid category

repair後はacquisition valueとconstraintを再評価します。詳細は07章です。

---

## 12. Mixed space

```math
x=(x_c,x_g)
```

に対し、category enumeration、`optimize_acqf_mixed`、evolutionary optimizer、discrete neighborhood searchを使います。

integer category codeをcontinuous gradientで直接optimizationすることは通常不適切です。input transformがcategory columnを変更しないことも確認します。

---

## 13. High-dimensional BO

代表的な構造仮定：

- relevant original dimensionが少数：SAAS
- low-dimensional linear subspace：REMBO
- high-variance linear manifold：PCA
- nonlinear representation：VAE-GP、DKL
- local structure：trust region

これらは交換可能ではありません。詳細は14章です。

---

## 14. Look-ahead

Knowledge Gradientはfuture posterior decisionの改善を評価します。

```math
\mathrm{KG}(x)
=
\mathbb E_{y_x}
\left[
\max_{x'}\mu_{t+1}(x')
\right]
-
\max_{x'}\mu_t(x').
```

multi-step look-aheadはfuture observationとinner optimizationをnestedに扱います。model conditioning、fantasy tensor、mixed／constraint inner problemに大きな計算コストがかかります。

---

## 15. Stopping criterion

- evaluation budget到達
- acquisition valueが十分小さい
- recommendationが安定
- improvementが一定回数ない
- target achievement probabilityが十分高い／低い
- best utilityのcredible intervalが十分狭い
- operational successを達成

acquisition familyやscaleを跨いでvalue thresholdを直接比較しないようにします。

---

## 16. Evaluation protocol

1. 複数initial design
2. 複数seed
3. 同一evaluation budget
4. 同一noise／perturbation distribution
5. predictive metricとdecision metric
6. optimizer failure／duplicate rate
7. wall-clock

代表的decision metricはsimple regret、recommendation regret、feasible regret、hypervolume、target achievement probability、repeated-execution performanceです。

---

## 17. `bochan`実装との対応

高水準workflow：

```python
optimizer = BayesianOptimizer(
    model_config=...,
    acquisition_config=...,
    fit_config=...,
    bounds=bounds,
)

optimizer.fit(train_X, train_Y)
candidates = optimizer.suggest(q=q)
```

主要source：

```text
src/bochan/api/registry/model.py
src/bochan/api/registry/acquisition.py
src/bochan/api/factory.py
src/bochan/optim/
```

標準qEI、qNEI、qUCB、qPI、qKG、qEHVI、qNEHVIはregistryからBoTorch classへ解決されます。aliasだけでは`best_f`、`X_baseline`、objective、constraint、reference pointは決まりません。

---

## 18. 設定checklist

1. objectiveとdirection
2. observation typeとlikelihood
3. boundsとcategory
4. known constraint
5. unknown outcome constraint
6. modelとinference
7. input／outcome transform
8. acquisition
9. baseline／current best
10. qとpending
11. optimizerとrestart
12. rounding／repair
13. stopping／evaluation
14. random seed
