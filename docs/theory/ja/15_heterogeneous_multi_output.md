# 15. 異種多出力モデル

異種多出力問題では、1回の実験からcontinuous response、binary label、ordinal grade、multiclass failure modeなど、異なるsample spaceを持つ複数の応答が得られます。

```math
\mathbf y(x)
=
[y_{\mathrm{property}},
 y_{\mathrm{pass/fail}},
 y_{\mathrm{grade}},
 y_{\mathrm{failure\ mode}}].
```

これらを1つのGaussian target tensorへ単純にstackしてはいけません。各出力に対応するlikelihoodとposterior semanticsが必要です。

Pareto optimization、scalarization、constraintは07章で扱います。本章ではmodelとposterior constructionに集中します。

---

## 1. Homogeneousとheterogeneous output

### Homogeneous output

同じ応答型・同じlikelihood familyを共有します。例は複数のcontinuous property、複数のbinary labelです。

### Heterogeneous output

```math
y_r\in\mathbb R,
\qquad
y_b\in\{0,1\},
```

```math
y_o\in\{0,\ldots,K_o-1\},
\qquad
y_c\in\{0,\ldots,K_c-1\}.
```

各outputに異なるlikelihoodとprediction interpretationが必要です。

---

## 2. 3つのmodeling level

### 2.1 Independent submodel

```math
p(f_1,\ldots,f_m\mid\mathcal D)
=
\prod_{j=1}^{m}p(f_j\mid\mathcal D_j).
```

outputごとにkernel、likelihood、input transform、training data、fit procedureを変えられます。

### 2.2 Correlated latent heterogeneous model

shared latent processを使い、task-specific likelihoodへ接続します。output間で情報共有します。

### 2.3 Objective-space wrapper

submodelを独立に保持し、各outputをscalar decision channelへ変換して、BoTorch acquisitionへ共通interfaceを提供します。

現在の`HybridMultiOutputModel`は主に3つ目を実装します。wrapper自身はcross-output statistical dependenceを新たに導入しません。

---

## 3. Independent submodel

output$j$のdataを

```math
\mathcal D_j
=
\{(x_{ij},y_{ij})\}_{i=1}^{n_j}
```

とします。$n_j$やinput setがoutputごとに異なっていて構いません。

利点：

- 異なるlikelihoodを使える
- missing／asynchronous outputを自然に扱える
- output-specific noise modelを設定できる
- negative transferを避けられる

制約：

- cross-output information transferがない
- cross-output posterior covarianceがない
- joint event probabilityは独立近似になる

---

## 4. Shared latent heterogeneous model

shared latent GPを

```math
u_q(x)\sim\mathcal{GP}(0,k_q),
\qquad q=1,\ldots,Q
```

とし、output$j$のlatent predictorを

```math
f_j(x)
=
\sum_{q=1}^{Q}a_{jq}u_q(x)
```

とします。

latent covarianceは

```math
\mathrm{Cov}[f_j(x),f_l(x')]
=
\sum_{q=1}^{Q}a_{jq}a_{lq}k_q(x,x').
```

各outputにはGaussian、Bernoulli、ordered-logit、Categorical、Poissonなどtask-specific likelihoodを適用します。

これはLinear Model of Coregionalizationの一種です。情報共有できる一方、identifiability、negative transfer、inference costが増えます。

---

## 5. Raw correlationとの違い

latent covarianceとobserved label correlationは同じではありません。観測上のdependenceは次に影響されます。

- latent covariance
- link function
- class imbalance
- cutpoint
- observation noise
- missing pattern
- input distribution

raw label correlationをtask covarianceとして直接解釈しないでください。

---

## 6. Missing・asynchronous output

observation indicatorを

```math
m_{ij}=\mathbf1[y_{ij}\text{ is observed}]
```

とするとlikelihoodは

```math
p(\mathbf y_{\mathrm{obs}}\mid\mathbf f)
=
\prod_{i,j:m_{ij}=1}p_j(y_{ij}\mid f_j(x_i)).
```

です。

missing valueをfabricated targetで埋めてdense `n x m` tensorにするとlikelihoodが変わります。independent submodelではoutputごとのdataをauthoritativeとします。

outputの到着時刻が異なる場合、data stateは

```math
\mathcal D_t
=(\mathcal D_{1,t},\ldots,\mathcal D_{m,t})
```

です。hybrid wrapperは全outputが同じtraining rowやpending statusを持つと仮定してはいけません。

---

## 7. Decision-space transformation

各outputをscalar channelへ変換します。

```math
t_j(x)
=
T_j[p_j(y_j\mid x,\mathcal D_j)].
```

combined vectorは

```math
\mathbf t(x)
=[t_1(x),\ldots,t_m(x)].
```

例：

### Regression

```math
t_j=s_jw_jy_j.
```

### Binary probability

```math
t_j=P(Y_j=c^*\mid x).
```

### Multiclass acceptable-set probability

```math
t_j=\sum_{k\in A_j}P(Y_j=k\mid x).
```

### Ordinal expected utility

```math
t_j=\sum_ku_{jk}P(Y_j=k\mid x).
```

変換によって共通Tensor channelは得られますが、statistical dependenceは新しく生まれません。

---

## 8. `OutputSpec`

`bochan.models.hybrid.OutputSpec`は各scalar channelを定義します。

```text
name
task_type
model
output_index
sign
weight
eq_target
utility_values
positive_class
transform
```

- `name`：安定したoutput識別子
- `task_type`：regression／binary／ordinal／multiclass
- `output_index`：submodel内のoutput選択
- `sign`, `weight`：directionとlinear scaling
- `eq_target`：target-distance objective
- `utility_values`：class probabilityからexpected utility
- `positive_class`：選択class probability
- `transform`：custom scalar transform

nonlinear transformでは

```math
h(\mathbb E[Y])\ne\mathbb E[h(Y)]
```

なので、meanだけにtransformを適用した場合はexact transformed momentではありません。

---

## 9. Posterior mode

wrapperは次のようなmodeを扱います。

```text
objective
mean
latent
probability
expected_utility
```

意味はtask typeによって異なります。意味のないmodeを暗黙fallbackさせず、明示的にerrorとする方が安全です。

---

## 10. Class utility moment

class probability$p_k$とutility$u_k$に対して

```math
\mu_U=\sum_kp_ku_k
```

```math
\sigma_U^2
=
\sum_kp_k(u_k-\mu_U)^2.
```

これはgiven class probability下のrealized discrete utilityのmomentです。probability functionのepistemic uncertaintyは別途integrateする必要があります。

---

## 11. `HybridPosterior`

current `HybridPosterior`は

```text
mean:     batch_shape x q x m
variance: batch_shape x q x m
```

を持ち、proxy sampleを

```math
T_j^{(s)}
=
\mu_j+\sqrt{v_j}\epsilon_j^{(s)},
\qquad
\epsilon_j^{(s)}\sim\mathcal N(0,1)
```

として生成します。

full covariance matrixは保持しないため、proxy samplingではq candidate間・heterogeneous output間が独立です。

この近似が影響するもの：

- joint tail probability
- scalarization variance
- joint chance constraint
- q-batch redundancy
- multi-output information gain
- hypervolume distribution

BoTorch interoperabilityには有用ですが、full correlated heterogeneous posteriorではありません。

---

## 12. Alternative posterior construction

1. native posterior sampleをtaskごとにdrawし、sample-wiseにtransform
2. marginal distributionをcopulaでcouple
3. shared-latent heterogeneous variational GP
4. independent submodel + calibrated residual dependence model

sample-wise native transformはnonlinear utilityやdiscrete supportをより忠実に保持します。

---

## 13. Input transform

submodelが同一transform objectを共有する場合だけ、wrapperでcommon `input_transform`として扱えます。

注意点：

- submodelごとにraw／transformed input conventionが異なる
- categorical dimensionが異なる
- PCA／VAEなど異なるlatent mappingを使う
- 一部だけ`InputPerturbation`でqを展開する
- distance penaltyがshared transformed spaceを仮定する

原則としてraw candidateを各submodelへ渡し、submodel自身にtransformさせます。

---

## 14. Conditioningとfantasy

hybrid look-aheadにはoutputごとのcondition supportが必要です。

- Gaussian regression：exact conditioning可能
- variational classification：再構築／近似refit
- ordinal：variational optimizationが必要
- asynchronous output：観測されるoutput subsetが異なる

submodelがcompatible conditioningを提供しない場合、wrapperだけでexact joint fantasyは実現できません。

---

## 15. Calibrationとmodel selection

各submodelをtask適切なmetricで評価します。

- regression：RMSE、NLPD、coverage
- binary：Brier、log loss、calibration
- multiclass：categorical log loss、class-wise calibration
- ordinal：RPS、cumulative calibration、class MAE

独立modelをbaselineとし、correlated modelがdata-poor output、joint probability、sequential decisionを実際に改善するか検証します。

---

## 16. `bochan`実装との対応

```text
src/bochan/models/hybrid/multi_output.py
src/bochan/models/hybrid/specs.py
src/bochan/models/hybrid/posterior.py
src/bochan/models/hybrid/prediction.py
src/bochan/acquisition/objective/hybrid.py
src/bochan/api/factory.py
src/bochan/api/configs.py
```

current contract：

1. submodelからscalar channelを選択
2. `OutputSpec`とposterior modeで変換
3. mean／varianceを`[..., q, m]`へstack
4. `HybridPosterior`を返す
5. wrapper自身はcross-output covarianceを追加しない

---

## 17. 確認項目

1. 各outputのsupportとlikelihood
2. independentかcorrelatedか
3. missing／asynchronous outputの表現
4. posterior mode
5. decision-space transformation
6. variance channelの意味
7. nonlinear transformのmoment処理
8. cross-output covarianceの有無
9. q-point covarianceの有無
10. input transform compatibility
11. conditioning support
12. outputごとのpredictive metric
13. combined decision metric
