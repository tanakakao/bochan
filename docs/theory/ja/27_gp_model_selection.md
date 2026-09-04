# 27. GPモデルの選択：SingleTask・MultiTask・SAAS・Deep Kernel

獲得関数を選ぶ前に、posteriorが問題の構造を適切に表現している必要があります。本章では、`bochan` / BoTorchで利用するGP系モデルを「何を仮定しているか」「いつ使うか」という観点で整理します。

## 1. モデル選択の基本

最初に確認するのは次の軸です。

1. outputは連続値か、分類・ordinal・count等か
2. outputは1つか複数か
3. output間相関を学習したいか
4. 入力次元は高いか
5. raw inputに適切な距離を定義できるか
6. noiseは一定か入力依存か
7. fidelity/task/sourceの違いがあるか
8. データ量に対してモデル複雑度が妥当か

複雑なモデルほど常に良いわけではありません。BOでは平均予測精度だけでなくposterior uncertaintyの品質が意思決定へ直接影響します。

---

## 2. SingleTaskGP：標準baseline

典型的な回帰は

```math
y_i=f(x_i)+\epsilon_i,
\qquad
f\sim\mathcal{GP}(m,k)
```

です。

単一output、連続入力、比較的少量のデータでは、まず標準GPをbaselineにします。

### 向いている場合

- 単一の連続目的変数
- 数十〜数百点程度のsmall-data BO
- 入力次元が低〜中程度
- smoothness assumptionが妥当
- uncertaintyをBOへ利用したい

### 利点

- posteriorの意味が明確
- acquisitionとの互換性が高い
- 学習・診断が比較的容易
- ARD lengthscaleを確認できる

ただしARD lengthscaleは因果的重要度ではありません。

---

## 3. Known noiseとlearned noise

観測noiseが既知なら

```math
\epsilon_i\sim\mathcal N(0,\sigma_i^2)
```

として`train_Yvar`を利用できます。

同じ測定条件でも既知の標準誤差がある場合、noiseを再推定するより情報を与える方が合理的です。

一方、noiseが未知ならlikelihood parameterとして推定します。

重要なのはposterior latent uncertaintyとmeasurement noiseを区別することです。

---

## 4. Heteroscedastic model：入力依存noise

工程条件によって測定ばらつきが変わる場合、

```math
y=f(x)+\epsilon(x),
\qquad
\epsilon(x)\sim\mathcal N(0,\sigma^2(x))
```

と考えます。

homoscedastic GPで大きなnoise領域を説明すると、function uncertaintyとobservation noiseが混同される場合があります。

### 向いている例

- 高温域ほど測定ばらつきが増える
- 装置・条件で再現性が異なる
- repeated measurementsからlocal varianceを推定できる

ただしnoise model自体にもデータが必要です。少量データでは過剰な複雑化になり得ます。

---

## 5. Multi-outputとModelList

複数output

```math
\mathbf y(x)=[y_1(x),\ldots,y_m(x)]
```

があっても、必ず相関modelが必要とは限りません。

各outputを独立GPで学習するModelList型は

```math
p(\mathbf f\mid D)
=\prod_j p(f_j\mid D_j)
```

とみなせます。

### 向いている場合

- output間相関を仮定したくない
- outputごとにnoiseやtraining dataが異なる
- 実装の単純さ・安定性を優先

multi-objectiveだからmulti-task GPが必須、というわけではありません。独立posteriorでもEHVI/NEHVIを構成できます。

---

## 6. MultiTaskGP

output/task間の情報共有を明示的にmodel化する場合、例えば

```math
\operatorname{Cov}[f_a(x),f_b(x')]
=B_{ab}k(x,x')
```

とします。

あるtaskの観測が別taskのposteriorを改善できるのが利点です。

### 向いている場合

- task間に統計的関連がある
- 一部taskのデータが少ない
- 同じ入力領域を複数taskで評価している

### 注意

相関を誤って仮定するとnegative transferが起こります。raw target correlationが高いだけでmulti-taskが必ず有効とは限りません。

---

## 7. KroneckerMultiTaskGP

全taskが共通の入力gridで観測される場合、covarianceを

```math
K=B\otimes K_X
```

のようにKronecker構造で扱えることがあります。

### 向いている場合

- 同一の`X`について全output/taskを観測
- task間相関を使いたい
- dense multi-output table

### 向いていない場合

- taskごとに入力点が大きく異なる
- missing outputが多い
- task構造を共有する根拠が弱い

`KroneckerMultiTaskGP`のtask covarianceは「8個の目的を単に一緒に持っている」こととは異なり、cross-task statistical structureをmodel化します。

---

## 8. 高次元：ARDの限界

入力次元`d`が増えると、通常GPはlengthscaleを多数推定する必要があります。

データ数`n`に対して`d`が大きい場合、posterior hyperparameter uncertaintyが大きくなり、acquisitionも不安定になります。

高次元では、実際に効くdimensionが少ないというsparsity assumptionを利用できる場合があります。

---

## 9. SAAS GP

SAASは多くのdimensionを実質的にinactiveとするstrong sparsity priorを用いる高次元BO向けの考え方です。

概念的にはinverse lengthscale

```math
\rho_j=1/\ell_j
```

の多くを0付近へshrinkし、一部だけ大きくするpriorを置きます。

### 向いている場合

- 高次元だがeffective dimensionは低いと考えられる
- expensive experimentでデータが少ない
- fully Bayesian inferenceの計算コストを許容できる

### 注意

SAASはcandidateのk-sparsity制約

```math
\|x\|_0\le k
```

とは別物です。SAASがsparseにするのは主にfunction sensitivityです。

---

## 10. Deep Kernel Learning

raw inputでstationary kernelが適切でない場合、neural encoder

```math
z=h_\theta(x)
```

を通し、

```math
k_{\mathrm{DKL}}(x,x')
=k_0(h_\theta(x),h_\theta(x'))
```

とします。

### 向いている場合

- 高次元descriptor
- image / spectrum / learned materials representation
- CrabNet / Roost / ALIGNN等のencoder representation
- raw Euclidean distanceが物理的類似度を表しにくい

### リスク

small dataで大きなencoderをend-to-end trainingすると、representationとuncertaintyの両方が不安定になり得ます。

比較すべきbaselineは

```text
raw GP
frozen encoder + GP
fine-tuned encoder + GP
end-to-end DKL
```

です。

---

## 11. Deep GP

Deep GPはGPを階層的に合成し、

```math
f(x)=f_L(f_{L-1}(\cdots f_1(x)))
```

のような非stationary・非線形function priorを表現します。

DKLではdeterministic neural representation + GPであるのに対し、Deep GPでは中間mapping自体も確率過程です。

### 利点

- nonstationarityを柔軟に表現
- 階層的なuncertainty propagation

### 欠点

- variational inferenceが複雑
- training costが高い
- posterior sampling/acquisitionが重い
- small-data BOで単純GPを上回る保証はない

---

## 12. Mixed input

continuous + categorical入力では、categoryを単純な連続値として距離計算すると、例えばcategory `0,1,2` に偽の順序を導入します。

mixed kernel/modelでは

```text
continuous similarity
× categorical similarity
```

のように変数型を区別します。

材料探索では元素種、装置、雰囲気、process routeなどで重要です。

---

## 13. Multi-fidelity GP

fidelity変数`s`を含め

```math
y=f(x,s)+\epsilon
```

とします。

例:

- coarse simulation / high-accuracy DFT
- MLIP / DFT
- simulation / experiment
- short / long experiment

重要なのはfidelity間のcorrelationとcost差を利用して、`x`だけでなく評価sourceも意思決定できる点です。

Residual GP

```math
y_H(x)=f_0(x)+\delta(x)
```

とは目的が異なります。residual correctionは固定baselineのbias補正、multi-fidelityは複数情報源をmodel内で扱う設計です。

---

## 14. Non-Gaussian GP

応答分布がGaussianでない場合、latent GP

```math
f\sim\mathcal{GP}
```

に対してlikelihoodを変更します。

| 応答 | 代表likelihood |
|---|---|
| binary | Bernoulli |
| multiclass | Categorical |
| ordinal | ordered likelihood |
| count | Poisson / Negative Binomial |
| `(0,1)`連続値 | Beta |
| 正の連続値 | Gamma |

posteriorが通常closed formでなく、variational inference等が必要になります。

ここで重要なのは、latent `f`のGaussian uncertaintyをそのまま目的物性のGaussian uncertaintyと解釈しないことです。likelihoodを通したpredictive quantityをacquisitionへ渡します。

---

## 15. モデル選択表

| 状況 | 第一候補 |
|---|---|
| 単一連続output・低〜中次元 | SingleTaskGP |
| 既知の観測variance | known-noise GP |
| 入力依存noise | heteroscedastic model |
| 複数独立output | ModelList |
| 関連task | MultiTaskGP |
| 共通Xのdense task output | KroneckerMultiTaskGP |
| 高次元・effective dimension小 | SAAS |
| learned representationが必要 | frozen encoder + GP / DKL |
| 強いnonstationarity | Deep GPを比較候補にする |
| continuous + categorical | mixed GP |
| 複数fidelity | multi-fidelity GP |
| binary/count/positive等 | task-specific non-Gaussian GP |

---

## 16. BOでは予測RMSEだけで選ばない

BOで必要なのはpoint predictionだけではありません。

例えば2モデルのRMSEが同程度でも、posterior varianceが過小ならEI/UCB/BALD等の探索挙動は悪化します。

少なくとも次を確認します。

- predictive accuracy
- negative log predictive density等
- interval coverage
- calibration
- extrapolation behavior
- acquisitionによるsequential regret / discovery performance

最終的には「最も予測精度が高いmodel」ではなく、「現在の意思決定problemに必要なposteriorを提供するmodel」を選びます。

次章では、このモデル選択をacquisition・workflowと組み合わせたdecision mapへまとめます。
