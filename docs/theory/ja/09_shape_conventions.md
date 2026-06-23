# 09. Tensor shapeとinterface contract

Tensor shapeはBoTorch互換componentの数学的定義の一部です。class axisをoutput axisとして扱う、qを早くreduceする、input perturbationを別candidateと誤認するなど、多くの不具合はaxis semanticsの誤りです。

本章を`bochan`のcanonical shape referenceとします。

---

## 1. Axis symbol

| 記号 | 意味 |
|---|---|
| `n` | training observation数 |
| `d` | raw input dimension |
| `d_internal` | transformed／projected dimension |
| `q` | jointに選択するnominal candidate数 |
| `m` | model output channel数 |
| `m_obj` | transform後objective channel数 |
| `K` | class数 |
| `n_w` | candidateあたりperturbation数 |
| `S` | posterior MC sample数 |
| `F` | fantasy sample数 |
| `H` | hyperparameter／fully Bayesian sample数 |
| `L` | DeepGP sample axis |
| `batch_shape` | optimizerが独立評価するt-batch |
| `model_batch_shape` | batched model parameter／task axis |

同じsizeでも意味の異なるaxisがあります。

---

## 2. Training data shape

### Single-output regression

```text
train_X:    n x d
train_Y:    n x 1
train_Yvar: n x 1   optional
```

### Multi-output regression

```text
train_X:    n x d
train_Y:    n x m
train_Yvar: n x m
```

同じshapeでもindependent batched output、multitask、ModelListでcovariance semanticsは異なります。

### Binary／multiclass／ordinal

```text
train_X: n x d
train_Y: n
```

binaryは0/1、multiclassとordinalはinteger labelです。ordinalは通常0から始まるconsecutive labelを要求します。

### Hybrid／ModelList

```text
train_X_j: n_j x d
train_Y_j: n_j x output_shape_j
```

outputごとに異なるdatasetを持てます。missing／asynchronous outputをdense targetへ無理に結合しません。

---

## 3. Raw inputとtransformed input

wrapperによって`train_inputs`の意味が異なります。

- raw inputを保持し`forward()`でtransform
- ordinal baseのようにtransformed inputを`train_inputs`へ保持
- DKL inner GPがfeature-space inputを保持
- PCA／REMBO／VAEがrawとlatentを両方保持

attribute名だけでraw／internal spaceを推定しないでください。

---

## 4. Candidate input

標準shapeは

```text
X: batch_shape x q x d
```

です。

```text
X: q x d
```

は1つのq-batch、

```text
X: num_restarts x q x d
```

は複数のt-batchです。

`B x q x d`に対し、`B`は独立評価するcandidate batch、`q`は1つのjoint batch内のpointです。acquisition outputは通常`B`であり`B x q`ではありません。

---

## 5. Posterior shape

### Single-output

```text
mean:     batch_shape x q x 1
variance: batch_shape x q x 1
```

custom posteriorでは最後のsingletonをsqueezeする場合があります。

### Multi-output

```text
mean:     batch_shape x q x m
variance: batch_shape x q x m
```

marginal varianceだけではcross-output covarianceを表しません。

### Multiclass probability

```text
class_probs: batch_shape x q x K
```

最後はclass axisです。

### Ordinal

```text
model.posterior(X).mean:      batch_shape x q x 1   latent
model.class_probs(X):         batch_shape x q x K
model.expected_utility(X, u): batch_shape x q
```

binary／multiclassの`posterior()`はprobability spaceですが、ordinalはlatentです。

---

## 6. Batch shapeとevent shape

posterior distributionはindependent batchとjoint eventを区別します。single-output q-point GPは概念的に

```text
batch_shape = B
event size  = q
```

です。multi-output covarianceは内部で`q*m`へflattenされる場合があります。

custom posteriorでは`batch_shape`、`event_shape`、`base_sample_shape`、`batch_range`を整合させます。

---

## 7. Posterior sampleとobjective

multi-output sampleの代表shapeは

```text
S x batch_shape x q x m
```

scalar MC objectiveは

```text
S x batch_shape x q
```

multi-objective MC objectiveは

```text
S x batch_shape x q x m_obj
```

を返します。objectiveがqを先にreduceすると、acquisition自身のbatch utilityを計算できません。

---

## 8. Fully Bayesian axis

hyperparameter sample数`H`があるとposterior meanは

```text
H x batch_shape x q x m
```

MC sample後は

```text
S x H x batch_shape x q x m
```

になります。leading axisをrankだけでaverageするとt-batchを壊す可能性があります。

---

## 9. DeepGP axis

DeepGPは

```text
L x batch_shape x q x m
```

のsample axisを追加する場合があります。wrapperはmixtureを保持、moment match、mean／between-sample varianceを計算するなどのpolicyを持ちます。

class batchやt-batchをDeepGP sample axisとしてaverageしないようにします。

---

## 10. ModelListとHybridPosterior

ModelListのscalar submodelを結合すると

```text
batch_shape x q x m
```

になります。

`HybridPosterior`も同じshapeのmean／varianceを持ちますが、current implementationはq-point間・heterogeneous output間のfull covarianceを保持しません。

shape compatibilityはjoint dependenceの保証ではありません。

---

## 11. Input perturbation

```text
X:       batch_shape x q x d
X_tilde: batch_shape x (q * n_w) x d
```

posterior sampleは

```text
S x batch_shape x q x n_w x m
```

へreshapeできます。

joint covariance

```text
batch_shape x (q*n_w) x (q*n_w)
```

をnominal q covarianceへreduceする場合、

```text
batch_shape x q x n_w x q x n_w
```

へreshapeし、両perturbation axisを平均します。

```math
[\Sigma_q]_{ij}
=
\frac1{n_w^2}
\sum_{r=1}^{n_w}
\sum_{s=1}^{n_w}
\Sigma_{(i,r),(j,s)}.
```

---

## 12. Multiclass class batch

class-wise SVGPはkernel／inducing pointのbatch shapeに`K`を持ちます。wrapperはinput

```text
batch_shape x q x d
```

へclass singletonを挿入し、内部で

```text
batch_shape x K x q x d
```

へbroadcastします。probability posteriorは

```text
batch_shape x q x K
```

を返します。このclass batchをensemble axisとしてaverageしてはいけません。

---

## 13. Ordinal boundary axis

K classならcutpointは`K-1`個です。

```text
boundary score: batch_shape x q x (K - 1)
```

`target_boundary_idx`はclassではなくcutpointを選びます。boundary reduction後は`batch_shape x q`、q reduction後は`batch_shape`です。

---

## 14. Output reductionとq reduction

score`batch_shape x q x m`に対し、output reductionとq reductionは別です。

```math
\max_q\mathrm{mean}_m a_{qm}
\ne
\mathrm{mean}_m\max_q a_{qm}.
```

mean、sum、max、minなどnonlinear reductionでは順序が結果を変えます。

---

## 15. Constraintとreference tensor

constraintが各point独立か、q全体へ作用するか、raw／normalized spaceのどちらか、rounding前後のどちらかを確認します。

pending／observed referenceはconstantとしてdetachし、model-consistent transform後のdistance spaceで比較します。

---

## 16. 代表的shape failure

- task output scaleとkernel length scale axisの不一致
- input transform二重適用によるconstant posterior
- class axisをDeepGP sample axisとしてaverage
- qとoutputをpermuteせずreshape
- `q*n_w`をq candidateとして扱う
- multitask covarianceをelement countだけでreshape
- objectiveがqをcollapse

---

## 17. Debugging手順

次を順番にprint／assertします。

```text
raw X
transformed X
posterior.mean
posterior.variance
posterior covariance / batch / event shape
posterior samples
objective(samples)
pointwise score
perturbation reduction後
output/class/boundary reduction後
final acquisition value
repaired candidates
```

shapeだけでなく

```text
samples: [S, H, B, q, m]
```

のようにaxis semanticsを記録します。

---

## 18. Assertion例

```python
assert X.shape[-1] == input_dim
assert posterior.mean.shape[-2] == X.shape[-2]
assert objective_values.shape[-1] == X.shape[-2]
assert acq_value.shape == X.shape[:-2]
```

multiclassでは

```python
assert probs.shape[-1] == num_classes
assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]))
```

perturbationでは

```python
assert q_expanded == q * n_w
```

を確認します。

---

## 19. `bochan`実装との対応

```text
src/bochan/models/
src/bochan/models/components/heteroscedastic.py
src/bochan/models/components/multiclass.py
src/bochan/models/hybrid/posterior.py
```

acquisition baseにはpointwise score alignment、extra sample axis reduction、covariance extraction、q-batch finalization helperがあります。

多くのacquisitionは`@t_batch_mode_transform()`を使いますが、custom output、class、perturbation axisまでは自動解決しません。

custom posteriorはBoTorch sampler dispatcherへの登録が必要な場合があります。

---

## 20. 新規componentのshape template

model：

```text
train_X
train_Y
raw X
internal X
posterior mean
posterior variance
posterior sample
model batch axis
output/class axis
observation_noise behavior
```

objective：

```text
input sample shape
output shape
reduced axis and order
q*n_w behavior
```

acquisition：

```text
X shape
posterior accessor
pointwise score
class/output/boundary reduction
q reduction
final shape
```

最低限、`q=1`、`q>1`、multiple t-batch、multi-output、`n_w>1`、mixed input、pending、DeepGP、ensemble、gradientをtestします。
