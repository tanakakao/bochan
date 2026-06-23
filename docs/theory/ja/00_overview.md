# 00. 全体像と読書ガイド

このディレクトリは、`bochan`の理論リファレンスです。互いに独立した機能メモを並べるのではなく、参考書として順番に読める構成としています。

目的は、実用的なベイズ最適化ソフトウェアで一貫していなければならない次の4層を接続することです。

1. 数学的な問題設定
2. 確率モデル
3. 意思決定基準
4. `bochan`上の具体的な実装

そのため各章には、理論だけでなく実装との対応を記載します。数式は各構成要素の意味を説明し、ソースマップはその対象を実装するクラス、メソッド、Tensor、ディレクトリを示します。

---

## 1. 対象範囲

`bochan`は、主に次の3種類の逐次設計を扱います。

### ベイズ最適化

高い効用を持つ入力を探索します。

```math
x^*\in\arg\max_{x\in\mathcal X}u(x).
```

### 能動学習

モデルを改善し、不確かさを減らし、または情報利得を最大化する観測点を選択します。

```math
x_{t+1}\in\arg\max_{x\in\mathcal X}
I(\text{future observation};\text{learning target}\mid\mathcal D_t).
```

### レベル集合推定

次のような領域や境界を同定します。

```math
L_h^+=\{x:f(x)\ge h\},
\qquad
B_h=\{x:f(x)=h\}.
```

同じsurrogate modelを3つすべてに利用できる場合がありますが、獲得関数と評価損失は異なります。最適化に適したモデルを使っているだけでは、能動学習やLSEを正しく行っているとは限りません。

---

## 2. 逐次設計ループ

反復` t `における観測データを

```math
\mathcal D_t
=\{(x_i,y_i)\}_{i=1}^{n_t}
```

とします。

確率モデルは事後分布

```math
p(f\mid\mathcal D_t)
```

を定義します。

objectiveまたはposterior transformは、モデル出力を意思決定に必要な量へ変換します。獲得関数は、候補batch `X`でデータを取得する価値

```math
\alpha_t(X;\mathcal D_t)
```

を評価します。

次のbatchは

```math
X_{t+1}
\in
\arg\max_{X\in\mathcal X^q}
\alpha_t(X;\mathcal D_t)
```

として選択されます。

実験やsimulatorから新しい観測が得られたら、データへ追加して同じサイクルを繰り返します。

実装上は次の流れです。

```text
training data
    -> model construction
    -> model fitting
    -> posterior / samples
    -> objective or posterior transform
    -> acquisition function
    -> acquisition optimizer
    -> candidate post-processing
    -> experiment
    -> updated training data
```

`bochan`では、モデルの仮定、利用者の選好、optimizer制約が暗黙に混ざらないよう、これらの段階を分離します。

---

## 3. 区別すべき4つの数学空間

実装上の重大な誤りの多くは、異なる空間を交換可能なものとして扱うことで起こります。

### 3.1 入力空間

元の設計変数は

```math
x\in\mathcal X\subseteq\mathbb R^d
```

または連続変数とカテゴリ変数の混合空間に属します。

モデル内部では

```math
z=T(x)
```

を利用する場合があります。`T`は正規化、PCA、REMBO、VAE encoder、neural feature mapなどです。

内部表現を使う場合でも、candidate optimizationがどの空間で行われ、元の入力との逆写像またはwrapper関係がどう定義されるかを明確にしなければなりません。

### 3.2 潜在関数空間

GPは通常、

```math
f(x)
```

をモデル化します。

回帰では観測応答に近い量であることが多い一方、分類や順序モデルでは単なる潜在scoreです。

### 3.3 観測空間

尤度は潜在値から観測への確率分布を定義します。

```math
y\sim p(y\mid f(x)).
```

例として、Gaussian応答、Bernoulli label、Categorical label、順序class、count、正の連続量があります。

### 3.4 意思決定空間／目的空間

利用者が価値を置く量を

```math
u(x)
=
T_{\mathrm{decision}}
[p(y\mid x,\mathcal D)]
```

とします。

例は次のとおりです。

- 回帰応答
- 成功確率
- 制約を満たす確率
- 順序classの期待効用
- 多目的vector
- 入力摂動下のリスク尺度

threshold、`best_f`、reference point、constraintは、それを利用する獲得関数と同じ意思決定空間で表現する必要があります。

---

## 4. モデルの分類

リポジトリでは、次の応答型を扱います。

| 応答型 | 代表的な尤度 | 主な予測対象 |
|---|---|---|
| 連続Gaussian | Gaussian | 平均、共分散、sample |
| 有界連続量 | Beta | 有界な応答分布 |
| 正の連続量 | Gamma | 正の応答分布 |
| count | Poisson / Negative Binomial | count分布 |
| 二値label | Bernoulli | class確率と潜在score |
| 非順序多クラスlabel | Categorical / softmax | class確率vector |
| 順序class | ordered logit | class確率、cutpoint、utility |
| 複数の同種出力 | 共通または独立の尤度 | vector posterior |
| 異種出力 | task別の尤度 | 変換後のobjective vector |

同じ応答型に対しても、次の構成を組み合わせられます。

- exact inferenceまたはvariational inference
- 連続入力または混合入力
- homoscedastic noiseまたはheteroscedastic noise
- Deep KernelまたはDeepGP
- 高次元priorまたはprojection
- single-output、multi-output、hybrid wrapper

---

## 5. 意思決定コンポーネントの役割

`bochan`では、次の役割を区別します。

### Model

潜在関数または予測応答に関する不確かさを表現します。

### Likelihood

潜在変数を条件とした観測分布を定義します。

### Posterior transform

posteriorの表現を変換します。たとえば潜在logitから、選択したclass確率や効用へ変換します。

### MC objective

posterior sampleをscalarまたはvectorの意思決定値へ写像します。

### Acquisition function

1点または複数点を観測する価値を定義します。

### Acquisition optimizer

search-space constraintの下で獲得関数を最大化します。

### Candidate post-processing

optimization中またはoptimization後に、丸め、repair、sparsity、domain固有の妥当性制約を適用します。

これらを同じ役割として扱ってはいけません。たとえば`x`に対する線形制約はoptimizer制約ですが、実験成功確率はモデル化されたoutcome constraintです。

---

## 6. 本書の構成

各トピックには1つの主担当章を割り当てています。

### Part I：基礎理論と逐次意思決定

| 章 | ファイル | 主な役割 |
|---|---|---|
| 00 | `00_overview.md` | 用語、構造、読書ガイド |
| 01 | `01_gaussian_process_models.md` | GP確率論、条件付き分布、推論、posterior契約 |
| 02 | `02_bayesian_optimization.md` | optimization、regret、逐次loop、q-batch、noise |
| 03 | `03_acquisition_functions.md` | BO獲得関数とそのoptimization |
| 04 | `04_active_learning.md` | モデル学習のための情報量と不確かさ |
| 05 | `05_level_set_estimation.md` | LSE問題、loss、confidence set、評価 |
| 06 | `06_classification_and_ordinal_bo.md` | 二値・多クラス・順序BOのdecision objective |
| 07 | `07_multi_objective_and_constraints.md` | Pareto、hypervolume、scalarization、constraint |
| 08 | `08_input_perturbation_and_risk.md` | robust objective、chance constraint、VaR、CVaR |
| 09 | `09_shape_conventions.md` | Tensor軸とinterface contract |

### Part II：モデル族と実装詳細

| 章 | ファイル | 主な役割 |
|---|---|---|
| 10 | `10_regression_models_and_likelihoods.md` | Gaussianおよびnon-Gaussian回帰 |
| 11 | `11_classification_models.md` | 二値・多クラス分類モデル |
| 12 | `12_ordinal_models.md` | ordered-logit、cutpoint、順序不確かさ |
| 13 | `13_heteroscedastic_and_robust_models.md` | 入力依存noise、outlier、robust likelihood |
| 14 | `14_deep_and_high_dimensional_models.md` | DKL、DeepGP、SAAS、PCA、REMBO、VAE-GP |
| 15 | `15_heterogeneous_multi_output.md` | 異種尤度とhybrid posterior |
| 16 | `16_level_set_mathematics_and_implementation.md` | task別LSEの実装式とsource correspondence |

基礎章では共通概念を一度だけ定義し、詳細章ではモデル固有・実装固有の挙動に集中します。

---

## 7. 推奨読書順

### 初めてガウス過程ベイズ最適化を使う場合

1. 00章
2. 01章
3. 10章
4. 02章
5. 03章
6. 09章

### 分類または順序最適化

1. 00章、01章
2. 11章または12章
3. 06章
4. 03章
5. 09章

### 能動学習

1. 01章、04章
2. 対象モデルの章（10、11、12章）
3. 09章
4. 観測ノイズが入力依存なら13章

### レベル集合推定

1. 05章
2. 対象モデルの章
3. 16章
4. robust／perturbed LSEなら08章、09章

### 多目的・異種出力

1. 07章
2. 15章
3. 確率または順序効用をobjectiveにする場合は06章、12章
4. 09章

### 高次元・deep model

1. 01章、10章
2. 14章
3. 02章、03章
4. 09章

---

## 8. 記号

| 記号 | 意味 |
|---|---|
| $n$ | 観測数 |
| $d$ | 入力次元 |
| $q$ | 同時に選択するcandidate数 |
| $m$ | 出力数またはobjective次元数 |
| $K$ | class数 |
| $n_w$ | nominal candidateあたりの入力摂動sample数 |
| $X$ | candidate tensorまたはdesign matrix |
| $\mathcal D_t$ | 反復`t`までの観測データ |
| $f$ | 潜在関数 |
| $y$ | 観測応答 |
| $u$ | utilityまたはobjective value |
| $k$ | covariance kernel |
| $\mu,\Sigma$ | posterior meanとcovariance |
| $\alpha$ | acquisition function。文脈によりrisk level |
| $h$ | level-set threshold |

同じGreek symbolが複数分野で一般的に使われる場合、各章で局所的な意味を明示します。

---

## 9. 実装アーキテクチャ

主要なsource directoryは、数学上の役割分離に対応します。

```text
src/bochan/models/          probabilistic models and posterior wrappers
src/bochan/likelihoods/     custom observation likelihoods
src/bochan/fit/             model-specific fitting procedures
src/bochan/acquisition/     BO, Active Learning, and LSE acquisitions
src/bochan/optim/           acquisition optimizers and candidate repair
src/bochan/api/             high-level configuration and registries
```

代表的な高水準registryは次のとおりです。

```text
src/bochan/api/model_registry.py
src/bochan/api/acquisition_registry.py
```

registryは利用者向けのmodel名・acquisition名を具体的なclassへ解決します。理論的な意味はalias名だけではなく、最終的に解決されたclassによって決まります。

---

## 10. Posterior空間の契約

各model familyについて、少なくとも次を明示する必要があります。

1. `posterior(X)`はどの確率変数を表すか
2. latent、predictive、probability、utility、proxyのどれか
3. varianceにobservation noiseを含むか
4. `rsample()`は`X`に関して微分可能か
5. 最後のoutput軸は何か
6. model-batch、task、DeepGP sampleなどの追加軸があるか
7. training時・evaluation時にどのinput transformを適用するか
8. look-ahead acquisitionのためのcondition／fantasyを利用できるか

現在の重要な違いは次のとおりです。

- binary classificationの`posterior()`はprobability-space posteriorを返す
- multiclass classificationの`posterior()`はclass probabilityを返す
- base ordinalの`posterior()`はscalar latent posteriorを返し、`class_probs()`が順序class probabilityを返す
- `HybridPosterior`は共通output空間の周辺平均・分散を保持するが、出力間共分散を表現しない

これらは偶然の実装詳細ではなく、獲得関数との互換性を決める契約です。

---

## 11. 各章の記述規約

可能な限り、各章を次の順序で記述します。

1. 解くべき問題
2. 数学的な定式化
3. 統計的仮定と近似
4. BoTorch／GPyTorch概念との対応
5. `bochan`のclass・fileとの対応
6. posteriorとTensor shapeの契約
7. exact generative modelではなく近似である箇所

実用的なベイズ最適化で起きる問題の多くは、概念の誤りよりもinterfaceの誤りです。latentとpredictive、probabilityとutility、epistemicとobservation variance、scalarとheterogeneous output、`q`と`q * n_w`を区別することが重要です。

---

## 12. 本リファレンスの位置づけ

この文書は、現在のリポジトリ実装を説明します。

一部は標準的な統計モデルのexact implementationですが、別の一部はBoTorch互換posteriorやacquisition interfaceを提供するためのengineering approximationです。

特に、次の近似は明示的に区別します。

- residualに基づくheteroscedastic noise fitting
- `HybridPosterior`における独立normal proxy sampling
- DeepGP sample次元のmoment reduction
- diversityを与える実用的なdistance penalty
- 完全なgenerative modelとは異なるscore-level risk aggregation

一般的な教科書用語と現在の実装が異なる場合は、実際の数式とposterior契約を優先します。
