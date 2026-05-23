# Part I. モデル

本 Part では、本実装で扱う Gaussian Process 系モデルの全体像を整理する。  
まず、回帰・分類・順序回帰というタスク別の基本概念を説明し、その後に Deep Kernel GP、Deep GP、Heteroscedastic GP、Robust Relevance Pursuit、SAAS GP、PCA / REMBO などの拡張モデルを説明する。

本実装では、モデルは単に予測値を返すための部品ではなく、Bayesian Optimization、Active Learning、Level-set Estimation の各獲得関数に対して、平均・分散・確率・潜在スコア・期待効用などを提供する確率モデルとして位置づける。

---

## 1. モデル全体の整理

### 1.1 本実装で扱うモデルの全体像

本実装では、以下の 3 種類のタスクを基本単位として扱う。

1. **回帰**
2. **分類**
3. **順序回帰**

これらは観測値 `Y` の意味、likelihood、posterior の解釈が異なる。

| タスク | 観測値 `Y` | 代表的 likelihood | posterior の主な意味 | 代表的な用途 |
|---|---|---|---|---|
| 回帰 | 連続値 | Gaussian likelihood | 目的値の平均・分散 | 目的値の最大化・最小化 |
| Non-Gaussian 回帰 | カウント・正値・比率など | Poisson / Gamma / Negative Binomial / Beta | link function 後の平均・不確かさ | 非正規データの最適化 |
| 2値分類 | 0 / 1 | Bernoulli likelihood | 潜在スコアまたはクラス確率 | 良否判定、feasibility 判定 |
| 多クラス分類 | 0, 1, ..., K-1 | Categorical / Dirichlet 近似など | 各クラス確率 | 複数カテゴリ分類 |
| 順序回帰 | 順序付きカテゴリ | Ordinal likelihood | 潜在スコア・クラス確率・期待効用 | 段階評価、グレード評価 |

さらに、各タスクに対して以下の拡張モデルを組み合わせる。

| 拡張モデル | 主な目的 |
|---|---|
| Deep Kernel GP | ニューラルネットワークによる特徴抽出と GP の組み合わせ |
| Deep GP | GP を多層化し、より柔軟な非線形性を表現 |
| Heteroscedastic GP | 入力依存ノイズを扱う |
| Robust Relevance Pursuit | 関連変数を抽出しながら GP を構築 |
| SAAS GP | 高次元入力のうち少数の有効変数を仮定する sparse な GP |
| PCA GP | データに基づく線形次元削減 |
| REMBO GP | ランダム埋め込みによる高次元 Bayesian Optimization |

---

### 1.2 回帰・分類・順序回帰の違い

回帰・分類・順序回帰は、いずれも「入力 `x` に対して出力 `y` を予測する」という点では共通している。  
しかし、出力 `y` の性質が異なるため、GP の使い方も変わる。

#### 回帰

回帰では `y` は連続値である。  
典型的には、潜在関数 `f(x)` に GP 事前分布を置き、観測ノイズを加えた形で表す。

```text
f(x) ~ GP(m(x), k(x, x'))
y = f(x) + ε
ε ~ N(0, σ²)
```

この場合、`posterior.mean` は目的値の予測平均、`posterior.variance` は目的値の不確かさとして解釈できる。

#### 分類

分類では `y` はカテゴリラベルである。  
2値分類では、潜在関数 `f(x)` を sigmoid 関数などで確率に変換する。

```text
f(x) ~ GP(m(x), k(x, x'))
p(y = 1 | x) = sigmoid(f(x))
```

この場合、posterior をどの空間で返すかが重要である。  
潜在空間で返す場合、`posterior.mean` は分類境界のどちら側にいるかを表すスコアである。  
確率空間で返す場合、`posterior.mean` は `p(y=1|x)` に近い意味を持つ。

#### 順序回帰

順序回帰では `y` はカテゴリであるが、カテゴリ間に順序がある。

```text
0: 悪い
1: 普通
2: 良い
```

この場合、分類と同様にカテゴリラベルを扱うが、`0 < 1 < 2` のような順序関係をモデルに反映させる。  
典型的には、潜在関数 `f(x)` と cutpoints によってカテゴリを分ける。

```text
class 0: f(x) <= c1
class 1: c1 < f(x) <= c2
class 2: c2 < f(x)
```

---

### 1.3 posterior の種類

本実装では、posterior の意味を明確に区別することが重要である。

#### latent posterior

`latent posterior` は、GP が直接モデル化している潜在関数 `f(x)` の posterior である。

```text
f(x) | D
```

分類や順序回帰では、観測ラベルそのものではなく、この潜在関数を likelihood に通すことでカテゴリ確率を得る。

latent posterior は以下の用途と相性がよい。

- 分類境界探索
- level-set estimation
- Straddle
- cutpoint 近傍探索
- latent UCB

2値分類では、latent posterior を使う場合の境界は通常 `f(x)=0` である。

#### probability posterior

`probability posterior` は、latent posterior を sigmoid、softmax、ordinal probability などに変換した後の posterior である。

2値分類では、以下のように解釈できる。

```text
posterior.mean ≈ p(y = 1 | x)
```

probability posterior は以下の用途と相性がよい。

- 良品確率の最大化
- feasibility probability
- 確率値を目的値とした Bayesian Optimization
- entropy sampling
- 確率ベースの active learning

2値分類では、probability posterior を使う場合の境界は通常 `p(y=1|x)=0.5` である。

#### utility posterior

`utility posterior` は、クラス確率や順序カテゴリ確率をスカラーの効用に変換したものである。

例えば、順序カテゴリ `0, 1, 2` に対して以下の効用値を与える。

```text
utility_values = [0.0, 0.5, 1.0]
```

各クラス確率を `p_k(x)` とすると、期待効用は次のように計算できる。

```text
E[u(y) | x] = Σ_k p_k(x) u_k
```

utility posterior は以下の用途と相性がよい。

- 順序回帰の Bayesian Optimization
- 多目的最適化
- 分類・順序回帰を連続目的値のように扱う場合
- Hybrid model における目的値変換

---

### 1.4 single-output / multi-output / hybrid model

本実装では、出力の持ち方に応じて以下のモデル構成を扱う。

#### single-output model

1つの入力 `x` に対して、1つの出力を予測するモデルである。

```text
x -> y
```

例：

- 1つの連続目的値
- 1つの良否判定
- 1つの順序評価

#### multi-output model

1つの入力 `x` に対して、複数の同種の出力を予測するモデルである。

```text
x -> [y1, y2, ..., ym]
```

例：

- 複数の連続目的値
- 複数の良否判定
- 複数の順序評価

BoTorch 的には、`ModelListGP` によって複数の single-output model をまとめる方法と、1つのモデルから複数出力を返す方法がある。

本実装では、モデルごとの柔軟性を優先し、分類や順序回帰では `ModelList` 的な構成を基本にしつつ、必要に応じて multi-output posterior を扱える objective を用意する。

#### hybrid model

hybrid model は、異なる種類のモデルを組み合わせる構成である。

```text
x -> [regression output, classification output, ordinal output]
```

例：

- 回帰 + 分類
- 回帰 + 順序回帰
- 分類 + 順序回帰
- 回帰 + 分類 + 順序回帰

Hybrid model では、各出力をそのまま同じ意味で扱うことはできない。  
そのため、獲得関数に渡す前に objective を用いて、各出力をスカラー値・確率値・期待効用などに変換する必要がある。

---

### 1.5 モデルと獲得関数の関係

モデルは、入力 `X` に対して posterior を返す。  
獲得関数は、その posterior を用いて「次に評価すべき候補点」を選択する。

```text
model.posterior(X)
        ↓
objective / posterior transform
        ↓
acquisition function
        ↓
candidate selection
```

本実装では、モデルと獲得関数の役割を以下のように分ける。

| 要素 | 役割 |
|---|---|
| Model | posterior を返す |
| Objective | posterior samples や mean を目的値空間に変換する |
| Acquisition function | 候補点の有用性をスコア化する |
| optimize_acqf | 獲得関数を最大化して候補点を得る |

---

## 2. 回帰モデルの基本概念

### 2.1 Gaussian Process Regression

Gaussian Process Regression は、連続値を予測するための基本的な GP モデルである。  
未知関数 `f(x)` に GP 事前分布を置き、観測値 `y` はその関数値にノイズが加わったものとして扱う。

```text
f(x) ~ GP(m(x), k(x, x'))
y = f(x) + ε
ε ~ N(0, σ²)
```

ここで、

- `m(x)` は平均関数
- `k(x, x')` はカーネル関数
- `ε` は観測ノイズ

である。

GP 回帰では、観測データ `D = {(x_i, y_i)}` が与えられると、未観測点 `x*` における予測分布を得ることができる。

```text
p(f(x*) | D)
```

この予測分布の平均と分散が、Bayesian Optimization や Active Learning において重要な役割を持つ。

---

### 2.2 Gaussian likelihood

標準的な GP 回帰では、観測ノイズをガウス分布として扱う。

```text
p(y | f) = N(y | f, σ²)
```

この仮定により、厳密な posterior を比較的扱いやすくなる。  
BoTorch の `SingleTaskGP` は、この標準的な GP 回帰モデルの代表例である。

---

### 2.3 posterior の解釈

回帰モデルにおいて、`posterior.mean` と `posterior.variance` は比較的直感的に解釈できる。

| 値 | 意味 |
|---|---|
| `posterior.mean` | 目的値の予測平均 |
| `posterior.variance` | 目的値の予測不確かさ |

Bayesian Optimization では、平均が高い点と不確かさが大きい点のバランスをとりながら候補点を選ぶ。

Active Learning では、主に不確かさが大きい点を選ぶ。

Level-set Estimation では、平均がしきい値に近く、かつ不確かさが大きい点を選ぶ。

---

### 2.4 Non-Gaussian Regression

現実のデータでは、観測値が連続値であってもガウス分布で表現しにくい場合がある。  
そのような場合、Gaussian likelihood ではなく、データの性質に合った likelihood を使う。

| データの種類 | 代表的 likelihood | 例 |
|---|---|---|
| カウントデータ | Poisson | 欠陥数、発生回数 |
| 過分散カウント | Negative Binomial | ばらつきの大きいカウント |
| 正の連続値 | Gamma | 寿命、強度、時間 |
| 0〜1 の比率 | Beta | 割合、確率、収率 |

Non-Gaussian regression では、latent function `f(x)` を link function に通して、likelihood のパラメータに変換する。

例として Poisson GP では、平均パラメータ `λ(x)` を正に保つ必要があるため、以下のような変換を用いる。

```text
λ(x) = exp(f(x))
```

または数値安定性のために、

```text
λ(x) = softplus(f(x))
```

を用いる場合もある。

---

### 2.5 回帰モデルと獲得関数の接続

回帰モデルは、Bayesian Optimization と最も直接的に接続できる。

#### Bayesian Optimization

目的値を最大化する場合、以下のような獲得関数を使う。

- Expected Improvement
- Log Expected Improvement
- Upper Confidence Bound
- Knowledge Gradient

#### Active Learning

モデルの予測精度を上げたい場合、以下を使う。

- posterior variance
- Integrated Posterior Variance
- qNegIntegratedPosteriorVariance

#### Level-set Estimation

しきい値 `h` に対する境界を知りたい場合、以下を使う。

- Straddle
- ICU
- Boundary Variance

---

### 2.6 回帰モデルで注意すべき点

#### 観測ノイズ込みか、潜在関数か

回帰では、以下の2種類を区別する必要がある。

```text
f(x): ノイズなしの潜在関数
y(x): ノイズ込みの観測値
```

Bayesian Optimization では、通常はノイズなしの潜在関数 `f(x)` を最適化したい。  
一方、観測予測や不確かさ評価では、ノイズ込みの予測が必要な場合もある。

#### `best_f` の扱い

EI 系の獲得関数では、現在までの最良値 `best_f` が必要になる。  
回帰では、観測値の最大値を使う場合もあれば、モデル posterior mean の最大値を使う場合もある。

ノイズが大きい場合、単純に観測値の最大値を使うと不安定になるため、posterior mean ベースの `best_f` を使う方が安定する場合がある。

---

## 3. 分類モデルの基本概念

### 3.1 分類問題の整理

分類モデルでは、出力 `y` は連続値ではなくカテゴリラベルである。  
本実装では、分類を以下のように整理する。

| 分類形式 | 出力 | 例 |
|---|---|---|
| Binary classification | 1つの 0/1 | 良品 / 不良 |
| Multi-class classification | 1つのクラス | A / B / C のどれか |
| Multi-label classification | 複数の 0/1 | 複数条件を同時に満たすか |
| Multi-output classification | 複数の分類タスク | 複数品質項目の合否 |

#### Binary classification

2値分類では、出力は `0` または `1` である。

```text
y ∈ {0, 1}
```

実装上は、潜在関数 `f(x)` を sigmoid 関数に通して、`p(y=1|x)` を得る。

#### Multi-class classification

多クラス分類では、出力は複数クラスのうち1つである。

```text
y ∈ {0, 1, ..., K-1}
```

各クラスに対して確率を返し、最も確率が高いクラスを予測クラスとする。

#### Multi-label classification

Multi-label classification では、1つの入力に対して複数の 0/1 ラベルが対応する。

```text
x -> [y1, y2, ..., ym]
yi ∈ {0, 1}
```

これは複数の binary classification を同時に行う問題として扱える。

#### Multi-output classification

Multi-output classification は、複数の分類出力を持つ広い概念である。  
multi-label classification も multi-output classification の一種として捉えられる。

---

### 3.2 Gaussian Process Classifier

Gaussian Process Classifier は、分類ラベルを直接 GP で回帰するのではなく、まず潜在関数 `f(x)` に GP 事前分布を置き、その値を確率に変換する。

2値分類では、典型的には次のように表現する。

```text
f(x) ~ GP(m(x), k(x, x'))
p(y = 1 | x) = sigmoid(f(x))
```

ここで、`f(x)` は分類境界に対する潜在スコアである。  
`f(x)` が大きいほど `y=1` の確率が高く、`f(x)` が小さいほど `y=0` の確率が高い。

---

### 3.3 Bernoulli likelihood

2値分類では、観測ラベル `y` は Bernoulli 分布に従うと考える。

```text
p(y | f) = Bernoulli(sigmoid(f))
```

回帰の Gaussian likelihood と異なり、Bernoulli likelihood は Gaussian likelihood ではない。  
そのため、分類 GP では posterior を厳密に閉形式で扱うことが難しく、近似推論が必要になる。

実装上は、以下のような近似推論が使われる。

- Laplace approximation
- Expectation Propagation
- Variational inference
- Predictive likelihood approximation

BoTorch / GPyTorch ベースの実装では、分類モデルを近似 GP として構成し、`VariationalELBO` などを使って学習することが多い。

---

### 3.4 latent posterior と probability posterior

分類モデルでは、posterior の空間を明確に区別する必要がある。

#### latent posterior

latent posterior は、潜在関数 `f(x)` の posterior である。

```text
posterior.mean = E[f(x)]
posterior.variance = Var[f(x)]
```

この場合、2値分類の境界は通常 `f(x)=0` である。

latent posterior は、以下の用途に向いている。

- Straddle
- Level-set Estimation
- 境界探索
- latent UCB

#### probability posterior

probability posterior は、latent posterior を sigmoid で確率に変換した後の posterior である。

```text
posterior.mean ≈ p(y = 1 | x)
```

この場合、2値分類の境界は通常 `p(y=1|x)=0.5` である。

probability posterior は、以下の用途に向いている。

- 良品確率の最大化
- Probability of Feasibility
- 分類確率を目的値とする Bayesian Optimization
- entropy sampling
- BALD

---

### 3.5 Multi-class GP Classifier

多クラス分類では、入力 `x` に対して複数クラスの確率を返す。

```text
p(y = k | x), k = 0, 1, ..., K-1
```

実装方法としては、以下のような方針がある。

#### one-vs-rest 型

各クラスに対して binary classifier を作り、それぞれのクラスらしさを推定する。

```text
class 0 vs rest
class 1 vs rest
class 2 vs rest
```

この方法は実装しやすいが、各クラス確率の整合性を別途考える必要がある。

#### multi-class latent 型

各クラスに対応する latent function を持ち、softmax などでクラス確率に変換する。

```text
f_k(x) ~ GP
p(y=k|x) = softmax(f_1(x), ..., f_K(x))_k
```

この方法は多クラス分類として自然だが、実装と推論は複雑になる。

#### Dirichlet 的な近似

クラス確率ベクトルを Dirichlet 分布的に扱うことで、クラス確率とその不確かさを表現する方法もある。  
本実装では、多クラス classification の posterior を BoTorch の獲得関数に接続するために、posterior samples の shape と確率制約を明確に扱う必要がある。

---

### 3.6 `n_classes` / `n_components` の整理

多クラス分類では、以下の2つを混同しないようにする。

| 用語 | 意味 |
|---|---|
| `n_classes` | 観測ラベルとして存在するクラス数 |
| `n_components` | モデル内部で使う潜在成分数 |

多くの分類実装では、ユーザーが明示的に指定したいのは `n_classes` である。  
一方、内部的に Dirichlet 近似や latent component を使う場合は、`n_components` が必要になることがある。

実装 API としては、外部からは `num_classes` または `n_classes` を指定し、内部で必要に応じて `n_components` を決める設計が分かりやすい。

---

### 3.7 Multi-label / Multi-output Classification

Multi-label classification は、複数の binary classification を同時に扱う問題として実装できる。

```text
x -> [p1, p2, ..., pm]
```

ここで、`pj` は j 番目のラベルが 1 である確率を表す。

本実装では、以下のような統合方法を想定する。

#### weighted sum

複数の分類確率を重み付き和でまとめる。

```text
score(x) = Σ_j w_j p_j(x)
```

#### product / all-positive

すべての条件を満たす確率に近いスコアとして、確率の積を使う。

```text
score(x) = Π_j p_j(x)
```

#### minimum

最も悪い分類確率を重視する。

```text
score(x) = min_j p_j(x)
```

このような変換は、`MultiOutputClassificationScoreObjective` のような objective で扱うと、獲得関数側をシンプルに保てる。

---

### 3.8 分類モデルと獲得関数の接続

分類モデルは、以下のような獲得関数と接続される。

| 目的 | 代表的獲得関数 | posterior の使い方 |
|---|---|---|
| 分類器を改善したい | BALD / Entropy / Uncertainty Sampling | probability posterior |
| 境界を探したい | Straddle / ICU | latent posterior または probability posterior |
| 良品確率を最大化したい | EI / UCB / PoF | probability posterior |
| 複数分類目的を最適化したい | EHVI / NParEGO | multi-output probability objective |

#### BALD

BALD は、予測ラベルとモデルパラメータの相互情報量を使って、モデルにとって情報量の多い点を選ぶ。

#### Straddle

Straddle は、平均が境界に近く、かつ不確かさが大きい点を選ぶ。  
分類では、latent 空間なら `f(x)=0`、確率空間なら `p(y=1|x)=0.5` が境界になる。

---

## 4. 順序回帰モデルの基本概念

### 4.1 順序回帰とは

順序回帰は、分類と回帰の中間的な問題である。  
出力はカテゴリラベルだが、カテゴリ間に順序がある。

例：

```text
0: 悪い
1: 普通
2: 良い
```

この場合、`0` と `2` は単なる別カテゴリではなく、`0 < 1 < 2` という順序関係を持つ。

分類モデルではクラス間の順序を考慮しないことが多いが、順序回帰ではこの順序情報をモデル化する。

---

### 4.2 Ordinal GP

Ordinal GP では、潜在関数 `f(x)` に GP 事前分布を置き、cutpoints によってクラスを分ける。

```text
f(x) ~ GP(m(x), k(x, x'))
```

3クラスの例では、次のように表せる。

```text
class 0: f(x) <= c1
class 1: c1 < f(x) <= c2
class 2: c2 < f(x)
```

ここで、`c1` と `c2` が cutpoints である。

---

### 4.3 ordinal likelihood

Ordinal likelihood は、潜在関数 `f(x)` と cutpoints から各クラスの確率を計算する。

例えば、probit 型の ordinal likelihood では、標準正規分布の累積分布関数 `Φ` を用いて、各クラス確率を表す。

```text
p(y = 0 | f) = Φ(c1 - f)
p(y = 1 | f) = Φ(c2 - f) - Φ(c1 - f)
p(y = 2 | f) = 1 - Φ(c2 - f)
```

このように、ordinal GP では latent function の値が大きくなるほど、より高いクラスに入る確率が高くなる。

---

### 4.4 class probability

順序回帰では、最終的な予測として各クラスの確率を得ることが重要である。

```text
[p(y=0|x), p(y=1|x), p(y=2|x)]
```

このクラス確率は、以下の用途に使える。

- 最も確率が高いクラスを予測する
- 期待効用を計算する
- entropy を計算する
- cutpoint 近傍の不確かさを評価する

---

### 4.5 expected utility

順序回帰を Bayesian Optimization に接続する場合、カテゴリラベルをそのまま最適化するよりも、期待効用として連続値化する方が扱いやすい。

クラスごとの効用値を次のように定義する。

```text
utility_values = [u0, u1, u2]
```

各クラス確率を使って期待効用を計算する。

```text
E[u(y)|x] = p0(x)u0 + p1(x)u1 + p2(x)u2
```

例えば、

```text
utility_values = [0.0, 0.5, 1.0]
```

とすれば、高いクラスほど良いという目的値に変換できる。

---

### 4.6 Ordinal posterior の解釈

Ordinal GP では、posterior を以下の複数の形で解釈できる。

| posterior の種類 | 意味 | 主な用途 |
|---|---|---|
| latent posterior | `f(x)` の平均・分散 | cutpoint 境界探索 |
| class probability | 各クラスの確率 | 分類予測、entropy |
| expected utility | クラス確率から計算した効用 | Bayesian Optimization |
| predicted class | 最も確率が高いクラス | 最終予測 |

この区別を曖昧にすると、獲得関数の閾値や目的値の意味がずれる。  
特に、Straddle や Level-set Estimation では latent 空間の cutpoint を使うのか、expected utility のしきい値を使うのかを明確にする必要がある。

---

### 4.7 順序回帰モデルと獲得関数の接続

| 目的 | 代表的獲得関数 | 使う量 |
|---|---|---|
| 高い順序評価を探す | EI / UCB / KG | expected utility |
| 順序分類器を改善する | entropy / BALD 的指標 | class probability |
| クラス境界を探す | Straddle / ICU | latent posterior + cutpoints |
| 特定クラス領域を探す | Level-set Estimation | class probability または utility |

---

## 5. 拡張モデルの全体像

### 5.1 拡張モデルを導入する理由

基本的な GP、GPC、Ordinal GP だけでも多くの問題を扱える。  
しかし、実データでは以下のような課題が生じる。

| 課題 | 対応する拡張モデル |
|---|---|
| 非線形性が強い | Deep Kernel GP / Deep GP |
| 入力ごとにノイズが違う | Heteroscedastic GP |
| 入力次元が高い | SAAS / RRP / PCA / REMBO |
| 重要な変数が一部だけ | SAAS / RRP |
| 表現力が不足する | Deep GP |
| 特徴抽出を学習したい | Deep Kernel GP |
| 候補点探索を低次元化したい | REMBO |

拡張モデルは、回帰・分類・順序回帰のいずれにも適用できる場合がある。  
ただし、分類や順序回帰では likelihood や posterior 変換が複雑になるため、回帰モデルよりも shape や posterior の扱いに注意が必要である。

---

### 5.2 表現力を高めるモデル

表現力を高める目的では、主に以下を使う。

- Deep Kernel GP
- Deep GP

Deep Kernel GP は、ニューラルネットワークによる特徴抽出と GP を組み合わせる。  
Deep GP は、GP 自体を多層化する。

---

### 5.3 ノイズを扱うモデル

入力点ごとに観測ノイズが異なる場合は、Heteroscedastic GP を使う。

特に、Active Learning では「ノイズが大きい点」と「モデルが知らない点」を区別することが重要である。  
ノイズが大きいだけの点を選び続けると、追加観測してもモデル改善につながりにくい可能性がある。

---

### 5.4 高次元・変数選択を扱うモデル

高次元入力を扱う場合は、以下の方針がある。

| 方針 | 代表モデル |
|---|---|
| 重要変数を sparse prior で表現 | SAAS |
| 重要変数を探索・選択 | RRP |
| データから低次元表現を学習 | PCA |
| ランダム低次元空間で BO | REMBO |

---

## 6. Deep Kernel GP

### 6.1 基本概念

Deep Kernel GP は、ニューラルネットワークで入力特徴量を変換し、その変換後の特徴空間で GP を構築するモデルである。

```text
z = NN(x)
f(z) ~ GP
```

通常の GP では、カーネルは元の入力空間 `x` 上で定義される。  
Deep Kernel GP では、ニューラルネットワークで得た特徴量 `z` 上でカーネルを定義する。

これにより、元の入力空間では単純なカーネルで表現しにくい複雑な構造を、特徴空間では扱いやすくできる。

---

### 6.2 通常 GP との違い

| 観点 | 通常 GP | Deep Kernel GP |
|---|---|---|
| 入力 | 元の `x` | NN で変換した `z` |
| 表現力 | カーネルに依存 | NN + カーネル |
| 学習対象 | カーネルハイパーパラメータ | NN パラメータ + GP ハイパーパラメータ |
| 向く問題 | 小〜中規模、滑らかな関数 | 非線形性が強い問題 |

---

### 6.3 回帰への適用

回帰では、Deep Kernel GP は以下のように構成される。

```text
x -> feature extractor -> z -> Gaussian GP -> y
```

`posterior.mean` は目的値の予測平均、`posterior.variance` は目的値の不確かさとして扱える。

通常 GP よりも柔軟な非線形性を表現できる一方で、ニューラルネットワーク部分の学習が不安定になる場合がある。

---

### 6.4 分類への適用

分類では、Deep Kernel GP の出力を latent score として扱い、sigmoid や softmax によって確率へ変換する。

2値分類の例：

```text
x -> feature extractor -> z -> latent GP -> f(z)
p(y=1|x) = sigmoid(f(z))
```

実装上は、`forward` が latent distribution を返し、`posterior` が probability posterior を返す設計にすると、BoTorch の獲得関数と接続しやすい。

---

### 6.5 順序回帰への適用

順序回帰では、Deep Kernel GP の latent output を cutpoints と組み合わせてクラス確率を計算する。

```text
x -> feature extractor -> z -> latent GP -> f(z)
f(z) + cutpoints -> class probabilities
```

expected utility を計算すれば、Bayesian Optimization にも接続できる。

---

### 6.6 実装上の注意

#### `forward` と `posterior`

Deep Kernel GP では、`forward` と `posterior` の役割を分けると整理しやすい。

| メソッド | 役割 |
|---|---|
| `forward` | 学習用の latent distribution を返す |
| `posterior` | BoTorch 獲得関数用の posterior を返す |

分類や順序回帰では、`forward` は latent distribution、`posterior` は probability posterior または utility posterior を返す設計が考えられる。

#### input_transform との関係

`input_transform` を使う場合、NN に入る前に変換するのか、GP に入る前に変換するのかを明確にする必要がある。

一般には、BoTorch 互換性を考えると、raw input `X` を `posterior(X)` に渡し、モデル内部で input transform と feature extractor を適用する構成が分かりやすい。

#### 学習安定性

Deep Kernel GP は NN と GP を同時に学習するため、通常 GP よりも最適化が不安定になる場合がある。  
学習率、初期値、特徴量次元、正則化に注意する。

---

## 7. Deep GP

### 7.1 基本概念

Deep GP は、GP を多層に積み重ねたモデルである。

```text
h1(x) ~ GP
h2(h1) ~ GP
f(h2) ~ GP
```

Deep Kernel GP が「NN + GP」であるのに対し、Deep GP は「GP + GP + ...」という構成である。

---

### 7.2 Deep Kernel GP との違い

| 観点 | Deep Kernel GP | Deep GP |
|---|---|---|
| 構成 | NN + GP | GP の多層構造 |
| 中間表現 | deterministic な NN 特徴量 | 確率的な GP 出力 |
| 推論 | 比較的扱いやすい | 近似推論が必須 |
| 不確かさ | 最終 GP で表現 | 各層で不確かさを伝播 |
| 実装難度 | 中 | 高 |

Deep GP は、層ごとに不確かさを持つため表現力が高いが、学習と posterior の扱いが複雑になる。

---

### 7.3 回帰への適用

回帰では、Deep GP の最終出力を連続値の潜在関数として扱う。

```text
x -> GP layer 1 -> GP layer 2 -> f(x)
y = f(x) + ε
```

学習には、通常の exact MLL ではなく、変分推論に基づく目的関数を使う。

代表的には、

- `VariationalELBO`
- `DeepApproximateMLL`

などを使う。

---

### 7.4 分類への適用

分類では、Deep GP の最終出力を latent score として扱う。

```text
x -> Deep GP -> f(x)
p(y=1|x) = sigmoid(f(x))
```

2値分類では Bernoulli likelihood と組み合わせる。  
multi-class classification では、複数の latent output を softmax 的に扱う設計が考えられる。

---

### 7.5 順序回帰への適用

順序回帰では、Deep GP の latent output を cutpoints によってクラス確率へ変換する。

```text
x -> Deep GP -> f(x)
f(x) + cutpoints -> ordinal probabilities
```

Deep GP を使うことで複雑な順序境界を表現できる可能性があるが、学習が不安定になりやすく、cutpoints の推定にも注意が必要である。

---

### 7.6 実装上の注意

#### variational inference

Deep GP では exact posterior を扱うことが難しいため、変分推論を用いる。  
そのため、`mll` として exact GP 用の `ExactMarginalLogLikelihood` ではなく、Deep GP に対応した MLL を使う必要がある。

#### posterior shape

Deep GP の posterior samples は、通常の GP よりも shape が複雑になりやすい。  
特に、以下の次元を明確に管理する必要がある。

```text
sample_shape x batch_shape x q x output_dim
```

#### BoTorch acquisition との接続

BoTorch の獲得関数に接続するためには、`posterior(X)` が BoTorch 互換の posterior オブジェクトを返す必要がある。  
分類や順序回帰では、latent posterior をそのまま返すか、確率・期待効用に変換して返すかを明確にする。

---

## 8. Heteroscedastic GP

### 8.1 基本概念

通常の GP 回帰では、観測ノイズの分散 `σ²` は一定と仮定する。

```text
y = f(x) + ε
ε ~ N(0, σ²)
```

一方、Heteroscedastic GP では、ノイズ分散が入力 `x` に依存すると考える。

```text
y = f(x) + ε(x)
ε(x) ~ N(0, σ²(x))
```

これにより、ある領域では観測が安定しているが、別の領域ではばらつきが大きい、といった状況を表現できる。

---

### 8.2 回帰への適用

回帰では、Heteroscedastic GP は次の2つの要素を持つ。

| 要素 | 役割 |
|---|---|
| mean model | 潜在関数 `f(x)` を推定する |
| noise model | 入力依存ノイズ `σ²(x)` を推定する |

観測値のばらつきが入力に依存する場合、通常の等分散 GP よりも現実的な不確かさ評価が可能になる。

---

### 8.3 分類への適用

分類における heteroscedasticity は、ラベルノイズや判定の曖昧さとして現れる。  
例えば、ある領域ではラベルが安定しているが、別の領域では同じ入力条件でもラベルがばらつく場合がある。

ただし、分類においては以下を区別する必要がある。

| 不確かさ | 意味 |
|---|---|
| epistemic uncertainty | データ不足によるモデルの不確かさ |
| aleatoric uncertainty | データ生成過程そのもののノイズ |
| boundary uncertainty | 分類境界付近にいることによる曖昧さ |

Active Learning では、基本的には epistemic uncertainty が大きい点を選ぶ方が、モデル改善につながりやすい。

---

### 8.4 順序回帰への適用

順序回帰では、heteroscedasticity は順序ラベルのばらつきとして現れる。

例えば、同じ入力条件でも評価者や測定条件によって、クラス 1 とクラス 2 の間でラベルが揺れる場合がある。

Heteroscedastic ordinal model では、latent score の不確かさに加えて、入力依存のラベルノイズや cutpoint 近傍の不安定性を考慮できる。

---

### 8.5 Active Learning での利用

Heteroscedastic model を Active Learning に使う場合、単純に予測分散が大きい点を選ぶと、ノイズが大きいだけの点を選んでしまう可能性がある。

そのため、本実装では以下のような考え方を取る。

```text
acquisition_score = epistemic_uncertainty - noise_penalty * predicted_noise
```

このように noise penalty を入れることで、観測しても改善しにくい高ノイズ領域を避けやすくなる。

---

### 8.6 実装上の注意

#### noise が log variance か variance か

ノイズモデルの出力が `log variance` なのか `variance` なのかを明確にする必要がある。

```text
noise_is_log_var = True
```

のようなフラグを持たせる場合、獲得関数側で正しく variance に変換する必要がある。

#### default noise

ノイズモデルを持たないモデルにも hetero 系獲得関数を適用したい場合、`default_sigma` や `default_noise` を用意しておくと汎用性が高くなる。

---

## 9. Robust Relevance Pursuit

### 9.1 基本概念

Robust Relevance Pursuit は、多数の入力変数の中から、目的値や分類境界に本当に関係する変数を抽出しながら GP を構築するための考え方である。

高次元データでは、すべての変数が出力に効いているとは限らない。  
irrelevant feature が多いと、GP のカーネル学習が不安定になり、Bayesian Optimization の候補点探索も難しくなる。

RRP は、このような状況で重要変数を見つけ、不要な変数の影響を抑えることを目的とする。

---

### 9.2 高次元問題での課題

高次元入力では、以下の問題が起きやすい。

- 距離計算が不安定になる
- カーネルの lengthscale 学習が難しくなる
- irrelevant feature によって posterior がぼやける
- 獲得関数の最適化が難しくなる
- 候補点が高次元空間に散らばりすぎる

そのため、重要変数の抽出や次元削減が重要になる。

---

### 9.3 回帰への適用

回帰では、RRP によって目的値に効く変数を抽出する。

```text
x = [x1, x2, ..., xd]
relevant variables = [x2, x5, x9]
```

重要変数を特定できれば、以下のメリットがある。

- モデル解釈性が上がる
- GP の学習が安定する
- BO の探索効率が上がる
- 不要な変数方向への探索を減らせる

---

### 9.4 分類への適用

分類では、RRP によって分類境界に効く変数を抽出する。

例えば良否分類では、すべてのプロセス変数が良否に効いているとは限らない。  
RRP により、良品 / 不良の境界を決める主要因子を絞り込める。

分類モデルで RRP を使う場合、重要度は latent score やクラス確率に対して評価することになる。

---

### 9.5 順序回帰への適用

順序回帰では、RRP によって順序スコアに効く変数を抽出する。

例えば、品質ランク `0, 1, 2` がある場合、RRP はランクを上げ下げする主要因子を見つけるために使える。

ordinal model では、重要変数が以下のどれに効いているかを区別すると解釈しやすい。

- latent score
- cutpoint 近傍の不確かさ
- expected utility
- predicted class

---

### 9.6 SAAS との違い

RRP と SAAS はどちらも高次元問題で使われるが、目的と使い方が異なる。

| 観点 | RRP | SAAS |
|---|---|---|
| 主目的 | 関連変数の抽出・削減 | sparse prior による高次元 BO |
| 変数選択 | 明示的に重要変数を扱う | lengthscale の sparse prior により暗黙的に扱う |
| 解釈性 | 高い | 中程度 |
| 推論 | 実装依存 | fully Bayesian が中心 |
| 向く用途 | 特徴量選択・解釈・モデル簡略化 | 高次元 Bayesian Optimization |

---

## 10. SAAS GP

### 10.1 基本概念

SAAS は Sparse Axis-Aligned Subspace の略であり、高次元入力のうち少数の変数だけが目的関数に効いているという仮定を置く GP モデルである。

高次元 Bayesian Optimization では、すべての次元を同じように探索すると効率が悪い。  
SAAS GP では、ARD lengthscale に sparse prior を置くことで、重要な変数だけが短い lengthscale を持ち、重要でない変数は長い lengthscale を持つように誘導する。

---

### 10.2 高次元 BO での位置づけ

SAAS GP は、高次元入力を持つが、実際には少数の変数だけが効いている問題に向いている。

```text
d is large
effective dimension is small
```

このような問題では、SAAS GP によって有効次元を暗黙的に推定しながら BO を進めることができる。

---

### 10.3 回帰への適用

BoTorch では、SAAS は主に fully Bayesian GP として実装される。  
代表的には `SaasFullyBayesianSingleTaskGP` のようなモデルがある。

推論には NUTS などの MCMC が使われることが多く、通常の exact GP より計算コストは高い。

一方で、高次元 BO では有効な選択肢になる。

---

### 10.4 分類・順序回帰への拡張

分類や順序回帰に SAAS 的な考え方を適用する場合、latent function に対するカーネルの lengthscale に sparse prior を置く構成が考えられる。

2値分類では、

```text
f(x) ~ SAAS GP
p(y=1|x) = sigmoid(f(x))
```

順序回帰では、

```text
f(x) ~ SAAS GP
f(x) + cutpoints -> ordinal probabilities
```

のように扱う。

ただし、分類・順序回帰では Gaussian likelihood ではないため、推論と posterior 変換が複雑になる。  
そのため、回帰の SAAS GP よりも実装上の注意が多い。

---

### 10.5 fully Bayesian 推論と shape の注意

SAAS GP では、ハイパーパラメータの posterior samples を持つため、通常の GP よりも posterior の batch shape が増えることがある。

そのため、獲得関数や objective に渡す前に、以下を確認する必要がある。

```text
posterior.mean.shape
posterior.variance.shape
samples.shape
```

特に multi-output や ModelList と組み合わせる場合、extra batch dimension を適切に reduce する処理が必要になることがある。

---

### 10.6 RRP との使い分け

SAAS は、重要変数を sparse prior によって暗黙的に扱う。  
RRP は、重要変数をより明示的に抽出・削減する。

| 状況 | 推奨 |
|---|---|
| 高次元 BO をそのまま進めたい | SAAS |
| 重要変数を明示的に解釈したい | RRP |
| 変数を削減して再学習したい | RRP |
| fully Bayesian に不確かさを扱いたい | SAAS |
| 計算コストを抑えたい | RRP または PCA / REMBO |

---

## 11. PCA / REMBO

### 11.1 次元削減モデルの目的

高次元入力をそのまま GP に入れると、以下の問題が生じる。

- カーネル学習が難しい
- 距離が意味を持ちにくい
- BO の候補点最適化が難しい
- 必要な初期データ数が増える
- posterior の不確かさが大きくなりやすい

このような場合、入力空間を低次元潜在空間に変換してから GP を構築する。

```text
x ∈ R^d
z ∈ R^r, r << d
```

---

### 11.2 PCA GP

PCA GP では、データから主成分を学習し、入力を低次元空間へ写像する。

```text
z = PCA(x)
f(z) ~ GP
```

PCA は線形次元削減であり、データの分散が大きい方向を主成分として抽出する。

#### 回帰への適用

回帰では、PCA 後の潜在変数 `z` を入力として GP 回帰を行う。

```text
x -> PCA -> z -> GP -> y
```

#### 分類への適用

分類では、PCA 後の潜在変数 `z` を入力として GPC を構築する。

```text
x -> PCA -> z -> latent GP -> p(y=1|x)
```

#### 順序回帰への適用

順序回帰では、PCA 後の潜在変数 `z` を入力として ordinal GP を構築する。

```text
x -> PCA -> z -> latent GP -> ordinal probabilities
```

---

### 11.3 REMBO GP

REMBO は Random EMbedding Bayesian Optimization の略であり、高次元空間の最適化を低次元ランダム空間で行う方法である。

低次元潜在変数 `z` をランダム射影行列 `A` によって元空間へ写像する。

```text
x = A z
```

そして、BO は `z` 空間で行う。

```text
z -> x = A z -> f(x)
```

高次元空間そのものを直接探索するのではなく、低次元空間で候補点を探索することで、探索効率を上げることを狙う。

---

### 11.4 PCA と REMBO の違い

| 観点 | PCA | REMBO |
|---|---|---|
| 次元削減方法 | データから主成分を学習 | ランダム射影 |
| 主用途 | データ構造の低次元表現 | 高次元 BO |
| 解釈性 | 比較的高い | 低い |
| 元空間への写像 | PCA inverse transform | random projection |
| データ依存性 | あり | 基本的にランダム |
| 向く場面 | 既存データに低次元構造がある | 有効次元が低いが方向が不明 |

---

### 11.5 raw input / latent input の注意

PCA / REMBO 系モデルでは、raw input と latent input の区別が重要である。

```text
raw input: 元の入力空間の X
latent input: PCA / REMBO 後の Z
```

実装上は、以下を明確にする必要がある。

| 変数 | 意味 |
|---|---|
| `train_inputs_raw` | 次元削減前の元の入力 |
| `train_inputs` | モデル内部で使う入力 |
| `latent_X` | 低次元潜在空間の入力 |
| `X` | 外部 API から渡される raw input |

BoTorch 互換性を考えると、ユーザーは基本的に raw input `X` を `posterior(X)` に渡し、モデル内部で latent space に変換する設計が分かりやすい。

---

### 11.6 categorical variables がある場合

PCA や REMBO を categorical variables に直接適用すると、カテゴリの意味が崩れる可能性がある。

そのため、連続変数とカテゴリ変数が混在する場合は、以下のような方針が必要になる。

1. 連続変数だけを PCA / REMBO に通す
2. カテゴリ変数はそのまま保持する
3. 潜在連続変数とカテゴリ変数を結合して GP に渡す

```text
x = [x_continuous, x_categorical]

z = PCA(x_continuous)

model input = [z, x_categorical]
```

この設計により、カテゴリ変数を壊さずに高次元連続変数だけを低次元化できる。

---

## 12. 拡張モデルの使い分け

### 12.1 目的別の選び方

| 状況 | 推奨モデル |
|---|---|
| 標準的な連続値回帰 | Gaussian GP |
| カウント・正値・比率データ | Non-Gaussian GP |
| 2値分類 | Binary GPC |
| 多クラス分類 | Multi-class GPC |
| 順序付きカテゴリ | Ordinal GP |
| 非線形性が強い | Deep Kernel GP |
| 通常 GP では表現力が不足 | Deep GP |
| 入力ごとにノイズが異なる | Heteroscedastic GP |
| 高次元で重要変数が少ない | SAAS / RRP |
| 重要変数を明示的に抽出したい | RRP |
| 次元削減して扱いたい | PCA GP |
| 高次元 BO を低次元空間で行いたい | REMBO |

---

### 12.2 タスク別に使える拡張モデル

| 拡張モデル | 回帰 | 分類 | 順序回帰 | 備考 |
|---|---:|---:|---:|---|
| Deep Kernel GP | ○ | ○ | ○ | feature extractor + GP |
| Deep GP | ○ | ○ | ○ | 変分推論と shape に注意 |
| Heteroscedastic GP | ○ | ○ | ○ | noise model の設計が重要 |
| RRP | ○ | ○ | ○ | 重要変数抽出に有用 |
| SAAS GP | ○ | △ | △ | 分類・順序回帰では custom posterior に注意 |
| PCA GP | ○ | ○ | ○ | raw / latent input の管理が重要 |
| REMBO GP | ○ | ○ | ○ | 元空間への写像と bounds に注意 |

`△` は実装可能だが、標準的な Gaussian regression よりも custom likelihood、posterior 変換、shape 処理への注意が必要であることを表す。

---

### 12.3 拡張モデルと獲得関数の関係

拡張モデルを使っても、最終的には獲得関数に渡す posterior の形を BoTorch 互換にそろえる必要がある。

| モデル | 獲得関数に渡す主な量 |
|---|---|
| 回帰 GP | mean / variance / posterior samples |
| Binary GPC | probability または latent score |
| Multi-class GPC | class probabilities |
| Ordinal GP | class probabilities / expected utility / latent score |
| Heteroscedastic GP | epistemic uncertainty + noise estimate |
| Deep GP | posterior samples |
| SAAS GP | extra batch dimension を含む posterior |
| PCA / REMBO GP | raw X を内部で latent X に変換した posterior |

---

### 12.4 実装上の注意点まとめ

#### posterior の意味を明示する

分類や順序回帰では、`posterior.mean` が何を意味するかを必ず明確にする。

```text
latent score なのか
probability なのか
expected utility なのか
```

#### `forward` と `posterior` を分ける

学習用の `forward` と、BoTorch 獲得関数用の `posterior` は役割が異なる。

| メソッド | 主な用途 |
|---|---|
| `forward` | MLL / ELBO による学習 |
| `posterior` | 獲得関数・予測・候補点探索 |

#### shape を確認する

特に以下の次元に注意する。

```text
sample_shape
batch_shape
q
n_w
output_dim
num_classes
```

#### raw input と transformed input を区別する

PCA、REMBO、input_transform、Deep Kernel では、どの空間の入力を扱っているかが重要である。

```text
raw input
normalized input
latent input
feature input
```

#### ModelList では出力ごとの意味をそろえる

ModelList を使う場合、各モデルの posterior の意味が異なる可能性がある。  
そのため、獲得関数に渡す前に objective で意味をそろえる。

---

## 13. Part I のまとめ

本 Part では、モデルを以下の流れで整理した。

1. 回帰・分類・順序回帰という基本タスクを整理した。
2. posterior を latent / probability / utility に分けて説明した。
3. 回帰では連続値の平均・分散が直接的な意味を持つことを確認した。
4. 分類では latent score と class probability の違いが重要であることを確認した。
5. 順序回帰では latent score、cutpoints、class probability、expected utility の関係を整理した。
6. Deep Kernel GP、Deep GP、Heteroscedastic GP、RRP、SAAS、PCA / REMBO を横断的な拡張モデルとして説明した。
7. 各拡張モデルを回帰・分類・順序回帰にどう適用するかを整理した。

以降の Part では、これらのモデルが返す posterior をどのように objective に変換し、Bayesian Optimization、Active Learning、Level-set Estimation の獲得関数に接続するかを説明する。

---

## 参考文献

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Williams, C. K. I., & Barber, D. (1998). Bayesian Classification with Gaussian Processes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). Deep Kernel Learning.
- Damianou, A., & Lawrence, N. D. (2013). Deep Gaussian Processes.
- Eriksson, D., Jankowiak, M. (2021). High-dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces.
- Wang, Z., Zoghi, M., Hutter, F., Matheson, D., & de Freitas, N. (2016). Bayesian Optimization in a Billion Dimensions via Random Embeddings.
